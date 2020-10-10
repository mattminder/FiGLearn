import torch
import numpy as np
import pandas as pd
from helpers import to_torch, vals_to_A, vals_to_L, symsqrt, A_to_L
from generators import filter_matrix #, filter_matrix_nnet
from NNet import NNet
from copy import deepcopy
from matplotlib import pyplot as plt


class NeurIMP:
    
    
    def __init__(self, graph=None, h=None, seed=42):
        """
        graph (None or torch.Tensor or np.ndarray):
            If None, the graph is initialized as random matrix. Use this when
            you are planning to learn the graph from scratch.
            If array, it can be one of the following:
            - a 2D array containing the adjacency matrix
            - a 2D array containing the graph Laplacian 
            - a 1D array containing logit values of the upper 
              triangle of the adjacency matrix
            Which of the three is provided is determined automatically 
            by the shape and whether negative values are present in the array.
        h (None or callable):
            If None, a feed-forward neural net with 5 hidden layers with 30 neurons 
            each is instantiated. 
            If you want to use a custom model architecture, set h to your torch model.
            If you want to use a custom filter function (not necessarily a neural net),
            you can provide it here. Note that in that case you have to set learn_h to 
            False when fitting the graph. The function must take a torch.Tensor as input and
            return a torch.Tensor of the same size, and must map positive real values to 
            positive real values.
        seed (int):
            Random seed for graph and h instantiation
        """
        
        _seed(seed)
        self.__init_graph(graph)
        self.__init_h(h)
        self.optim_h = None 
        self.optim_L = None 
        self.loss_hist = []
        self.__h_of_L = None

    def fit_graph(self, mat, lr_L=1e-2, lr_h=1e-4, nit=3000, nit_h_per_iter=3,
                  learn_h=True, mat_is_cov=False, fine_tune=False,
                  seed=23, verbose=100,):
        """
        Fits graph and filter to input signals.
        
        mat (np.ndarray):
            Either the signal to fit where columns correspond to graph nodes
            (if input_is_cov set to False) or an empirical covariance matrix of
            the signal (if input_is_cov set to True)        
        lr_L (float):
            Learning rate for the graph Laplacian update
        lr_h (float):
            Learning rate for the filter update
        nit (int):
            Number of total iterations
        nit_h_per_iter (int):
            Number of updates to the filter per Laplacian update. 
        learn_h (bool):
            Whether to learn the filter h. Set this to False if the object
            was created with a custom filter function h.
        mat_is_cov (bool):
            Specifies whether mat is covariance matrix or signal matrix
        fine_tune (bool):
            Whether to continue optimization for further fine-tuning the model. In
            this case no new optimizers are created.
        seed (int):
            Random seed
        verbose (int):
            Interval after which progress is printed. Set to zero if no output wanted.
        """
    
        _seed(seed)
        self.__h_of_L = None   # forget about previously computed h(L)

        _, d = mat.shape

        if mat_is_cov:
            cov = to_torch(mat)
        else:
            cov = to_torch(np.cov(mat.T))

        target = symsqrt(cov)

        # create new optimizers only in case we're not fine tuning
        if not fine_tune:
            self.optim_h = torch.optim.SGD(h.parameters(), lr=lr_h) 
            self.optim_L = torch.optim.Adam([self.__w], lr=lr_L) 

        for iter in range(int(nit)):
            if learn_h:
                _start_model_tracking(self.__h)
                evals, evecs = self.get_L_decomp()
                for _ in range(int(nit_h_per_iter)):
                    self.__optim_h_step(evals, evecs)
                _stop_model_tracking(self.__h)

            cost = __optim_L_step(target)
            self.loss_hist.append(cost)

            _be_verbose(iter, nit, verbose, cost)    
    
    def fit_filter(self, mat, n_iters=200, lr_nnet=1e-4, seed=42,
                   mat_is_cov=False):
        """
        Fits filter on graph topology. 

        mat (np.ndarray):
            Either the signal to fit where columns correspond to graph nodes
            (if mat_is_cov set to False) or an empirical covariance matrix 
            (if mat_is_cov set to True)
        n_iters (int):
            Number of epochs for fitting
        lr_nnet (float):
            Learning rate for neural network
        mat_is_cov (bool):
            Specifies whether input mat is covariance matrix or observed signals.
        """
        _seed(seed)
        self.__h_of_L = None   # forget about previously computed h(L)
        
        if mat_is_cov:
            cov = mat
        else:
            cov = np.cov(mat.T)
        sqrt_cov = symsqrt(to_torch(cov))
        evals, evecs = self.get_L_decomp()

        self.optim_h = torch.optim.SGD(self.h.parameters(), lr=lr_nnet)
        
        _start_model_tracking(self.__h)
        for i in range(int(n_iters)):
            self.__optim_h_step(evals, evecs)
        _stop_model_tracking(self.__h)
        
    def impute_missing(self, signal, mask=None, lr=1e-2, n_iters=1000, 
                       seed=42, verbose=None):
        """
        Imputes the missing values in a signal using the graph and 
        filter.
        
        signal (np.ndarray):
            One or more signals to be interpolated
        mask (None or np.ndarray):
            Values to be considered during the imputation. If is 1D vector,
            the same mask will be used for all signals. If it is None, 
            all np.nan values are masked.
        lr (float):
            Learning rate
        n_iters (int):
            Number of iterations
        seed (int):
            Random initialization of generator
        verbose (int):
            Interval after which progress is printed. Set to zero if no output wanted.
        """
        _seed(seed)
        
        def impute_single_signal(signal, mask):
            """processes a single signal"""
            if mask is None:
                mask = ~np.isnan(signal)
            else:
                mask = mask & (~np.isnan(signal))

            target = torch.Tensor(signal[mask])
            tot_nodes = mask.sum()

            # initialize generating signal at random
            generating = torch.Tensor(np.random.normal(size=signal.shape))
            generating.requires_grad = True
            optimizer = torch.optim.Adam([generating], lr=lr)

            for i in range(int(n_iters)):
                optimizer.zero_grad()
                filtered = generating @ self.h_of_L
                loss = ((filtered[mask] - target)**2).sum() / tot_nodes
                loss.backward()
                optimizer.step()

                _be_verbose(i, n_iters, verbose, loss)

            out = (generating @ self.h_of_L).detach().numpy()

            # put the values that we actually know
            out[mask] = signal[mask]
            return generating, out
        
        def is_oneD(arr):
            """checks array is 1D or 2D with single row"""
            return arr.ndim==1 or arr.shape[0]==1
        
        if is_oneD(signal):
            return impute_single_signal(signal, mask)
        else:
            verbose=False
            if is_oneD(mask):
                return np.array(zip(*[impute_single_signal(s, mask)
                                      for s in signal]))
            else:
                return np.array(zip(*[impute_single_signal(s, m)
                                      for s, m in zip(signal, mask)]))                
        
    def __init_graph(self, graph):
                
        # A random adjacency is created if None
        if graph is None:
            self.w = torch.rand(size=(1,(d*(d-1)//2)), dtype=torch.float64)  - 0.5        
        else:
            graph_torch = torch.Tensor(graph)
            
            if graph_torch.ndim > 2:
                raise ValueError('graph must be None or one- or two-dimensional tensor.')
            
            if graph_torch.ndim = 1:
                self.w = graph_torch
            elif torch.any(graph_torch < 0):
                self.L = graph_torch
            else:
                self.A = graph_torch

    def __init_h(self, h):
        
        if h is None:
            self.h = NNet()
        else:
            self.h = h    
    
    def __optim_h_step(evals, evecs):
        self.optim_h.zero_grad()
        filtered_evals = self.__h(evals)
        filtered_L = evecs @ torch.diag(filtered_evals.flatten()) @ evecs.T
        cost = ((filtered_L - sqrt_empirical_cov)**2).sum() 
        cost.backward()
        if np.isnan(cost.detach().numpy()):
            raise RuntimeError('NAN LOSS ENCOUNTERED')
        self.optim_h.step()
        return cost.item()
        
    def __optim_L_step(self, target):
        self.__w.requires_grad = True
        self.optim_L.zero_grad()
        L = w_to_L(self.__w)
        filtered_L = filter_matrix(L, h)
        cost = ((filtered_L - target)**2).sum()
        cost.backward()
        self.optim_L.step()
        self.__w.requires_grad = False
        self.w = self.__w # assures that L and A are updated
        return cost.item()

    def save(self):
        raise NotImplementedError
    
    @classmethod
    def load(self):
        raise NotImplementedError
    
    def plot(self):
        plt.figure(figsize=(12,4))

        plt.subplot(131)
        plt.imshow(self.A>.5)
        plt.title('Rounded Imputed Adjacency')
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(self.A)
        plt.title('Imputed Adjacency')
        plt.axis('off')

        plt.subplot(133)
        evals, _ = np.linalg.eig(self.L)
        x = torch.Tensor(np.linspace(0.01, evals.max()+1, num=100)).unsqueeze_(-1)
        plt.plot(x.numpy(), self.__h(x).detach().numpy())
        plt.legend(['Fitted', 'True'])
        plt.xlabel('Eigenvalue')
        plt.ylabel('Kernel');    
    
    def plot_loss(self):
        plt.plot(self.loss_hist)
        plt.yscale('log')
    
    def _wrong_graph_input_error():
        raise ValueError('W')
        
    def get_L_decomp(self):
        evals, evecs = torch.symeig(self.L, eigenvectors=True)
        evals = evals.unsqueeze_(-1)
        return evals, evecs

    @property
    def A(self):
        return self.__A.detach()
    
    @A.setter
    def A(self, value):
        _check_valid_adjacency(value)
        self.__A = to_torch(value)
        self.__L = A_to_L(value)
        self.__w = A_to_w(self.__A)
    
    @property
    def L(self):
        return self.__L.detach()
    
    @L.setter
    def L(self, value):
        _check_valid_laplacian(value)
        self.__L = to_torch(value)
        self.__A = L_to_A(value)
        self.__w = A_to_w(self.__A)
        
    @property
    def w(self):
        return self.__w.detach()

    @w.setter
    def w(self, value):
        self.__w = to_torch(value)
        self.__A = w_to_A(value)
        self.__L = A_to_L(self.__A)
    
    @property
    def h(self):
        return self.__h

    @h.setter
    def h(self, value):
        assert hasattr(value, '__call__'), 'The assigned value h is not callable.'
        self.__h = value

    @property
    def h_of_L(self):
        if self.__h_of_L is None:
            self.__h_of_L = filter_matrix(self.L, self.h)
        return self.__h_of_L
        
        
def _check_valid_laplacian(L, tol=1e-3):
    L = to_torch(L)
    _check_symmetric(L, 'Laplacian')
    if L.ndim != 2:
        raise ValueError('Adjacency Matrix must have two dimensions')
    if torch.any(L.sum(axis=0) > tol):
        raise ValueError('Laplacian rows do not sum to zero. '
                         'Note that self-loops are not supported.')


def _check_valid_adjacency(A, tol=1e-3):
    A = to_torch(A)
    _check_symmetric(mat, 'Adjacency Matrix')
    if A.ndim != 2:
        raise ValueError('Adjacency Matrix must have two dimensions')
    if torch.any(torch.abs(torch.diag(A)) > tol):
        raise ValueError('Non-zero diagonal entries (i.e. self-loops) '
                         'in the adjacency matrix are not supported.')
    if torch.any(A<0):
        raise ValueError('Adjacency matrix weights must be positive.')
        
        
def _check_symmetric(mat, name='Matrix', tol=1e-5):
    mat = to_torch(mat)
    if torch.any((mat-mat.T) < tol):
        raise ValueError(name + ' is not symmetric.')
    

def _seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    
def _be_verbose(it, total, verbose, cost):
    if verbose and (it==0 or (it+1) % verbose == 0 or it+1==total):
        print('\r[Epoch %4d/%d] loss: %f' % (it+1, total, cost), end='')


def _stop_model_tracking(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def _start_model_tracking(model):
    for param in model.parameters():
        param.requires_grad = True
    return model


