import torch
import numpy as np
from helpers import to_torch, w_to_A, w_to_L, symsqrt, A_to_L, L_to_A, A_to_w
from NNet import NNet
from matplotlib import pyplot as plt
from copy import deepcopy


class FiGLearn:
    
    
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
        self._init_graph(graph)
        self._init_h(h)
        self.optim_h = None 
        self.optim_L = None 
        self.loss_hist = []
        self._h_of_L = None

    def fit_graph(self, mat, lr_L=1e-2, lr_h=1e-3, nit=3000, nit_h_per_iter=3,
                  learn_h=True, mat_is_cov=False, fine_tune=False,
                  seed=23, verbose=100, optim_h='gd', optim_L='adam'):
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
        optim_h (str):
            Either 'gd' or 'adam'. Specifies method to use for optimization of the 
            neural network. Gradient descent yields to better results but slightly slows down 
            convergence. If adam is used, we recommend a learning rate of 1e-4. 
            Default is 'gd'.
        optim_L (str):
            Either 'gd' or 'adam'. 
        """
    
        _seed(seed)
        self._h_of_L = None   # forget about previously computed h(L)
        
        _, d = mat.shape
        if self._w is None: # if no w provided, chose random
            self.w = torch.rand(size=(1,(d*(d-1)//2)), dtype=torch.float64)  - 0.5        
            
        if mat_is_cov:
            cov = to_torch(mat)
        else:
            cov = to_torch(np.cov(mat.T))

        target = symsqrt(cov)

        self._create_optimizers(fine_tune, lr_L, lr_h, optim_L, optim_h)

        for iter in range(int(nit)):
            if learn_h:
                _start_model_tracking(self._h)
                evals, evecs = self.get_L_decomp(ignore=0)
                for _ in range(int(nit_h_per_iter)):
                    self._optim_h_step(target, evals, evecs)
                _stop_model_tracking(self._h)

            cost = self._optim_L_step(target)
            self.loss_hist.append(cost)

            _be_verbose(iter, nit, verbose, cost)    
    
    def fit_filter(self, mat, n_iters=1000, lr_nnet=1e-4, seed=42,
                   mat_is_cov=False, round_adj=False):
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
        round_adj (bool):
            Whether to round the adjacency matrix with a threshold of 0.5 before 
            fitting the filter.
        """
        _seed(seed)
        self._h_of_L = None   # forget about previously computed h(L)
        
        if mat_is_cov:
            cov = mat
        else:
            cov = np.cov(mat.T)
        sqrt_cov = symsqrt(to_torch(cov))
        evals, evecs = self.get_L_decomp(round=round_adj)

        self.optim_h = torch.optim.Adam(self._h.parameters(), lr=lr_nnet)
        
        _start_model_tracking(self._h)
        for i in range(int(n_iters)):
            self._optim_h_step(sqrt_cov, evals, evecs)
        _stop_model_tracking(self._h)
        
    def infer_missing(self, signal, mask=None, lr=1e-2, n_iters=1000, 
                       seed=42, verbose=None):
        """
        Infers the missing values in a signal using the graph and 
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
        
        def infer_single_signal(signal, mask):
            """processes a single signal"""
            
            signal = signal.flatten()
            if mask is None:
                mask = ~np.isnan(signal)
            else:
                mask = (mask & (~np.isnan(signal))).flatten()

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
            return generating.detach().numpy(), out
        
        def is_oneD(arr):
            """checks array is 1D or 2D with single row"""
            return arr.ndim==1 or arr.shape[0]==1
        
        if is_oneD(signal):
            return infer_single_signal(signal, mask)
        else:
            verbose=False
            if mask is None or is_oneD(mask):
                gen, imp = zip(*[infer_single_signal(s, mask)
                                 for s in signal])
            else:
                gen, imp = zip(*[infer_single_signal(s, m)
                                 for s, m in zip(signal, mask)])
            return np.array(gen), np.array(imp)
        

            
    def round(self, mat=None, threshold=.5, copy=False, **kwargs):
        """
        Rounds the inferred adjacency matrix to 0 or 1. If sample is provided,
        fits the filter to the rounded adjacency matrix. (recommended)
        
        mat (np.ndarray or None):
            Input to fit_filter. If None, the filter isn't refitted.
        threshold (float):
            Value above which an edge is retained.
        copy (bool):
            Whether to return a copy with the rounded matrix. If set to False,
            rounding is performed on self.
        **kwargs:
            Keyword arguments passed to fit_filter
        """
        
        if copy:
            copied = deepcopy(self)
            copied.round(mat=mat, threshold=threshold, copy=False, **kwargs)
            return copied
        
        else:
            self._h_of_L=None
            self.A = (self.A>threshold).float()
            if mat is not None:
                self.fit_filter(mat, **kwargs)
        
    def _init_graph(self, graph):
                
        # A random adjacency is created if None
        if graph is None:
            self._w = None
            self._L = None
            self._A = None
        else:
            graph_torch = torch.Tensor(graph)
            
            if graph_torch.ndim > 2:
                raise ValueError('graph must be None or one- or two-dimensional tensor.')
            
            if graph_torch.ndim == 1:
                self.w = graph_torch
            elif torch.any(graph_torch < 0):
                self.L = graph_torch
            else:
                self.A = graph_torch

    def _init_h(self, h):
        
        if h is None:
            self.h = NNet()
        else:
            self.h = h    
            
    def _create_optimizers(self, fine_tune, lr_L, lr_h, optim_L, optim_h):
        
        if fine_tune:
            # only adjust learning rates
            update_lr(self.optim_h, lr_h)
            update_lr(self.optim_L, lr_L)

        else:
            # create new optimizers
            if optim_h=='adam':
                self.optim_h = torch.optim.Adam(self._h.parameters(), lr=lr_h) 
            elif optim_h=='gd':
                self.optim_h = torch.optim.SGD(self._h.parameters(), lr=lr_h) 
            else:
                raise ValueError('Only "adam" and "gd" are supported for optim_h')
                
            if optim_L=='adam':
                self.optim_L = torch.optim.Adam([self._w], lr=lr_L) 
            elif optim_L=='gd':
                self.optim_L = torch.optim.SGD([self._w], lr=lr_L) 
            else:
                raise ValueError('Only "adam" and "gd" are supported for optim_L')
    
    
    def _optim_h_step(self, target, evals, evecs):
        
        self.optim_h.zero_grad()
        filtered_evals = self._h(evals)
        filtered_L = evecs @ torch.diag(filtered_evals.flatten()) @ evecs.T
        cost = ((filtered_L - target)**2).sum() 
        cost.backward()
        if np.isnan(cost.detach().numpy()):
            raise RuntimeError('NAN LOSS ENCOUNTERED')
        self.optim_h.step()
        return cost.item()
        
    def _optim_L_step(self, target):
        
        self._w.requires_grad = True
        self.optim_L.zero_grad()
        L = w_to_L(self._w)
        filtered_L = filter_matrix(L, self._h)
        cost = ((filtered_L - target)**2).sum()
        cost.backward()
        if torch.any(torch.isnan(self._w.grad)):
            raise RuntimeError('nan gradient encountered. Decrease learning rate.')
        self.optim_L.step()
        self._w.requires_grad = False
        self.w = self._w # assures that L and A are updated
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
        plt.title('Rounded Inferred Adjacency')
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(self.A)
        plt.title('Inferred Adjacency')
        plt.axis('off')

        plt.subplot(133)
        evals, _ = np.linalg.eig(self.L)
        x = torch.Tensor(np.linspace(0.01, evals.max()+1, num=100)).unsqueeze_(-1)
        plt.plot(x.numpy(), self._h(x).detach().numpy())
        plt.legend(['Fitted', 'True'])
        plt.xlabel('Eigenvalue')
        plt.ylabel('Kernel');    
    
    def plot_loss(self):
        
        plt.plot(self.loss_hist)
        plt.yscale('log')
        plt.title('Loss')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Wasserstein Loss')
            
    def get_L_decomp(self, round=False, ignore=0):
        
        if round:
            Around = (self.A > .5).float()
            Lround = A_to_L(Around)
            evals, evecs = torch.symeig(Lround, eigenvectors=True)
        else:
            evals, evecs = torch.symeig(self.L, eigenvectors=True)
        evals = evals.unsqueeze_(-1)
        if ignore>0:
            return evals[:-ignore], evecs[:,:-ignore]
        else:
            return evals, evecs
    
    def filter_signal(self, signal):
        
        return signal @ self._h_of_L
        
    @property
    def A(self):
        
        return self._A.detach()
    
    @A.setter
    def A(self, value):
        
        _check_valid_adjacency(value)
        self._A = to_torch(value)
        self._L = A_to_L(value)
        self._w = A_to_w(self._A)
    
    @property
    def L(self):
        
        return self._L.detach()
    
    @L.setter
    def L(self, value):
        
        _check_valid_laplacian(value)
        self._L = to_torch(value)
        self._A = L_to_A(value)
        self._w = A_to_w(self._A)
        
    @property
    def w(self):
        
        return self._w.detach()

    @w.setter
    def w(self, value):
        
        self._w = to_torch(value)
        self._A = w_to_A(value)
        self._L = A_to_L(self._A)
        
    @property
    def h(self):
        
        return deepcopy(self._h)

    @h.setter
    def h(self, value):
        
        assert hasattr(value, '__call__'), 'The assigned value h is not callable.'
        self._h = value

    @property
    def h_of_L(self):
        
        if self._h_of_L is None:
            self._h_of_L = filter_matrix(self.L, self.h)
        return self._h_of_L
        
        
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
    _check_symmetric(A, 'Adjacency Matrix')
    if A.ndim != 2:
        raise ValueError('Adjacency Matrix must have two dimensions')
    if torch.any(torch.abs(torch.diag(A)) > tol):
        raise ValueError('Non-zero diagonal entries (i.e. self-loops) '
                         'in the adjacency matrix are not supported.')
    if torch.any(A<0):
        raise ValueError('Adjacency matrix weights must be positive.')
        
        
def _check_symmetric(mat, name='Matrix', tol=1e-5):
    
    mat = to_torch(mat)
    if torch.any((mat-mat.T) > tol):
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


def filter_matrix(L, h):
    
    """Applies function h to eigenvalues of matrix L"""
    evals, evecs = torch.symeig(L, eigenvectors=True)
    shape = evals.shape
    return evecs @ torch.diag(h(evals.view(shape[0],-1)).flatten()) @ evecs.T


def update_lr(optim, new_lr):
    
    for g in optim.param_groups:
        g['lr'] = new_lr