import torch
import numpy as np
import pandas as pd
from helpers import to_torch, vals_to_A, vals_to_L, symsqrt
from generators import filter_matrix, filter_matrix_nnet
from NNet import NNet
from copy import deepcopy



def fit_filter(L, sqrt_empirical_cov, h=None, optimizer=None, n_iters=200,
               lr_nnet=1e-4, random_seed=42):
    """
    Fits filter on graph topology defined by L to the covariance matrix 
    in target. 
    
    Parameters
    ----------
    L (torch.tensor):
        Graph Laplacian
    sqrt_empirical_cov (torch.tensor):
        Matrix square root of empirical covariance matrix
    h (NNet):
        Neural network of pretrained weights (optional)
        or None for a new neural network
    optimizer:
        Torch optimizer to use (optional)
        or None for SGD
    n_iters:
        Number of epochs for fitting
    lr_nnet:
        Learning rate for neural network (only applies if optimizer is not 
        specified)
        
    Returns
    -------
    NNet:
        Neural network fitting the filter.
    """
    
    # TODO: Support L and cov as np.ndarrays
    # TODO: doesnt seem to be reproducible

    torch.manual_seed(random_seed)
    
    if h is None:
        h = fit_normal_kernel()
    if optimizer is None:
        optimizer = torch.optim.SGD(h.parameters(), lr=lr_nnet)
    
    evals, evecs = torch.symeig(L, eigenvectors=True)   
    evals = evals.unsqueeze_(-1)
    for i in range(n_iters):
        optimizer.zero_grad()
        filtered_evals = h(evals)
        filtered_L = evecs @ torch.diag(filtered_evals.flatten()) @ evecs.T
        cost = ((filtered_L - sqrt_empirical_cov)**2).sum() 
        cost.backward()
        if np.isnan(cost.detach().numpy()):
            raise RuntimeError('NAN LOSS ENCOUNTERED')
        optimizer.step()   
    return h


def impute_graph(y, lr=.01, lr_nnet=1e-3, nit_nnet=3, start=None, h_start=None,
                 n_epochs=3000, random_seed=23, verbose=100):
    """
    Impute graph by alterating between fitting neural network
    and Laplacian
    """
    _, d = y.shape
    
    C = to_torch(np.cov(y.T))
    target = symsqrt(C)
    
    history = pd.DataFrame(columns=['Loss', 'Nb_Sign_Switch','Nb_Zero',
                                   'Nb_One','Mean_Step','Median_Step',
                                   'Vals_sum'], 
                           index=range(n_epochs))

    torch.manual_seed(random_seed)
    
    best_cost = 1e10
    best_vals = None
    best_h = None
    
    if start is None:
        vals = torch.rand(size=(1,(d*(d-1)//2)), dtype=torch.float64)
    else:
        start = to_torch(start)
        vals = start[torch.triu(torch.ones(d, d), 1) == 1].unsqueeze_(0)
        
    # Initialize h
    if h_start is None:
        h = NNet()
    else:
        h = h_start
    
    nnet_optimizer = torch.optim.SGD(h.parameters(), lr=lr_nnet)
    l_optimizer = torch.optim.Adam([vals], lr=lr)
    
    for epoch in range(n_epochs):
        vals.requires_grad = False
        L = vals_to_L(torch.sigmoid(vals))
        
        h = start_model_tracking(h)
        h = fit_filter(L, target, h, nnet_optimizer, n_iters=nit_nnet)
        h = stop_model_tracking(h)
        
        vals.requires_grad = True
        l_optimizer.zero_grad()
        L = vals_to_L(torch.sigmoid(vals))
        
        filtered_L = filter_matrix(L, h)
        cost = ((filtered_L - target)**2).sum()
        
        if best_cost>cost.item():
            best_cost = cost.item()
            best_vals = torch.sigmoid(vals)
            best_h = deepcopy(h)
        
        try:
            cost.backward()
        except RuntimeError as e:
            print(e)
            return vals_to_A(proj_vals).detach().numpy(), history, h
        
        # Update
        l_optimizer.step()
        
        # History
        history.loc[epoch] = {'Loss': cost.item(),
                             #'Nb_Sign_Switch': ((proj_vals>.5) & ~(vals>.5)).sum().item(),
                             'Nb_Zero': (vals <= 0).sum().item(),
                             'Nb_One': (vals >= 1).sum().item(),
                             #'Mean_Step': torch.mean(proj_vals.grad*lr).item(),
                             #'Median_Step': torch.median(proj_vals.grad*lr).item(),
                             'Vals_sum': vals.sum().item()}
        
        if verbose and (epoch==0 or (epoch+1) % verbose == 0 or epoch+1==n_epochs):
            print('\r[Epoch %4d/%d] loss: %f' % (epoch+1, n_epochs, cost.item()), end='')
            
    return vals_to_A(best_vals).detach().numpy(), history, best_h


def project_onto_pos(vals):
    """Projects vals onto space of non-negative matrices."""
    neg_vals = vals.clone().detach().numpy()
    neg_vals[neg_vals>0] = 0
    return vals.float() - to_torch(neg_vals)

    
def project_onto_leq_one(vals):
    """Projects vals onto space of values with less or equal to one."""    
    large_vals = vals.clone().detach().numpy()
    large_vals[large_vals<=1] = 1
    return vals - to_torch(large_vals - 1)


def project_onto_zero_one(vals):
    """Projects vals onto space with values between zero and one."""
    proj1 = project_onto_pos(vals)
    return project_onto_leq_one(proj1)


def stop_model_tracking(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def start_model_tracking(model):
    for param in model.parameters():
        param.requires_grad = True
    return model


def fit_normal_kernel(nit=300, lr=1e-3):
    """
    Fits a standard normal kernel to the neural network, allows to hot-start 
    the filter. Returns fitted neural network.
    """
    torch.manual_seed(42)
    # create your optimizer
    
    x = torch.Tensor(2**np.linspace(-3,8, num=100)).unsqueeze_(-1)
    y = 1/(torch.sqrt(x))
    net = NNet(n_hidden=3)
    
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    
    history = []
    
    for i in range(nit):
        optimizer.zero_grad()
        output = net(x)
        loss = ((output - y)**2).sum()
        history.append(loss.item())
        loss.backward()
        optimizer.step()    # Does the update
    
    return net






"""
UNUSED
def fit_normal_kernel_scale(L, target, n_epochs=100):
    a = torch.Tensor([1])
    a.requires_grad=True
    optimizer = torch.optim.SGD([a], lr=.001)
    for i in range(n_epochs):
        optimizer.zero_grad()
        filtered_L = filter_matrix_nnet(L, lambda x: a * kernel_normal(x))
        cost = ((filtered_L - target)**2).sum() #+ alpha*(torch.exp(h.a))**2
        cost.backward()
        optimizer.step()
    return a.detach().item()
"""