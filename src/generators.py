import torch
import numpy as np
from helpers import to_torch
import networkx as nx
from networkx.generators.community import stochastic_block_model


def filter_matrix(L, h):
    """Applies function h to eigenvalues of matrix L"""
    evals, evecs = torch.symeig(L, eigenvectors=True)
    shape = evals.shape
    return evecs @ torch.diag(h(evals.view(shape[0],-1)).flatten()) @ evecs.T


def filter_matrix_nnet(L, h):
    """Applies function h to eigenvalues of matrix L"""
    evals, evecs = torch.symeig(L, eigenvectors=True)
    filtered_evals = h(evals.unsqueeze_(-1))
    return evecs @ torch.diag(filtered_evals.flatten()) @ evecs.T


def filter_signal(signal, L, h, n=100):
    """filters signal on graph specified by L using the filter defined by h"""
    filter_L = filter_matrix(L, h)
    return signal @ filter_L 

    
def gen_white_noise(n,d,random_state=None):
    """generate white noise, n observations and d features"""
    mean = np.zeros(d)
    cov = np.eye(d)
    if random_state is not None:
        np.random.seed(random_state)
    return to_torch(np.random.multivariate_normal(mean, cov, n))
    

def gen_dirac(n,d, seed=None, p=None):
    """generate signal which is -1 with p/2 and 1 with p/2, 0 otherwise"""
    if p is None:
        p = 2/d
        
    if seed is not None:
        np.random.seed(seed)
    unif = np.random.rand(n,d)
    out = np.zeros_like(unif)
    out[unif<p/2] = -1
    out[unif>(1-p/2)] = 1
    return to_torch(out[np.abs(out).sum(1) > 0])


def kernel_heat(eig, alpha=.1):
    return torch.exp(-alpha*eig)


def kernel_tikhonov(eig, alpha=1):
    return 1/(1 + alpha*eig)


def kernel_normal(eig, tol=1e-2):
    out = torch.zeros_like(eig)
    out[eig>tol] = torch.sqrt(1/eig[eig>tol])
    return out


def kernel_highpass(eig, par=.1):
    return par*eig/(1+par*eig)


def gen_and_filter(L, n, gen=gen_white_noise, ker=kernel_normal, seed=None):
    """Generate n random signals using the generator function gen, then
    filter the signal on the graph specified by L using the kernel ker"""
    if type(L) is not torch.Tensor:
        L = torch.Tensor(L)
    
    sig = gen(n, L.shape[0], seed)
    return filter_signal(sig, L, ker)


def generate_L_sbm(nnodes=10, p_in=.8, p_out=.1, seed=42, n_blocks=2):
    """generates L from stochastic block model with two clusters, with nnodes."""
    if n_blocks==2:
        G = stochastic_block_model([nnodes//2,nnodes//2],[[p_in,p_out],[p_out,p_in]], seed=seed)
        L = nx.laplacian_matrix(G).todense()
        return L
    else:
        prob_matrix = np.eye(n_blocks)*(p_in-p_out) + p_out
        G = stochastic_block_model([nnodes//n_blocks]*n_blocks, prob_matrix, seed=seed)            
        L = nx.laplacian_matrix(G).todense()
        return L
