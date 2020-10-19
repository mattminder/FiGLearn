import torch
import numpy as np
from copy import deepcopy

def vals_to_A(vals):
    """Converts values to adjacency matrix """
    _, l = vals.shape
    d = int(1/2 * (1 + np.sqrt(8*l+1))) # dimension of adjacency
    A = torch.zeros((d,d)).float()
    A[torch.triu(torch.ones_like(A, dtype=float), 1)==1] = vals.float()
    A = A+A.t()
    return A
    
    
def A_to_L(A, reg=None):
    if type(A) is not torch.Tensor:
        A = to_torch(A)
    dd = torch.sum(A,0)
    D = torch.diag(dd)
    if reg is not None:
        return (D-A) + torch.eye(A.shape[0])*reg
    else:
        return (D-A)

    
def L_to_A(L):
    if type(L) is not torch.Tensor:
        L = to_torch(L)

    out = -deepcopy(L)
    out[torch.eye(*L.shape).bool()] = 0
    return out


def A_to_w(A):
    """Converts A to the optimized vector, i.e. returns logit of the
    upper triangular values. A is clipped to (0.05, 0.95) to avoid inf"""
    clamped = torch.clamp(A, .05, .95) # avoid infinite values
    logit = torch.log(clamped) - torch.log(1-clamped)
    return logit[torch.triu(torch.ones(*A.shape), 1) == 1].unsqueeze_(0)
    
    
def w_to_A(w):
    """Converts values to adjacency matrix """
    sigm = torch.sigmoid(w)
    _, l = w.shape
    d = int(1/2 * (1 + np.sqrt(8*l+1))) # dimension of adjacency
    A = torch.zeros((d,d)).float()
    A[torch.triu(torch.ones_like(A, dtype=float), 1)==1] = sigm.float()
    A = A+A.t()
    return A

def w_to_L(w):
    return A_to_L(w_to_A(w))

    
def vals_to_L(vals, reg=1e-5):
    A = vals_to_A(vals)
    return A_to_L(A, reg)


def to_torch(arr):
    if type(arr) is torch.Tensor:
        return arr
    
    elif type(arr) is np.ndarray:
        return torch.from_numpy(arr).float()
    else:
        return torch.Tensor(arr, dtype=float)

    
def standardize(arr):
    return (arr - arr.mean(0)) / torch.sqrt(arr.var(0))


def symsqrt(a, cond=None, return_rank=False):
    """Computes the symmetric square root of a positive definite matrix"""
    s, u = torch.symeig(a, eigenvectors=True)
    cond_dict = {torch.float32: 1e3 * 1.1920929e-07, torch.float64: 1E6 * 2.220446049250313e-16}
    if cond in [None, -1]:
        cond = cond_dict[a.dtype]
    above_cutoff = (abs(s) > cond * torch.max(abs(s)))
    psigma_diag = torch.sqrt(s[above_cutoff])
    u = u[:, above_cutoff]
    B = u @ torch.diag(psigma_diag) @ u.t()
    if return_rank:
        return B, len(psigma_diag)
    else:
        return B

    
def pinv(A):
    """
    Return the pseudoinverse of A using the QR decomposition.
    """
    Q,R = torch.qr(A)
    return R.pinverse().mm(Q.t())
    
    
def w2_dist(C, L, add_to_Linv=0):
    C = to_torch(C)
    L_inv = to_torch(np.linalg.pinv(L))
    C_sqrt = symsqrt(C)
    sqrt_L_inv = symsqrt(L_inv)
    return ((sqrt_L_inv - C_sqrt)**2).sum()
    
    
def accuracy(Atrue, Aimp):
    d = Atrue.shape[0]
    return 1 - np.logical_xor(Atrue>0, Aimp>0).sum()/(d*(d-1))

