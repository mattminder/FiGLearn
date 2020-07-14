import torch

class NNet(torch.nn.Module):
    def __init__(self, n_hidden=3, h_size=10):
        super().__init__()
        modules = []
        modules.append(torch.nn.Linear(1, h_size))
        modules.append(torch.nn.ReLU())
        for k in range(n_hidden):
            modules.append(torch.nn.Linear(h_size, h_size))
            modules.append(torch.nn.ReLU())
            
        self.hidden = torch.nn.Sequential(*modules)
        self.last = torch.nn.Linear(h_size, 1, bias=False)
        # self.a = torch.nn.Parameter(torch.Tensor([-10]))
        # self.scale = torch.nn.Parameter(torch.Tensor([0]))
        
        
    def forward(self, x, tol=1e-3):
        out = torch.zeros_like(x)
        above_tol = x[x>tol].unsqueeze_(-1) # don't predict for x less than tolerance (singluarity)
        hid = self.hidden(above_tol)
        logout = self.last(hid)
        val = torch.exp(-logout)
        #val = 1/(exp+1e-2)
        #print(val)
        #val = 1/(exp + torch.exp(self.a)*torch.sqrt(above_tol))
        out[x>tol] = val.flatten()
        return out