import torch

class NNet(torch.nn.Module):
    
    
    def __init__(self, n_hidden=5, h_size=30):
        super().__init__()
        modules = []
        modules.append(torch.nn.Linear(1, h_size))
        modules.append(torch.nn.ReLU())
        for k in range(n_hidden):
            modules.append(torch.nn.Linear(h_size, h_size))
            modules.append(torch.nn.ReLU())
            
        self.hidden = torch.nn.Sequential(*modules)
        self.last = torch.nn.Linear(h_size, 1, bias=True)
        self.softplus = torch.nn.Softplus()
        
        
    def forward(self, x, tol=1e-3):
        hid = self.hidden(x)
        logout = self.last(hid)
        val = torch.exp(-logout)
        return self.softplus(logout)
