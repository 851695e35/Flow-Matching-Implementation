import torch
import torch.nn as nn
from zuko.utils import odeint
from torchdiffeq import odeint # needs to be changed in the code accordingly.
from typing import *


# Conditional Vector Field
class CondVF(nn.Module):
    def __init__(self,net: nn.Module) -> None:
        super().__init__()
        self.net = net
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(t,x)

    def wrapper(self, t: torch.Tensor, x: torch.Tensor):
        t = t * (torch.ones(len(x)).to(t.device))
        return self(t,x)

    # encode maps the data distribution to the prior distribution
    # t from 1 to 0  
    def encode(self,x_1:torch.Tensor) -> torch.Tensor:
        return odeint(self.wrapper, x_1, 1., 0., self.parameters())

    # decode maps the prior distribution to the data distribution
    # t from 0 to 1
    #     sol = odeint(
    #     ode_func,
    #     x_init,
    #     time_grid,
    #     method=method,
    #     options=ode_opts,
    #     atol=atol,
    #     rtol=rtol,
    # )
    # atol (float): Absolute tolerance, used for adaptive step solvers.
    # rtol (float): Relative tolerance, used for adaptive step solvers.
    
    def eulersolver(self,x_0:torch.Tensor, time_grid: torch.Tensor) -> torch.Tensor:
        # manually solve the ODE using midpoint
        # x_0 : initial condition
        # time_grid : time grid [t0,t1,...,tn]
        time_grid = time_grid.to(x_0.device)
        x_list = [x_0]
        for i in range(1,len(time_grid)):
            x_0 = x_0 + (time_grid[i] - time_grid[i-1]) * self.wrapper(time_grid[i],x_0)
            x_list.append(x_0)
        
        return torch.stack(x_list)
            
        
        
        
    
    def decode(self, x_0: torch.Tensor, time_grid: torch.Tensor,method) -> torch.Tensor:
        return odeint(self,x_0, time_grid, method = method)
    
    def decode_t0_t1(self, x_0, t0, t1):
        return odeint(self.wrapper, x_0, t0, t1, self.parameters())
    

# NN for VF approximation
class Net(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, h_dims: List[int], n_frequencies: int) -> None:
        super().__init__()
        # list : [in_dim+2*n_frequencies,h_dims]
        ins = [in_dim+2*n_frequencies] + h_dims
        # list : [h_dims,out_dim]
        outs = h_dims + [out_dim]
        self.n_frequencies = n_frequencies

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_d,out_d),
                nn.LeakyReLU()
            )
            for in_d,out_d in zip(ins,outs)
        ])
        self.top = nn.Sequential(nn.Linear(out_dim,out_dim))
    
    def time_encoder(self, t: torch.Tensor) -> torch.Tensor:
        freq = 2 * torch.arange(self.n_frequencies, device = t.device) * torch.pi
        t = freq * t[..., None]
        return torch.cat([t.cos(),t.sin()], dim = -1)
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t = self.time_encoder(t)
        x = torch.cat((x,t),dim = -1)

        for layer in self.layers:
            x = layer(x)
        
        x = self.top(x)

        return x



