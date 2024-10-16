import torch
import torch.nn as nn
from zuko.utils import odeint
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
    def decode(self, x_0: torch.Tensor) -> torch.Tensor:
        return odeint(self.wrapper, x_0, 0., 1., self.parameters())
    
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



