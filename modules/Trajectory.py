import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.datasets import make_moons, make_swiss_roll
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader
from zuko.utils import odeint
from tqdm import tqdm
from typing import *
import numpy as np
import pandas as pd


# Optimal Transport 
class OTFlowMatching:

    def __init__(self,sig_min: float = 0.001) -> None:
        super().__init__()
        self.sig_min = sig_min
        self.eps = 1e-5
    
    def psi_t(self, x: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor: #  eq 22 from Flow Matching paper
        # conditional flow
        # x , x_1 (Batch_Size, Dim)
        return (1-(1-self.sig_min)*t)*x + t*x_1
    
    def loss(self, v_t: nn.Module, x_1: torch.Tensor) -> torch.Tensor: #  eq 23 from Flow Matching paper
        # t ~ Unif([0,1])
        t = torch.rand(1,device = x_1.device) + torch.arange(len(x_1),device = x_1.device)/len(x_1)
        t = t % (1-self.eps)
        t = t[:,None].expand(x_1.shape)
        # x ~ p_t(x_0)
        x_0 = torch.randn_like(x_1)
        v_psi = v_t(t[:,0], self.psi_t(x_0, x_1, t))
        d_psi = x_1 - (1-self.sig_min)*x_0

        return torch.mean((v_psi - d_psi)**2)
    

# Variance Preserving Diffusion

class VPDiffusionFlowMatching:

    def __init__(self) -> None:
        super().__init__()
        self.beta_min = 0.1
        self.beta_max = 20.0
        self.eps = 1e-5
    
    def T(self, s: torch.Tensor) -> torch.Tensor:
        return self.beta_min * s + 0.5 * (s ** 2) * (self.beta_max - self.beta_min)
    
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def alpha(self, t: torch.Tensor) -> torch.Tensor:#  eq 18 from Flow Matching paper
        return torch.exp(-0.5 * self.T(t))
    
    def mu_t(self, t: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor: #  eq 18 from Flow Matching paper
        return self.alpha(1. - t) * x_1
    
    def sigma_t(self, t: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor: #  eq 18 from Flow Matching paper
        return torch.sqrt(1. - self.alpha(1. -t)**2)
    
    def u_t(self, t:torch.Tensor, x: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor: #  eq 19 from Flow Matching paper
        num = torch.exp(-self.T(1. - t))*x - torch.exp(-0.5 * self.T(1. -t))*x_1
        denum = 1. - torch.exp(-self.T(1. -t))
        return -0.5 * self.beta(1. -t) * (num/denum)
    
    def loss(self,v_t:nn.Module, x_1: torch.Tensor) -> torch.Tensor:
        # t ~ Unif([0,1])
        t = torch.rand(1,device = x_1.device) + torch.arange(len(x_1),device = x_1.device)/len(x_1)
        t = t % (1-self.eps)
        t = t[:,None].expand(x_1.shape)    

        # x ~p_t(x|x_1)
        x = self.mu_t(t,x_1) + self.sigma_t(t,x_1) * torch.randn_like(x_1)

        return torch.mean((v_t(t[:,0],x) - self.u_t(t,x,x_1))**2)  


# Variance Exploding Diffusion

class VEDiffusionFlowMatching:

    def __init__(self) -> None:
        super().__init__()
        self.sigma_min = 0.01
        self.sigma_max = 2.
        self.eps = 1e-5
    
    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_min * (self.sigma_max/self.sigma_min) ** t
    
    def dsigma_dt(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_t(t) * torch.log(torch.tensor(self.sigma_max/self.sigma_min))
    
    def u_t(self, t: torch.Tensor, x: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor: # eq 17 from Flow Matching paper
        return - (self.dsigma_dt(1.-t)/self.sigma_t(1.-t)) * (x - x_1)
    
    def loss(self, v_t:nn.Module, x_1: torch.Tensor) -> torch.Tensor:
        # t ~ Unif([0,1])
        t = torch.rand(1,device = x_1.device) + torch.arange(len(x_1),device = x_1.device)/len(x_1)
        t = t % (1-self.eps)
        t = t[:,None].expand(x_1.shape)    

        # x ~p_t(x|x_1) N(x|x_1, sigma_{1-t}I) eq 16 from Flow Matching paper
        # randn_like : Returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1.
        x = x_1 + self.sigma_t(1. - t) * torch.randn_like(x_1)

        return torch.mean((v_t(t[:,0],x) - self.u_t(t,x,x_1))**2)


# sub Variance Preserving Diffusion


class subVPDiffusionFlowMatching:

    def __init__(self) -> None:
        super().__init__()
        self.beta_min = 0.1
        self.beta_max = 20.0
        self.eps = 1e-5
    
    def T(self, s: torch.Tensor) -> torch.Tensor:
        return self.beta_min * s + 0.5 * (s ** 2) * (self.beta_max - self.beta_min)
    
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def alpha(self, t: torch.Tensor) -> torch.Tensor:#  
        return torch.exp(-0.5 * self.T(t))
    
    def mu_t(self, t: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor: #  
        return self.alpha(1. - t) * x_1
    
    def sigma_t(self, t: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor: #  
        return 1. - self.alpha(1. -t)**2
    
    # def u_t(self, t:torch.Tensor, x: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor: #  
    #     num = ((x - torch.exp(-0.5 * self.T(1. -t))*x_1)**2) * (1.+torch.exp(-self.T(1. - t)))
    #     denum = 1. - torch.exp(-self.T(1. -t))
    #     div1 = num/denum
    #     div1 = div1 - x
    #     return -0.5 * self.beta(1. -t) * div1
    def u_t(self, t:torch.Tensor, x: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor: #  
        num =  2 * torch.exp(-0.5*self.T(1. -t))*x - x_1* (torch.exp(-self.T(1. -t))+1)
        denum = 1. - torch.exp(-self.T(1. -t))
        div1 = num/denum
        div1 = div1 * (0.5 * torch.exp(-0.5*self.T(1. -t)))
        return  -self.beta(1. -t) * div1
    
    def loss(self,v_t:nn.Module, x_1: torch.Tensor) -> torch.Tensor:
        # t ~ Unif([0,1])
        t = torch.rand(1,device = x_1.device) + torch.arange(len(x_1),device = x_1.device)/len(x_1)
        t = t % (1-self.eps)
        t = t[:,None].expand(x_1.shape)    

        # x ~p_t(x|x_1)
        x = self.mu_t(t,x_1) + self.sigma_t(t,x_1) * torch.randn_like(x_1)

        return torch.mean((v_t(t[:,0],x) - self.u_t(t,x,x_1))**2)  
