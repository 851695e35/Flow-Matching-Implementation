import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs, make_s_curve, make_classification, make_swiss_roll
from sklearn.preprocessing import StandardScaler
from modules.Trajectory import OTFlowMatching, VPDiffusionFlowMatching, VESMLDDiffusionFlowMatching,VEEDMDiffusionFlowMatching, subVPDiffusionFlowMatching,CosineDiffusionFlowMatching
import matplotlib.pyplot as plt
def get_data(dataset: str, n_points: int, seed: int = 42) -> np.ndarray:
    # Set the random seed for reproducibility
    np.random.seed(seed)

    if dataset == "moons":
        data, _ = make_moons(n_points, noise=0.15, random_state=seed)
    elif dataset == "circles":
        data, _ = make_circles(n_points, noise=0.1, factor=0.5, random_state=seed)
    elif dataset == "blobs":
        data, _ = make_blobs(n_points, centers=3, random_state=seed)
    elif dataset == "noisy_circles":
        data, _ = make_circles(n_points, noise=0.2, factor=0.5, random_state=seed)
    elif dataset == "s_curve":
        data, _ = make_s_curve(n_points, noise=0.1, random_state=seed)
        data = data[:, [0, 2]]  # Use the first and last dimension to get a 2D view
    elif dataset == "classification":
        data, _ = make_classification(n_points, n_features=2, n_redundant=0, n_clusters_per_class=1, flip_y=0.1, n_classes=2, random_state=seed)
    elif dataset == "swiss":
        data, _ = make_swiss_roll(n_points, noise=0.15, random_state=seed)
        data = data[:, [0, 2]] / 10.0  # Rescaling to reduce the spread
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    return StandardScaler().fit_transform(data)
    


def get_model(name: str):
    if name == "ot":
        return OTFlowMatching()
    elif name == "vp":
        return VPDiffusionFlowMatching()
    elif name == "ve-smld":
        return VESMLDDiffusionFlowMatching()
    elif name == "ve-edm":
        return VEEDMDiffusionFlowMatching()
    elif name == "subvp":
        return subVPDiffusionFlowMatching()
    elif name == "cosine":
        return CosineDiffusionFlowMatching()
    else:
        raise ValueError(f"Unknown model: {name}")
    
    

def plot_losses(losses_dict, dataset_name):
    plt.figure(figsize=(10, 5))
    for model_name, losses in losses_dict.items():
        plt.plot(losses, label=f'{model_name} Loss')
    plt.title(f'Training Losses for {dataset_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"loss_img/loss_{dataset_name}.png")
    plt.close()  
    
    
    
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
"""This is an ad-hoc sampling schedule that was proposed in https://arxiv.org/abs/2206.00364 it works very well for cifar 10 so we added its implementation here. It did not yield an improvement on ImageNet."""
import torch
# stc
def t2lambda(t):
    return torch.log(t/(1-t))

def lambda2t(lambdat):
    return 1/(1+torch.exp(-lambdat))


def time_quadratic(nfes: int, rho:int):
    t_min = 0.0001
    t_max = 0.9999
    # t_i = (i/nfes)^rho (t_max-t_min) + t_min
    t_samples = torch.linspace(0, 1, nfes + 1)
    t_samples = t_samples ** rho * (t_max - t_min) + t_min
    return t_samples


def nfe2step(nfes: int, method = 'euler'):
    if method == 'euler':
        return nfes
    elif method == 'midpoint' or method == 'heun2':
        return (nfes+1)//2
    elif method == 'heun3':
        return (nfes+2)//3
    else:
        raise ValueError(f"method {method} not supported")
    


# stc
def get_time_discretization(nfes: int, schedule = "edm", method = 'euler'):
    
    nfes = nfe2step(nfes, method)
    
    if schedule == "edm":
        rho=7
        step_indices = torch.arange(nfes, dtype=torch.float64)
        sigma_min = 0.002
        sigma_max = 80.0
        sigma_vec = (
            sigma_max ** (1 / rho)
            + step_indices / (nfes - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        sigma_vec = torch.cat([sigma_vec, torch.zeros_like(sigma_vec[:1])])
        time_vec = (sigma_vec / (1 + sigma_vec)).squeeze()
        t_samples = 1.0 - torch.clip(time_vec, min=0.0, max=1.0)
    elif schedule == "time_uniform":
        t_samples = time_quadratic(nfes, 1)
    elif schedule == "time_quadratic":
        t_samples = time_quadratic(nfes, 2)
    elif schedule == "logsnr":
        t_min = 0.0001
        t_max = 0.9999
        lambda_min = t2lambda(torch.tensor(t_min))
        lambda_max = t2lambda(torch.tensor(t_max))
        lambda_samples = torch.linspace(lambda_min, lambda_max, nfes + 1)
        t_samples = lambda2t(lambda_samples)
        

    return t_samples


if __name__ == "__main__":
    # time_quadratic
    # t_samples = get_time_discretization(50,'time_quadratic')
    # print(t_samples)
    
    
    # from DL3 code gits timestep
    snr_min = 0.0292
    snr_max = 14.6146
    # convert them to logsnr
    logsnr_min = torch.log(torch.tensor(snr_min))
    logsnr_max = torch.log(torch.tensor(snr_max))
    
    # convert them to t and output
    t_min = lambda2t(torch.tensor(logsnr_min))
    t_max = lambda2t(torch.tensor(logsnr_max))
    print(t_min)
    print(t_max)




# a good example of how to use the get_schedule function from diff-sampler
def get_schedule(num_steps, sigma_min, sigma_max, device=None, schedule_type='polynomial', schedule_rho=7, net=None, dp_list=None):
    """
    Get the time schedule for sampling.

    Args:
        num_steps: A `int`. The total number of the time steps with `num_steps-1` spacings. 
        sigma_min: A `float`. The ending sigma during samping.
        sigma_max: A `float`. The starting sigma during sampling.
        device: A torch device.
        schedule_type: A `str`. The type of time schedule. We support three types:
            - 'polynomial': polynomial time schedule. (Recommended in EDM.)
            - 'logsnr': uniform logSNR time schedule. (Recommended in DPM-Solver for small-resolution datasets.)
            - 'time_uniform': uniform time schedule. (Recommended in DPM-Solver for high-resolution datasets.)
            - 'discrete': time schedule used in LDM. (Recommended when using pre-trained diffusion models from the LDM and Stable Diffusion codebases.)
        schedule_type: A `float`. Time step exponent.
        net: A pre-trained diffusion model. Required when schedule_type == 'discrete'.
    Returns:
        a PyTorch tensor with shape [num_steps].
    """
    if schedule_type == 'polynomial':
        step_indices = torch.arange(num_steps, device=device)
        t_steps = (sigma_max ** (1 / schedule_rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / schedule_rho) - sigma_max ** (1 / schedule_rho))) ** schedule_rho
    elif schedule_type == 'logsnr':
        logsnr_max = -1 * torch.log(torch.tensor(sigma_min))
        logsnr_min = -1 * torch.log(torch.tensor(sigma_max))
        t_steps = torch.linspace(logsnr_min.item(), logsnr_max.item(), steps=num_steps, device=device)
        t_steps = (-t_steps).exp()
    elif schedule_type == 'time_uniform':
        epsilon_s = 1e-3
        vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
        vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
        step_indices = torch.arange(num_steps, device=device)
        vp_beta_d = 2 * (np.log(torch.tensor(sigma_min).cpu() ** 2 + 1) / epsilon_s - np.log(torch.tensor(sigma_max).cpu() ** 2 + 1)) / (epsilon_s - 1)
        vp_beta_min = np.log(torch.tensor(sigma_max).cpu() ** 2 + 1) - 0.5 * vp_beta_d
        t_steps_temp = (1 + step_indices / (num_steps - 1) * (epsilon_s ** (1 / schedule_rho) - 1)) ** schedule_rho
        t_steps = vp_sigma(vp_beta_d.clone().detach().cpu(), vp_beta_min.clone().detach().cpu())(t_steps_temp.clone().detach().cpu())
    elif schedule_type == 'discrete':
        assert net is not None
        t_steps_min = net.sigma_inv(torch.tensor(sigma_min, device=device))
        t_steps_max = net.sigma_inv(torch.tensor(sigma_max, device=device))
        step_indices = torch.arange(num_steps, device=device)
        t_steps_temp = (t_steps_max + step_indices / (num_steps - 1) * (t_steps_min ** (1 / schedule_rho) - t_steps_max)) ** schedule_rho
        t_steps = net.sigma(t_steps_temp)
    else:
        raise ValueError("Got wrong schedule type {}".format(schedule_type))
    
    if dp_list is not None:
        return t_steps[dp_list].to(device)
    return t_steps.to(device)


    