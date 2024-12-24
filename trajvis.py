from modules.modules import CondVF, Net
from modules.utils import get_data, get_model, plot_losses,get_time_discretization
import torch
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import numpy as np

torch.manual_seed(0)

def run_sampling_and_plotting(dataset_name, trajectories, **kwargs):
    params = {**kwargs}
    n_points = params.get("n_points")
    n_samples = params.get("n_samples")
    n_trajectory = params.get("n_trajectory")
    n_epochs = params.get("n_epochs")
    schedule = params.get("schedule")
    nfe = params.get("nfe")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = get_data(dataset_name, n_points)
    fig_nums = len(trajectories) + 1
    fig, axes = plt.subplots(1, fig_nums, figsize=(fig_nums*3+3, 3), sharex='all', sharey='all')
    
    # Plot training data
    axes[0].hist2d(data[:, 0], data[:, 1], bins=164, cmap='viridis', range=[[-2, 2], [-2, 2]])
    axes[0].set_title(f'Training: {dataset_name}')
    
    for i, model_name in enumerate(trajectories):
        net = Net(2, 2, [512]*5, 10).to(device)
        # net = Net(2, 2, [1024]*10, 10).to(device)
        v_t = CondVF(net).to(device)
        model_path = f"models/{dataset_name}_{model_name}_{n_epochs}.pt"
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, weights_only=True)
            v_t.load_state_dict(state_dict)
            print(f"Loaded pretrained model for {model_name}.")
        else:
            raise ValueError(f"Model {model_name} does not exist for dataset {dataset_name}.")
        
        # sampling
        v_t.eval()
        time_grid = get_time_discretization(nfe, schedule, method = 'euler')
        
        with torch.no_grad():
            # init random noise
            
            x_0 = torch.randn(n_samples, 2, device=device)
            
            if model_name == "ve-smld" or model_name == "ve-edm":# init not with unit variance.
                sigma_max = 2.
                x_0 = x_0 * sigma_max
            # Track the sampling trajectory
            
            trajectory = v_t.eulersolver(x_0,time_grid).cpu().numpy()
            # print(trajectory.shape)
            # input("11")
            
        
        # Plot the trajectory for this model
        axes[i+1].hist2d(trajectory[-1, :, 0], trajectory[-1, :, 1], bins=164, cmap='viridis', range=[[-2, 2], [-2, 2]])
        
        # randomly choose n_trajectory trajectories idx
        indices = np.random.choice(n_samples, n_trajectory, replace=False)
        dot_size = 6
        line_width = 1
        count = 0
        for idx in indices:
            if count == 0:
                axes[i+1].scatter(trajectory[0, idx, 0], trajectory[0, idx, 1], s=dot_size, c='y', label="Start")
                axes[i+1].scatter(trajectory[-1, idx, 0], trajectory[-1, idx, 1], s=dot_size, c='w', label="End")
                axes[i+1].plot(trajectory[:, idx, 0], trajectory[:, idx, 1], c='r', linewidth=line_width, label="Trajectory")
                
            else:
                axes[i+1].scatter(trajectory[0, idx, 0], trajectory[0, idx, 1], s=dot_size, c='y')
                axes[i+1].scatter(trajectory[-1, idx, 0], trajectory[-1, idx, 1], s=dot_size, c='w')
                axes[i+1].plot(trajectory[:, idx, 0], trajectory[:, idx, 1], c='r', linewidth=line_width)
            count += 1
        
        axes[i+1].set_title(f'{model_name}')
        axes[i+1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f"trajectory_visual/trajectory_{dataset_name}.png")
    plt.close()


if __name__ == "__main__":
    
    datasets = ["moons", "circles", "blobs", "noisy_circles", "s_curve", "classification", "swiss"]
    # datasets = ["moons"]
    
    trajectories = ["ve-smld","ve-edm", "vp", "subvp", "ot","cosine"]
    # trajectories = ["cosine"]
    kwargs_sampling = {
        "n_points": 30_000,
        "n_samples": 30_000,
        "n_epochs": 200,
        "n_trajectory": 50,
        "schedule": 'time_uniform',
        "nfe": 200,
    }
    
    
    for dataset in datasets:
        run_sampling_and_plotting(dataset, trajectories, **kwargs_sampling) # kwargs_debug or kwargs_training
        print(f"Finished plotting for {dataset}.")




