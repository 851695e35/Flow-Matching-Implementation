from modules.modules import CondVF, Net
from modules.utils import get_data, get_model, plot_losses,get_time_discretization
import torch
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import numpy as np

torch.manual_seed(0)

def run_training_and_plotting(dataset_name, trajectories, **kwargs):
    params = {**kwargs}
    n_points = params.get("n_points")
    n_samples = params.get("n_samples")
    n_epochs = params.get("n_epochs")
    batch_size = params.get("batch_size")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = get_data(dataset_name, n_points)
    dataset = torch.from_numpy(data).float().to(device)
    dataloader = DataLoader(TensorDataset(dataset), batch_size=batch_size)
    
    losses_dict = {}
    
    for i, model_name in enumerate(trajectories):
        model = get_model(model_name)
        # net = Net(2, 2, [1024]*10, 10).to(device)
        net = Net(2, 2, [512]*5, 10).to(device)
        v_t = CondVF(net).to(device)
        model_path = f"models/{dataset_name}_{model_name}_{n_epochs}.pt"
        

        optimizer = torch.optim.Adam(v_t.parameters(), lr=1e-3)
        losses = []

        pbar = tqdm(range(n_epochs), ncols=80, desc=f"Training {model_name} on {dataset_name}")
        for epoch in pbar:
            batch_losses = []
            for x_1, in dataloader:
                loss = model.loss(v_t, x_1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.detach().item())
            
            average_loss = sum(batch_losses) / len(batch_losses)
            losses.append(average_loss)
            pbar.set_postfix({"Loss": f"{losses[-1]:.4f}"})

        torch.save(v_t.state_dict(), model_path)
        losses_dict[model_name] = losses 


    # save the training plot
    plot_losses(losses_dict, dataset_name)


if __name__ == "__main__":
    
    datasets = ["moons", "circles", "blobs", "noisy_circles", "s_curve", "classification", "swiss"]
    # datasets = ["moons"]
    
    trajectories = ["ve-smld","ve-edm", "vp", "subvp", "ot","cosine"]
    # trajectories = ["cosine"]
    kwargs_training = {
        "n_points": 30_000,
        "n_samples": 30_000,
        "n_epochs": 200,
        "batch_size": 2048,
    }
    kwargs_debug = {
        "n_points": 100,
        "n_samples": 100,
        "n_epochs": 10,
        "batch_size": 2048,        
    }
    
    
    for dataset in datasets:
        run_training_and_plotting(dataset, trajectories, use_pretrained=False, **kwargs_training) # kwargs_debug or kwargs_training

