import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs, make_s_curve, make_classification
from sklearn.preprocessing import StandardScaler
from modules.Trajectory import OTFlowMatching, VPDiffusionFlowMatching, VEDiffusionFlowMatching, subVPDiffusionFlowMatching
import matplotlib.pyplot as plt
def get_data(dataset: str, n_points: int) -> np.ndarray:
    if dataset == "moons":
        data, _ = make_moons(n_points, noise=0.15)
    elif dataset == "circles":
        data, _ = make_circles(n_points, noise=0.1, factor=0.5)
    elif dataset == "blobs":
        data, _ = make_blobs(n_points, centers=3)
    elif dataset == "noisy_circles":
        data, _ = make_circles(n_points, noise=0.2, factor=0.5)
    elif dataset == "s_curve":
        data, _ = make_s_curve(n_points, noise=0.1)
        data = data[:, [0, 2]]  # Use the first and last dimension to get a 2D view
    elif dataset == "classification":
        data, _ = make_classification(n_points, n_features=2, n_redundant=0, n_clusters_per_class=1, flip_y=0.1, n_classes=2)
    elif dataset == "swiss":
        from sklearn.datasets import make_swiss_roll
        data, _ = make_swiss_roll(n_points, noise=0.15)
        data = data[:, [0, 2]] / 10.0  # Rescaling to reduce the spread
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    return StandardScaler().fit_transform(data)
    


def get_model(name: str):
    if name == "ot":
        return OTFlowMatching()
    elif name == "vp":
        return VPDiffusionFlowMatching()
    elif name == "ve":
        return VEDiffusionFlowMatching()
    elif name == "subvp":
        return subVPDiffusionFlowMatching()
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
    plt.savefig(f"toy_example/loss_{dataset_name}.png")
    plt.close()  