import os
import numpy as np
import torch
import click
import dnnlib
import legacy

@torch.no_grad()
def sample_w(G, device, n_samples, seed=0, truncation_psi=1.0, class_idx=None):
    rnd = np.random.RandomState(seed)
    z = torch.from_numpy(rnd.randn(n_samples, G.z_dim)).to(device)

    label = torch.zeros([n_samples, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise ValueError("Conditional network: must set class_idx")
        label[:, class_idx] = 1

    w = G.mapping(z, label, truncation_psi=truncation_psi)  # [N, w_dim] or [N, num_ws, w_dim] depending impl
    if w.ndim == 3:
        # Some impl returns ws directly; collapse to W by taking mean over layers
        w = w.mean(dim=1)
    return w.float().cpu().numpy()  # [N, w_dim]

def pca_directions(X, k=20):
    # X: [N, D]
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    # covariance
    C = (Xc.T @ Xc) / (Xc.shape[0] - 1)  # [D, D]
    # eigen decomposition (symmetric)
    evals, evecs = np.linalg.eigh(C)      # ascending
    idx = np.argsort(evals)[::-1]         # descending
    evals = evals[idx]
    evecs = evecs[:, idx]                 # columns are directions
    return mu[0], evals, evecs[:, :k]

@click.command()
@click.option("--network", "network_pkl", required=True)
@click.option("--out", "out_path", required=True, help="Where to save npz with PCA directions")
@click.option("--n", "n_samples", default=20000, show_default=True)
@click.option("--k", "k", default=50, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--trunc", "truncation_psi", default=1.0, show_default=True)
@click.option("--class", "class_idx", default=None, type=int)
def main(network_pkl, out_path, n_samples, k, seed, truncation_psi, class_idx):
    device = torch.device("cuda")
    print(f'Loading "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)

    print("Sampling W...")
    W = sample_w(G, device, n_samples=n_samples, seed=seed, truncation_psi=truncation_psi, class_idx=class_idx)
    print("Running PCA...")
    mu, evals, dirs = pca_directions(W, k=k)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sigma = np.sqrt(np.maximum(evals, 1e-12))
    np.savez(out_path, mean=mu, evals=evals, sigma=sigma, dirs=dirs)
    print(f"Saved: {out_path}")
    print("Top-10 evals:", evals[:10])

if __name__ == "__main__":
    main()
