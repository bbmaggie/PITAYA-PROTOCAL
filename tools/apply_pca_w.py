# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.
# Apply PCA directions (computed in W space) to generate edited images.
# Extended: sweep comps + layers and export grids.

import os
import re
from typing import List, Optional, Tuple, Union

import click
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import torch

import dnnlib
import legacy


# -------------------------
# helpers
# -------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
    """'1,2,5-7' -> [1,2,5,6,7]"""
    if isinstance(s, list):
        return s
    out = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        m = range_re.match(p)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            out.extend(list(range(a, b + 1)))
        else:
            out.append(int(p))
    return out

def parse_layers(s: str) -> Optional[List[int]]:
    """
    Examples:
      "all" -> None
      "0,1,2,3" -> [0,1,2,3]
      "0-5,8,10-12" -> [...]
    Layer index means ws index (0..num_ws-1)
    """
    s = (s or "").strip().lower()
    if s in ["all", "*", "none", ""]:
        return None
    return parse_range(s)

def parse_strengths(s: str) -> List[float]:
    return [float(x.strip()) for x in (s or "").split(",") if x.strip() != ""]

def load_pca_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load eigenvectors/eigenvalues from npz saved by tools/pca_w_directions.py.
    Returns:
      eigvecs: [k, w_dim]  (each row is a direction)
      eigvals: [k]
    """
    data = np.load(npz_path)
    keys = list(data.keys())

    vec_candidates = ["dirs", "eigvecs", "eigvec", "V", "components", "directions", "U"]
    val_candidates = ["evals", "eigvals", "eigval", "D", "values", "lambdas", "S"]

    vec = None
    val = None

    for kk in vec_candidates:
        if kk in keys:
            vec = data[kk]
            break
    for kk in val_candidates:
        if kk in keys:
            val = data[kk]
            break

    # Fallback by shapes
    if vec is None:
        for kk in keys:
            arr = data[kk]
            if arr.ndim == 2 and arr.shape[1] >= 16:
                vec = arr
                break
    if val is None and vec is not None:
        for kk in keys:
            arr = data[kk]
            if arr.ndim == 1 and arr.shape[0] == vec.shape[0]:
                val = arr
                break

    if vec is None:
        raise click.ClickException(f"Cannot find eigenvectors in {npz_path}. Keys={keys}")
    if val is None:
        val = np.arange(vec.shape[0], dtype=np.float32)

    vec = np.asarray(vec).astype(np.float32)
    val = np.asarray(val).astype(np.float32)

    if vec.ndim != 2:
        raise click.ClickException(f"Eigenvectors must be 2D. Got shape={vec.shape}")

    return vec, val

def make_ws_layer_mask(num_ws: int, layers: Optional[List[int]]) -> torch.Tensor:
    """Return mask shaped [1, num_ws, 1] with 1 for selected layers."""
    mask = torch.zeros([1, num_ws, 1], dtype=torch.float32)
    if layers is None:
        mask[:] = 1.0
        return mask
    for i in layers:
        if i < 0 or i >= num_ws:
            raise click.ClickException(f"Layer index {i} out of range [0, {num_ws-1}]")
        mask[0, i, 0] = 1.0
    return mask

def _try_get_font(size: int = 16):
    # No external fonts guaranteed inside docker; use default.
    try:
        return PIL.ImageFont.load_default()
    except Exception:
        return None

def make_grid(
    images: List[List[PIL.Image.Image]],
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    pad: int = 8,
    label_h: int = 22,
) -> PIL.Image.Image:
    """
    images: [rows][cols] PIL images (same size)
    """
    assert len(images) == len(row_labels)
    assert len(images[0]) == len(col_labels)

    rows = len(images)
    cols = len(images[0])
    w, h = images[0][0].size

    # Canvas sizes:
    # extra top for title + col labels, extra left for row labels
    top = label_h * 2 + pad  # title + col labels
    left = 120               # row label area
    grid_w = left + cols * w + (cols + 1) * pad
    grid_h = top + rows * h + (rows + 1) * pad

    canvas = PIL.Image.new("RGB", (grid_w, grid_h), (20, 20, 20))
    draw = PIL.ImageDraw.Draw(canvas)
    font = _try_get_font()

    # Title
    draw.text((pad, pad), title, fill=(235, 235, 235), font=font)

    # Col labels (strengths)
    for c, lab in enumerate(col_labels):
        x = left + pad + c * (w + pad)
        y = label_h + pad
        draw.text((x, y), lab, fill=(220, 220, 220), font=font)

    # Rows
    for r in range(rows):
        # Row label (layer idx)
        y0 = top + pad + r * (h + pad)
        draw.text((pad, y0), row_labels[r], fill=(220, 220, 220), font=font)

        for c in range(cols):
            x0 = left + pad + c * (w + pad)
            canvas.paste(images[r][c], (x0, y0))

    return canvas


# -------------------------
# main
# -------------------------

@click.command()
@click.option("--network", "network_pkl", required=True, help="Network pickle path or URL")
@click.option("--pca", "pca_npz", required=True, help="PCA directions npz, e.g. directions/pca_w_xxx.npz")
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--class", "class_idx", type=int, default=None, help="Class label index (for conditional nets)")
@click.option("--trunc", "truncation_psi", type=float, default=1.0, show_default=True)
@click.option("--noise-mode", type=click.Choice(["const", "random", "none"]), default="const", show_default=True)

# --- single mode (compatible with your old behavior)
@click.option("--comp", "comp_idx", type=int, default=0, show_default=True, help="Which PCA component (0 = strongest)")
@click.option("--layers", type=str, default="all", show_default=True,
              help='Which ws layers to edit, e.g. "all" or "0-5,8,10-12"')
@click.option("--strengths", type=str, default="-5,-3,-1,0,1,3,5", show_default=True,
              help='Comma list of strengths, e.g. "-5,-3,-1,0,1,3,5"')

# --- sweep mode (方案3)
@click.option("--sweep", is_flag=True, help="Enable sweep: for each comp, make a grid (rows=layers, cols=strengths)")
@click.option("--comp-range", type=str, default="0-5", show_default=True,
              help='When --sweep: which comps to scan, e.g. "0-5" or "0,1,2"')
@click.option("--layer-range", type=str, default="all", show_default=True,
              help='When --sweep: which layers to scan, e.g. "all" or "0-15" or "0,2,4"')
@click.option("--grid-pad", type=int, default=8, show_default=True, help="Grid padding in pixels")
@click.option("--save-tiles", is_flag=True, help="When --sweep: also save every tile png (can be many)")

@click.option("--outdir", type=str, required=True)
@click.option("--normalize", is_flag=True, help="Normalize direction to unit L2 norm before applying")
def main(
    network_pkl: str,
    pca_npz: str,
    seed: int,
    class_idx: Optional[int],
    truncation_psi: float,
    noise_mode: str,
    comp_idx: int,
    layers: str,
    strengths: str,
    sweep: bool,
    comp_range: str,
    layer_range: str,
    grid_pad: int,
    save_tiles: bool,
    outdir: str,
    normalize: bool
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'Loading network: "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # label
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException("Conditional network: must set --class")
        label[:, class_idx] = 1

    # load PCA
    eigvecs, eigvals = load_pca_npz(pca_npz)

    # Make sure eigvecs is [k, w_dim]
    if eigvecs.shape[0] == G.w_dim and eigvecs.shape[1] != G.w_dim:
        eigvecs = eigvecs.T

    k, w_dim = eigvecs.shape
    print(f"PCA loaded: k={k}, w_dim={w_dim}")

    # sample z and map to ws once
    rnd = np.random.RandomState(seed)
    z = torch.from_numpy(rnd.randn(1, G.z_dim).astype(np.float32)).to(device)
    ws = G.mapping(z, label, truncation_psi=truncation_psi)  # [1, num_ws, w_dim_net]
    num_ws = ws.shape[1]
    w_dim_net = ws.shape[2]

    # If PCA is for w_dim=512 but somehow mismatch, fail early
    if w_dim != w_dim_net:
        raise click.ClickException(f"PCA w_dim={w_dim} but network w_dim={w_dim_net}. (Your PCA must match the trained net.)")

    strength_list = parse_strengths(strengths)

    # -------------------------
    # sweep mode: grids per comp
    # -------------------------
    if sweep:
        comp_list = parse_range(comp_range)
        layer_list = parse_layers(layer_range)
        if layer_list is None:
            layer_list = list(range(num_ws))

        # Precompute strengths labels
        col_labels = [f"s={s:g}" for s in strength_list]
        row_labels = [f"layer {li}" for li in layer_list]

        for ci in comp_list:
            if ci < 0 or ci >= k:
                print(f"[warn] skip comp {ci} (out of range 0..{k-1})")
                continue

            direction = eigvecs[ci].copy()
            if normalize:
                direction = direction / (np.linalg.norm(direction) + 1e-8)

            print(f"\n[pc{ci:02d}] eigval={eigvals[ci]:.6f} scanning layers={len(layer_list)} strengths={len(strength_list)}")

            # Prepare dir tensor
            dir_t = torch.from_numpy(direction.reshape(1, 1, w_dim)).to(device)  # [1,1,w_dim]
            dir_t = dir_t.repeat(1, num_ws, 1)                                   # [1,num_ws,w_dim]

            # images grid: rows=layers, cols=strengths
            grid_images: List[List[PIL.Image.Image]] = []

            pc_dir = os.path.join(outdir, f"pc{ci:02d}")
            os.makedirs(pc_dir, exist_ok=True)

            for li in layer_list:
                # mask for single layer
                mask = make_ws_layer_mask(num_ws, [li]).to(device)  # [1,num_ws,1]
                row_imgs: List[PIL.Image.Image] = []

                for s in strength_list:
                    ws_edit = ws + (s * dir_t * mask)
                    img = G.synthesis(ws_edit, noise_mode=noise_mode)  # [1,C,H,W]
                    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                    img_pil = PIL.Image.fromarray(img[0].cpu().numpy(), "RGB")

                    if save_tiles:
                        tile_name = f"seed{seed:04d}_pc{ci:02d}_layer{li:02d}_s{str(s).replace('.', 'p')}.png"
                        img_pil.save(os.path.join(pc_dir, tile_name))

                    row_imgs.append(img_pil)

                grid_images.append(row_imgs)

            title = f"seed={seed}  pc={ci:02d} (eigval={eigvals[ci]:.4g})  cols=strengths  rows=layers"
            grid = make_grid(
                images=grid_images,
                row_labels=row_labels,
                col_labels=col_labels,
                title=title,
                pad=grid_pad,
            )
            grid_path = os.path.join(pc_dir, f"grid_pc{ci:02d}.png")
            grid.save(grid_path)
            print("Saved grid:", grid_path)

        return

    # -------------------------
    # single mode: same as your old behavior
    # -------------------------
    if comp_idx < 0 or comp_idx >= k:
        raise click.ClickException(f"--comp out of range. comp={comp_idx}, available 0..{k-1}")

    direction = eigvecs[comp_idx].copy()
    if normalize:
        direction = direction / (np.linalg.norm(direction) + 1e-8)

    print(f"Using component {comp_idx}, eigval={eigvals[comp_idx]:.6f}")

    layer_list = parse_layers(layers)
    mask = make_ws_layer_mask(num_ws, layer_list).to(device)  # [1,num_ws,1]
    dir_t = torch.from_numpy(direction.reshape(1, 1, w_dim)).to(device)  # [1,1,w_dim]
    dir_t = dir_t.repeat(1, num_ws, 1)  # [1,num_ws,w_dim]

    for s in strength_list:
        ws_edit = ws + (s * dir_t * mask)
        img = G.synthesis(ws_edit, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_pil = PIL.Image.fromarray(img[0].cpu().numpy(), "RGB")

        layers_tag = "all" if layer_list is None else f"layers_{layers.replace(',', '_').replace('-', 'to')}"
        fn = f"seed{seed:04d}_pc{comp_idx:02d}_s{str(s).replace('.', 'p')}_{layers_tag}.png"
        save_path = os.path.join(outdir, fn)
        img_pil.save(save_path)
        print("Saved:", save_path)


if __name__ == "__main__":
    main()
