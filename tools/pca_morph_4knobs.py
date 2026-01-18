# Copyright (c) 2026
# PCA-based morph with N semantic knobs (each knob = one PCA comp + layer mask).
# Generates frames and (optionally) encodes mp4 via ffmpeg.
# Updated: write params.jsonl while generating frames (one JSON per frame).
# Updated2: support ANY number of knobs via --knob-layers parts count.
#          --knob-comps can be 1 int (broadcast) or N ints.
#          --a/--b must be N floats matching knob count.

import os
import re
import json
import shutil
import subprocess
from typing import List, Optional, Tuple, Union, Dict

import click
import numpy as np
import PIL.Image
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
    out: List[int] = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in (s or "").split(","):
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
    "all" -> None
    "0-5,8,10-12" -> [...]
    """
    s = (s or "").strip().lower()
    if s in ["all", "*", "none", ""]:
        return None
    return parse_range(s)

def make_ws_layer_mask(num_ws: int, layers: Optional[List[int]]) -> torch.Tensor:
    """mask: [1, num_ws, 1]"""
    mask = torch.zeros([1, num_ws, 1], dtype=torch.float32)
    if layers is None:
        mask[:] = 1.0
        return mask
    for i in layers:
        if i < 0 or i >= num_ws:
            raise click.ClickException(f"Layer index {i} out of range [0, {num_ws-1}]")
        mask[0, i, 0] = 1.0
    return mask

def smoothstep(t: float) -> float:
    # classic smoothstep: 3t^2 - 2t^3
    return t * t * (3.0 - 2.0 * t)

def lerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    return a + (b - a) * t

def load_pca_npz(npz_path: str) -> Dict[str, np.ndarray]:
    data = np.load(npz_path)
    return {k: data[k] for k in data.keys()}

def get_dirs_evals_sigma(pca: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    keys = list(pca.keys())
    # Your npz uses: mean, evals, sigma, dirs
    vec_candidates = ["dirs", "eigvecs", "eigvec", "V", "components", "directions", "U"]
    val_candidates = ["evals", "eigvals", "eigval", "D", "values", "lambdas", "S"]
    sig_candidates = ["sigma", "std", "stdev"]

    vec = None
    val = None
    sig = None

    for kk in vec_candidates:
        if kk in keys:
            vec = pca[kk]
            break
    for kk in val_candidates:
        if kk in keys:
            val = pca[kk]
            break
    for kk in sig_candidates:
        if kk in keys:
            sig = pca[kk]
            break

    # fallback by shapes
    if vec is None:
        for kk in keys:
            arr = pca[kk]
            if arr.ndim == 2 and arr.shape[1] >= 16:
                vec = arr
                break
    if vec is None:
        raise click.ClickException(f"Cannot find PCA directions in {keys}")

    if val is None:
        val = np.arange(vec.shape[0], dtype=np.float32)

    vec = np.asarray(vec).astype(np.float32)
    val = np.asarray(val).astype(np.float32)
    if sig is not None:
        sig = np.asarray(sig).astype(np.float32)

    if vec.ndim != 2:
        raise click.ClickException(f"Directions must be 2D, got {vec.shape}")
    return vec, val, sig

def ensure_dir_shape(dirs: np.ndarray, w_dim: int) -> np.ndarray:
    # want [k, w_dim]
    if dirs.shape[1] == w_dim:
        return dirs
    if dirs.shape[0] == w_dim and dirs.shape[1] != w_dim:
        return dirs.T
    raise click.ClickException(f"Direction shape mismatch: dirs={dirs.shape}, expected (*,{w_dim})")

def direction_tensor(dirs: np.ndarray, comp: int, w_dim: int, num_ws: int, device, normalize: bool) -> torch.Tensor:
    d = dirs[comp].copy()
    if normalize:
        d = d / (np.linalg.norm(d) + 1e-8)
    d_t = torch.from_numpy(d.reshape(1, 1, w_dim)).to(device)  # [1,1,w_dim]
    return d_t.repeat(1, num_ws, 1)  # [1,num_ws,w_dim]

def synth_image(G, ws: torch.Tensor, noise_mode: str) -> PIL.Image.Image:
    img = G.synthesis(ws, noise_mode=noise_mode)  # [1,C,H,W]
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return PIL.Image.fromarray(img[0].cpu().numpy(), "RGB")

def ffmpeg_encode(frames_dir: str, fps: int, out_mp4: str) -> bool:
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "frame_%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        out_mp4
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError as e:
        print("[ffmpeg] failed:")
        try:
            print(e.stderr.decode("utf-8", errors="ignore")[:2000])
        except Exception:
            pass
        return False

def parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in (s or "").split(",") if x.strip() != ""]

def parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in (s or "").split(",") if x.strip() != ""]


# -------------------------
# CLI
# -------------------------

@click.command()
@click.option("--network", "network_pkl", required=True, help="Network pickle path or URL")
@click.option("--pca", "pca_npz", required=True, help="PCA npz, e.g. directions/pca_w_xxx.npz")

@click.option("--seed-a", type=int, default=0, show_default=True)
@click.option("--seed-b", type=int, default=1, show_default=True)
@click.option("--trunc", "truncation_psi", type=float, default=1.0, show_default=True)
@click.option("--noise-mode", type=click.Choice(["const", "random", "none"]), default="const", show_default=True)
@click.option("--class", "class_idx", type=int, default=None, help="Class label index (for conditional nets)")

# knob config:
#   knob count N is defined by number of '|' parts in --knob-layers.
#   --knob-comps can be 1 int (broadcast) or N ints.
@click.option("--knob-comps", type=str, default="3", show_default=True,
              help="PCA comp index for each knob. Either 1 int (broadcast) or N ints, e.g. '3' or '3,3,1,7'")
@click.option("--knob-layers", type=str, required=True,
              help="N layer ranges separated by '|', e.g. '0|1-6|7-9|10-13|14|15'")

# knob values for A and B (must be N floats)
@click.option("--a", "a_vals", type=str, required=True,
              help="N floats for A, matching knob count, e.g. '0,2,4,2,1,0'")
@click.option("--b", "b_vals", type=str, required=True,
              help="N floats for B, matching knob count, e.g. '0,-2,-4,2,-1,10'")

@click.option("--steps", type=int, default=120, show_default=True, help="Number of frames")
@click.option("--fps", type=int, default=30, show_default=True)
@click.option("--smooth", is_flag=True, help="Use smoothstep easing (recommended)")

@click.option("--normalize-dir", is_flag=True, help="Normalize each PCA direction to unit norm before applying")
@click.option("--use-sigma", is_flag=True,
              help="If npz contains sigma/std, scale each knob value by sigma[comp] (values become 'in std units')")

@click.option("--outdir", type=str, required=True, help="Output dir")
@click.option("--mp4", "out_mp4", type=str, default="", show_default=True,
              help="If set, encode mp4 to this path (relative to outdir allowed)")
@click.option("--keep-frames/--no-keep-frames", default=True, show_default=True,
              help="Keep png frames after encoding")

@click.option("--params-jsonl", type=str, default="params.jsonl", show_default=True,
              help="Write one JSON object per frame. Relative paths are resolved under outdir.")
def main(
    network_pkl: str,
    pca_npz: str,
    seed_a: int,
    seed_b: int,
    truncation_psi: float,
    noise_mode: str,
    class_idx: Optional[int],
    knob_comps: str,
    knob_layers: str,
    a_vals: str,
    b_vals: str,
    steps: int,
    fps: int,
    smooth: bool,
    normalize_dir: bool,
    use_sigma: bool,
    outdir: str,
    out_mp4: str,
    keep_frames: bool,
    params_jsonl: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'Loading network: "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)
    frames_dir = os.path.join(outdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # label
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException("Conditional network: must set --class")
        label[:, class_idx] = 1

    # load PCA
    pca = load_pca_npz(pca_npz)
    dirs, evals, sigma = get_dirs_evals_sigma(pca)

    # sample once to know dims
    def get_ws(seed: int) -> torch.Tensor:
        rnd = np.random.RandomState(seed)
        z = torch.from_numpy(rnd.randn(1, G.z_dim).astype(np.float32)).to(device)
        ws = G.mapping(z, label, truncation_psi=truncation_psi)  # [1,num_ws,w_dim]
        return ws

    wsA = get_ws(seed_a)
    wsB = get_ws(seed_b)

    num_ws = wsA.shape[1]
    w_dim = wsA.shape[2]

    dirs = ensure_dir_shape(dirs, w_dim)

    # knob layers define knob count
    knob_layer_parts = [p.strip() for p in knob_layers.split("|") if p.strip() != ""]
    if len(knob_layer_parts) < 1:
        raise click.ClickException("--knob-layers must have at least 1 part, e.g. '0|1-6|7-9'")

    knob_layer_lists = [parse_layers(part) for part in knob_layer_parts]
    n_knobs = len(knob_layer_lists)

    # parse comps
    knob_comp_list_in = parse_csv_ints(knob_comps)
    if len(knob_comp_list_in) == 1:
        knob_comp_list = knob_comp_list_in * n_knobs
    elif len(knob_comp_list_in) == n_knobs:
        knob_comp_list = knob_comp_list_in
    else:
        raise click.ClickException(
            f"--knob-comps must have 1 int (broadcast) or N ints (N={n_knobs}). "
            f"Got {len(knob_comp_list_in)} ints."
        )

    # parse A/B
    A = parse_csv_floats(a_vals)
    B = parse_csv_floats(b_vals)
    if len(A) != n_knobs or len(B) != n_knobs:
        raise click.ClickException(
            f"--a/--b must have N floats matching knob count N={n_knobs}. "
            f"Got len(a)={len(A)}, len(b)={len(B)}."
        )

    # Prebuild per-knob direction tensors + masks
    knob_dir_t: List[torch.Tensor] = []
    knob_mask_t: List[torch.Tensor] = []
    knob_sigma: List[float] = []

    for i in range(n_knobs):
        comp = knob_comp_list[i]
        if comp < 0 or comp >= dirs.shape[0]:
            raise click.ClickException(f"Knob{i+1} comp={comp} out of range 0..{dirs.shape[0]-1}")
        d_t = direction_tensor(dirs, comp, w_dim, num_ws, device, normalize_dir)
        m_t = make_ws_layer_mask(num_ws, knob_layer_lists[i]).to(device)
        knob_dir_t.append(d_t)
        knob_mask_t.append(m_t)

        if sigma is not None and comp < sigma.shape[0]:
            knob_sigma.append(float(sigma[comp]))
        else:
            knob_sigma.append(1.0)

    def apply_knobs(ws: torch.Tensor, vals: List[float]) -> torch.Tensor:
        out = ws.clone()
        for i in range(n_knobs):
            v = float(vals[i])
            if use_sigma:
                v = v * knob_sigma[i]
            out = out + (v * knob_dir_t[i] * knob_mask_t[i])
        return out

    wsA_edit = apply_knobs(wsA, A)
    wsB_edit = apply_knobs(wsB, B)

    # Save config for reproducibility
    cfg = {
        "network": network_pkl,
        "pca": pca_npz,
        "seed_a": seed_a,
        "seed_b": seed_b,
        "trunc": truncation_psi,
        "noise_mode": noise_mode,
        "knob_comps": knob_comp_list,
        "knob_layers": knob_layer_parts,
        "A": A,
        "B": B,
        "steps": steps,
        "fps": fps,
        "smooth": smooth,
        "normalize_dir": normalize_dir,
        "use_sigma": use_sigma,
    }
    with open(os.path.join(outdir, "morph_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # Resolve params.jsonl path
    params_path = params_jsonl
    if not os.path.isabs(params_path):
        params_path = os.path.join(outdir, params_path)

    # Generate frames + params.jsonl
    print(f"Generating {steps} frames to: {frames_dir}")
    print(f"Writing params.jsonl to: {params_path}")

    os.makedirs(os.path.dirname(params_path) or ".", exist_ok=True)
    with open(params_path, "w", encoding="utf-8") as jf:
        for fi in range(steps):
            t_linear = 0.0 if steps <= 1 else fi / (steps - 1)
            t_used = smoothstep(t_linear) if smooth else t_linear

            ws_t = lerp(wsA_edit, wsB_edit, t_used)
            img = synth_image(G, ws_t, noise_mode=noise_mode)
            frame_name = f"frame_{fi:06d}.png"
            img.save(os.path.join(frames_dir, frame_name))

            # record knob values along the interpolation path (interpretation helper)
            knob_vals_interp_linear = [A[i] + (B[i] - A[i]) * t_linear for i in range(n_knobs)]
            knob_vals_interp_used = [A[i] + (B[i] - A[i]) * t_used for i in range(n_knobs)]

            rec = {
                "frame": fi,
                "frame_file": os.path.join("frames", frame_name).replace("\\", "/"),
                "t_linear": float(t_linear),
                "t_used": float(t_used),
                "smooth": bool(smooth),

                "seed_a": int(seed_a),
                "seed_b": int(seed_b),
                "trunc": float(truncation_psi),
                "noise_mode": noise_mode,

                "knob_count": int(n_knobs),
                "knob_comps": knob_comp_list,
                "knob_layers": knob_layer_parts,
                "A": A,
                "B": B,

                "knob_vals_interp_linear": knob_vals_interp_linear,
                "knob_vals_interp_used": knob_vals_interp_used,

                "use_sigma": bool(use_sigma),
                "sigma_per_knob": [float(x) for x in knob_sigma],
                "effective_A": [float(A[i] * (knob_sigma[i] if use_sigma else 1.0)) for i in range(n_knobs)],
                "effective_B": [float(B[i] * (knob_sigma[i] if use_sigma else 1.0)) for i in range(n_knobs)],
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if fi % max(1, steps // 10) == 0:
                print(f"  frame {fi+1}/{steps}")

    # Encode mp4 (optional)
    if out_mp4:
        mp4_path = out_mp4
        if not (mp4_path.lower().endswith(".mp4")):
            mp4_path += ".mp4"
        if not os.path.isabs(mp4_path):
            mp4_path = os.path.join(outdir, mp4_path)

        print(f"Encoding mp4: {mp4_path}")
        ok = ffmpeg_encode(frames_dir, fps, mp4_path)
        if ok:
            print("Saved mp4:", mp4_path)
            if not keep_frames:
                shutil.rmtree(frames_dir, ignore_errors=True)
                print("Removed frames:", frames_dir)
        else:
            print("[warn] ffmpeg not available or failed. Frames kept at:", frames_dir)
            print("       You can encode on host with:")
            print(f"       ffmpeg -y -framerate {fps} -i frame_%06d.png -c:v libx264 -pix_fmt yuv420p -crf 18 out.mp4")

    print("Done.")


if __name__ == "__main__":
    main()
