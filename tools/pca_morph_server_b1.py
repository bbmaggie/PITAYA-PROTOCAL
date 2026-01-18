# Copyright (c) 2026
# B1: File-based single-frame "server" for TouchDesigner.
# - Loads StyleGAN3 + PCA once (keeps on GPU)
# - Watches ipc/request.json for changes
# - Renders ONE frame and writes ipc/latest.png (atomic replace)

import os
import re
import json
import time
from typing import List, Optional, Dict, Tuple, Union

import numpy as np
import PIL.Image
import torch
import click

import dnnlib
import legacy


# -------------------------
# helpers (mostly copied from your script)
# -------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
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
    s = (s or "").strip().lower()
    if s in ["all", "*", "none", ""]:
        return None
    return parse_range(s)

def make_ws_layer_mask(num_ws: int, layers: Optional[List[int]], device) -> torch.Tensor:
    """mask: [1, num_ws, 1]"""
    mask = torch.zeros([1, num_ws, 1], dtype=torch.float32, device=device)
    if layers is None:
        mask[:] = 1.0
        return mask
    for i in layers:
        if i < 0 or i >= num_ws:
            raise ValueError(f"Layer index {i} out of range [0, {num_ws-1}]")
        mask[0, i, 0] = 1.0
    return mask

def load_pca_npz(npz_path: str) -> Dict[str, np.ndarray]:
    data = np.load(npz_path)
    return {k: data[k] for k in data.keys()}

def get_dirs_evals_sigma(pca: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    keys = list(pca.keys())
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

    if vec is None:
        for kk in keys:
            arr = pca[kk]
            if arr.ndim == 2 and arr.shape[1] >= 16:
                vec = arr
                break
    if vec is None:
        raise ValueError(f"Cannot find PCA directions in keys: {keys}")

    if val is None:
        val = np.arange(vec.shape[0], dtype=np.float32)

    vec = np.asarray(vec).astype(np.float32)
    val = np.asarray(val).astype(np.float32)
    if sig is not None:
        sig = np.asarray(sig).astype(np.float32)

    if vec.ndim != 2:
        raise ValueError(f"Directions must be 2D, got shape={vec.shape}")
    return vec, val, sig

def ensure_dir_shape(dirs: np.ndarray, w_dim: int) -> np.ndarray:
    # want [k, w_dim]
    if dirs.shape[1] == w_dim:
        return dirs
    if dirs.shape[0] == w_dim and dirs.shape[1] != w_dim:
        return dirs.T
    raise ValueError(f"Direction shape mismatch: dirs={dirs.shape}, expected (*,{w_dim})")

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

def atomic_write_png(pil_img: PIL.Image.Image, out_path: str):
    tmp_path = out_path + ".tmp.png"
    pil_img.save(tmp_path)  # 这时扩展名是 .png，PIL 能识别
    os.replace(tmp_path, out_path)


def atomic_write_text(text: str, out_path: str):
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp_path, out_path)


# -------------------------
# Engine (keeps everything on GPU)
# -------------------------

class MorphEngine:
    def __init__(
        self,
        network_pkl: str,
        pca_npz: str,
        knob_comps: List[int],
        knob_layers_parts: List[str],
        truncation_psi: float,
        noise_mode: str,
        class_idx: Optional[int],
        normalize_dir: bool,
        use_sigma: bool,
        device: torch.device,
    ):
        self.device = device
        self.noise_mode = noise_mode
        self.trunc_default = float(truncation_psi)
        self.use_sigma = bool(use_sigma)

        # ✅ must exist before calling get_ws()
        self.ws_cache: Dict[Tuple[int, float], torch.Tensor] = {}

        # load network
        print(f'Loading network: "{network_pkl}"...')
        with dnnlib.util.open_url(network_pkl) as f:
            self.G = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore
        self.G.eval()

        # label (conditional support)
        self.label = torch.zeros([1, self.G.c_dim], device=device)
        if self.G.c_dim != 0:
            if class_idx is None:
                raise click.ClickException("Conditional network: must set --class")
            self.label[:, class_idx] = 1

        # load PCA
        print(f'Loading PCA: "{pca_npz}"...')
        pca = load_pca_npz(pca_npz)
        dirs, _, sigma = get_dirs_evals_sigma(pca)

        # sample once to know dims
        ws0 = self.get_ws(seed=0, truncation_psi=self.trunc_default)
        self.num_ws = ws0.shape[1]
        self.w_dim = ws0.shape[2]

        dirs = ensure_dir_shape(dirs, self.w_dim)
        self.dirs = dirs
        self.sigma_arr = sigma


        # parse knob layers
        self.knob_layers_lists = [parse_layers(part) for part in knob_layers_parts]
        self.n_knobs = len(self.knob_layers_lists)

        # broadcast comps if needed
        if len(knob_comps) == 1:
            knob_comps = knob_comps * self.n_knobs
        if len(knob_comps) != self.n_knobs:
            raise click.ClickException(f"knob comps count mismatch: got {len(knob_comps)}, expected {self.n_knobs}")
        self.knob_comps = knob_comps

        # prebuild per-knob dir + mask + sigma
        self.knob_dir_t: List[torch.Tensor] = []
        self.knob_mask_t: List[torch.Tensor] = []
        self.knob_sigma: List[float] = []

        for i in range(self.n_knobs):
            comp = self.knob_comps[i]
            if comp < 0 or comp >= self.dirs.shape[0]:
                raise click.ClickException(f"Knob{i+1} comp={comp} out of range 0..{self.dirs.shape[0]-1}")

            d_t = direction_tensor(self.dirs, comp, self.w_dim, self.num_ws, device, normalize_dir)
            m_t = make_ws_layer_mask(self.num_ws, self.knob_layers_lists[i], device)

            self.knob_dir_t.append(d_t)
            self.knob_mask_t.append(m_t)

            if self.sigma_arr is not None and comp < self.sigma_arr.shape[0]:
                self.knob_sigma.append(float(self.sigma_arr[comp]))
            else:
                self.knob_sigma.append(1.0)

        # warmup (avoid first-frame stutter)
        print("Warming up synthesis...")
        with torch.no_grad():
            _ = self.render(seed=0, knobs=[0.0]*self.n_knobs, truncation_psi=self.trunc_default, noise_mode=self.noise_mode)
        print("Engine ready.")

        # optional cache for ws by seed
        self.ws_cache: Dict[Tuple[int, float], torch.Tensor] = {}

    def get_ws(self, seed: int, truncation_psi: float) -> torch.Tensor:
        key = (int(seed), float(truncation_psi))
        if key in self.ws_cache:
            return self.ws_cache[key]

        rnd = np.random.RandomState(seed)
        z = torch.from_numpy(rnd.randn(1, self.G.z_dim).astype(np.float32)).to(self.device)
        ws = self.G.mapping(z, self.label, truncation_psi=truncation_psi)  # [1,num_ws,w_dim]
        self.ws_cache[key] = ws
        # keep cache small
        if len(self.ws_cache) > 32:
            self.ws_cache.pop(next(iter(self.ws_cache)))
        return ws

    def apply_knobs(self, ws: torch.Tensor, knobs: List[float]) -> torch.Tensor:
        if len(knobs) != self.n_knobs:
            raise ValueError(f"knobs length mismatch: got {len(knobs)}, expected {self.n_knobs}")
        out = ws
        for i in range(self.n_knobs):
            v = float(knobs[i])
            if self.use_sigma:
                v *= self.knob_sigma[i]
            out = out + (v * self.knob_dir_t[i] * self.knob_mask_t[i])
        return out

    def render(self, seed: int, knobs: List[float], truncation_psi: float, noise_mode: str) -> PIL.Image.Image:
        with torch.no_grad():
            ws = self.get_ws(seed=seed, truncation_psi=truncation_psi)
            ws_edit = self.apply_knobs(ws, knobs)
            img = synth_image(self.G, ws_edit, noise_mode=noise_mode)
            return img


# -------------------------
# CLI (server mode)
# -------------------------

@click.command()
@click.option("--network", "network_pkl", required=True, help="Network pickle path or URL")
@click.option("--pca", "pca_npz", required=True, help="PCA npz, e.g. directions/pca_w_xxx.npz")
@click.option("--trunc", "truncation_psi", type=float, default=1.0, show_default=True)
@click.option("--noise-mode", type=click.Choice(["const", "random", "none"]), default="const", show_default=True)
@click.option("--class", "class_idx", type=int, default=None, help="Class label index (for conditional nets)")
@click.option("--knob-comps", type=str, default="3", show_default=True,
              help="PCA comp index for each knob. Either 1 int (broadcast) or N ints, e.g. '3' or '3,3,1,7'")
@click.option("--knob-layers", type=str, required=True,
              help="N layer ranges separated by '|', e.g. '0|1-6|7-9|10-13|14|15'")
@click.option("--normalize-dir", is_flag=True, help="Normalize each PCA direction to unit norm before applying")
@click.option("--use-sigma", is_flag=True, help="Scale knob values by sigma[comp] if present")
@click.option("--ipc-dir", type=str, default="ipc", show_default=True, help="IPC directory containing request.json")
@click.option("--poll-ms", type=int, default=20, show_default=True, help="Polling interval in milliseconds")
def main(
    network_pkl: str,
    pca_npz: str,
    truncation_psi: float,
    noise_mode: str,
    class_idx: Optional[int],
    knob_comps: str,
    knob_layers: str,
    normalize_dir: bool,
    use_sigma: bool,
    ipc_dir: str,
    poll_ms: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    knob_layer_parts = [p.strip() for p in knob_layers.split("|") if p.strip() != ""]
    if len(knob_layer_parts) < 1:
        raise click.ClickException("--knob-layers must have at least 1 part, e.g. '0|1-6|7-9'")

    def parse_csv_ints(s: str) -> List[int]:
        return [int(x.strip()) for x in (s or "").split(",") if x.strip() != ""]

    knob_comp_list = parse_csv_ints(knob_comps)
    if len(knob_comp_list) < 1:
        raise click.ClickException("--knob-comps must have at least 1 int")

    os.makedirs(ipc_dir, exist_ok=True)
    req_path = os.path.join(ipc_dir, "request.json")
    out_png = os.path.join(ipc_dir, "latest.png")
    out_meta = os.path.join(ipc_dir, "latest_meta.json")

    engine = MorphEngine(
        network_pkl=network_pkl,
        pca_npz=pca_npz,
        knob_comps=knob_comp_list,
        knob_layers_parts=knob_layer_parts,
        truncation_psi=truncation_psi,
        noise_mode=noise_mode,
        class_idx=class_idx,
        normalize_dir=normalize_dir,
        use_sigma=use_sigma,
        device=device,
    )

    # create a default request.json if missing
    if not os.path.exists(req_path):
        default_req = {
            "seed": 0,
            "trunc": float(truncation_psi),
            "noise_mode": noise_mode,
            "knobs": [0.0] * engine.n_knobs,
            "frame_id": 0
        }
        atomic_write_text(json.dumps(default_req, ensure_ascii=False, indent=2), req_path)
        print("Created default request.json at:", req_path)

    print("Watching:", req_path)
    last_mtime = 0.0
    last_payload = None

    while True:
        try:
            st = os.stat(req_path)
            if st.st_mtime != last_mtime:
                last_mtime = st.st_mtime

                with open(req_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)

                # simple de-dup (some apps touch file without changes)
                if payload == last_payload:
                    time.sleep(poll_ms / 1000.0)
                    continue
                last_payload = payload

                seed = int(payload.get("seed", 0))
                trunc = float(payload.get("trunc", engine.trunc_default))
                nm = str(payload.get("noise_mode", engine.noise_mode))
                knobs = payload.get("knobs", None)
                frame_id = payload.get("frame_id", None)

                if knobs is None:
                    knobs = [0.0] * engine.n_knobs
                if not isinstance(knobs, list):
                    raise ValueError("knobs must be a list")

                # render
                img = engine.render(seed=seed, knobs=[float(x) for x in knobs], truncation_psi=trunc, noise_mode=nm)
                atomic_write_png(img, out_png)

                meta = {
                    "ok": True,
                    "time": time.time(),
                    "seed": seed,
                    "trunc": trunc,
                    "noise_mode": nm,
                    "knob_count": engine.n_knobs,
                    "knobs": [float(x) for x in knobs],
                    "frame_id": frame_id,
                    "out_png": "latest.png",
                }
                atomic_write_text(json.dumps(meta, ensure_ascii=False, indent=2), out_meta)

        except KeyboardInterrupt:
            print("Exiting.")
            break
        except Exception as e:
            err = {
                "ok": False,
                "time": time.time(),
                "error": repr(e),
            }
            try:
                atomic_write_text(json.dumps(err, ensure_ascii=False, indent=2), out_meta)
            except Exception:
                pass

        time.sleep(poll_ms / 1000.0)


if __name__ == "__main__":
    main()
