"""
app.py — Mechanical MNIST Demo App
====================================
Supports two ML models simultaneously:
  Model A: Matt's UNetMultiRegression (256x256, norm_stats_matt.npz)
  Model B: FNO MultiTaskFNO (64x64, norm_stats_fno.npy)

File structure expected:
    app.py
    model_matt.py        ← Matt's model architecture
    model_fno.py         ← FNO model architecture
    best_model_matt.pt
    best_model_fno.pt
    norm_stats_matt.npz
    norm_stats_fno.npy
    NumpyImageToGmsh.py
    Equibiaxial_Hyperelastic.py

Usage:
    streamlit run app.py
"""

import io
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import streamlit as st
import torch

REPO_DIR    = Path(__file__).parent.resolve()
EXAMPLE_DIR = REPO_DIR / "example"
DISP_VALS   = [0.0, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5]

# Model configs
MODELS = {
    "A": {
        "label":      "Model A — UNet (Matt)",
        "model_file": "model_matt",
        "class":      "UNetMultiRegression",
        "pt":         "best_model_matt.pt",
        "stats":      "norm_stats_matt.npz",
        "stats_fmt":  "npz",
        "img_size":   256,
        "ckpt_hparams": False,
    },
    "B": {
        "label":      "Model B — FNO",
        "model_file": "model_fno",
        "class":      "MultiTaskFNO",
        "pt":         "best_model_fno.pt",
        "stats":      "norm_stats_fno.npy",
        "stats_fmt":  "npy",
        "img_size":   64,
        "ckpt_hparams": True,  # FNO stores hparams in checkpoint
    },
}

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Mechanical MNIST — ML Demo",
    page_icon="⚙️",
    layout="wide",
)
st.title("⚙️ Mechanical MNIST — ML Inference & FEA Comparison")

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(model_key):
    cfg        = MODELS[model_key]
    model_mod  = __import__(cfg["model_file"])
    ModelClass = getattr(model_mod, cfg["class"])

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pt_path    = REPO_DIR / cfg["pt"]
    stats_path = REPO_DIR / cfg["stats"]

    if not pt_path.exists():
        return None, None, device, f"{cfg['pt']} not found"
    if not stats_path.exists():
        return None, None, device, f"{cfg['stats']} not found"

    ckpt = torch.load(pt_path, map_location=device, weights_only=False)

    if cfg["ckpt_hparams"]:
        hparams = ckpt.get("hparams", {"modes": 16, "width": 64, "n_layers": 4})
        model   = ModelClass(**hparams).to(device)
    else:
        model = ModelClass(in_channels=1).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if cfg["stats_fmt"] == "npz":
        stats = np.load(stats_path, allow_pickle=True)
    else:
        stats = np.load(stats_path, allow_pickle=True).item()

    return model, stats, device, None


# ─────────────────────────────────────────────
# CORE CONVERSION
# ─────────────────────────────────────────────
def to_binary_255(img_gray):
    """Any grayscale → binary uint8 0/255. Dark=fibre(E=10), Light=matrix(E=1)."""
    if img_gray.max() <= 1.0:
        return (img_gray * 255).astype(np.uint8)
    _, binary = cv2.threshold(img_gray.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    return binary


def binary_255_to_ml_input(binary_255, size):
    """0/255 binary → float32 (size x size), 0=fibre, 1=matrix."""
    resized = cv2.resize(binary_255, (size, size), interpolation=cv2.INTER_AREA)
    return (resized >= 128).astype(np.float32)


def load_dataset_txt(txt_path, size=400):
    img = np.loadtxt(txt_path)
    binary_255 = to_binary_255(img)
    if binary_255.shape != (size, size):
        binary_255 = cv2.resize(binary_255, (size, size), interpolation=cv2.INTER_NEAREST)
    return binary_255


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_rgb(img_rgb, min_radius=8, max_radius=14, min_dist=25, param2=12, size=400):
    gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized, (5, 5), 1)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1,
                               minDist=min_dist, param1=50, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)
    binary    = np.ones((size, size), dtype=np.uint8) * 255
    n_circles = 0
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        circles = [c for c in circles if
                   c[0]-c[2]>1 and c[0]+c[2]<size-1 and
                   c[1]-c[2]>1 and c[1]+c[2]<size-1]
        n_circles = len(circles)
        for (x, y, r) in circles:
            cv2.circle(binary, (x, y), r, 0, -1)
    return binary, n_circles


def preprocess_binary_upload(img_gray, size=400):
    binary_255 = to_binary_255(img_gray)
    return cv2.resize(binary_255, (size, size), interpolation=cv2.INTER_NEAREST)


# ─────────────────────────────────────────────
# ML INFERENCE
# ─────────────────────────────────────────────
def run_model(model_key, binary_255, model, stats, device):
    cfg      = MODELS[model_key]
    size     = cfg["img_size"]
    binary_ml = binary_255_to_ml_input(binary_255, size)

    inp = torch.from_numpy(binary_ml).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        if cfg["stats_fmt"] == "npz":
            # Matt: returns psi, force, disp
            se_pred, rf_pred, disp_pred = model(inp)
        else:
            # FNO/Shuan: returns disp, se, rf
            disp_pred, se_pred, rf_pred = model(inp)

    if cfg["stats_fmt"] == "npz":
        # Matt: npz with psi_mean, force_mean, disp_mean (scalar)
        psi_mean   = np.array(stats["psi_mean"])
        psi_std    = np.array(stats["psi_std"]) + 1e-10
        se         = se_pred[0].cpu().numpy() * psi_std + psi_mean
        f_mean     = np.array(stats["force_mean"])
        f_std      = np.array(stats["force_std"]) + 1e-10
        rf         = (rf_pred[0].cpu().numpy() * f_std + f_mean).reshape(7, 4)
        d_mean     = float(stats["disp_mean"])
        d_std      = float(stats["disp_std"]) + 1e-10
        disp_x     = disp_pred[0, 0].cpu().numpy() * d_std + d_mean
        disp_y     = disp_pred[0, 1].cpu().numpy() * d_std + d_mean
    else:
        # FNO/Shuan: npy with disp_x_mean, se_mean, rf_mean
        disp_x = disp_pred[0, 0].cpu().numpy() * stats["disp_x_std"] + stats["disp_x_mean"]
        disp_y = disp_pred[0, 1].cpu().numpy() * stats["disp_y_std"] + stats["disp_y_mean"]
        se_mean = np.array(stats["se_mean"])
        se_std  = np.array(stats["se_std"]) + 1e-10
        se      = se_pred[0].cpu().numpy() * se_std + se_mean
        rf_mean = np.array(stats["rf_mean"])
        rf_std  = np.array(stats["rf_std"]) + 1e-10
        rf      = (rf_pred[0].cpu().numpy() * rf_std + rf_mean).reshape(7, 4)

    return disp_x, disp_y, se, rf


# ─────────────────────────────────────────────
# FEA PIPELINE
# ─────────────────────────────────────────────
def save_binary_for_fea(binary_255, pattern_name):
    input_dir = REPO_DIR / "input_patterns"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir()
    np.savetxt(input_dir / f"{pattern_name}.txt", binary_255, fmt="%d")


def run_fea_pipeline(pattern_name, progress_bar, status_text):
    python      = sys.executable
    mesh_dir    = REPO_DIR / "mesh_files"
    results_dir = REPO_DIR / "Results"
    mesh_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    xdmf_path = mesh_dir / f"{pattern_name}.xdmf"

    status_text.text("Step 1/3 — Generating mesh geometry...")
    progress_bar.progress(10)
    r = subprocess.run([python, "NumpyImageToGmsh.py"], cwd=REPO_DIR,
                       capture_output=True, text=True)
    if r.returncode != 0:
        return False, f"NumpyImageToGmsh failed:\n{r.stderr}"

    mesh_script = mesh_dir / f"{pattern_name}.py"
    if not mesh_script.exists():
        return False, f"Mesh script not found: {mesh_script}"

    status_text.text("Step 2/3 — Running Gmsh meshing (~30s)...")
    progress_bar.progress(30)
    r = subprocess.run([python, str(mesh_script)], cwd=mesh_dir,
                       capture_output=True, text=True)
    if r.returncode != 0:
        return False, f"Mesh failed:\n{r.stderr}"

    if not xdmf_path.exists():
        return False, f"XDMF not generated: {xdmf_path}"

    status_text.text("Step 3/3 — Running FEniCS (~2-5 min)...")
    progress_bar.progress(50)
    r = subprocess.run([python, "Equibiaxial_Hyperelastic.py", str(xdmf_path)],
                       cwd=REPO_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        return False, f"FEniCS failed:\n{r.stderr}"

    progress_bar.progress(100)
    status_text.text("FEA complete!")
    return True, ""


def load_fea_results(pattern_name):
    d = REPO_DIR / "Results"
    def f(s): return d / f"{pattern_name}{s}"
    return (np.loadtxt(f("_pixel_disp_0.5_x.txt")),
            np.loadtxt(f("_pixel_disp_0.5_y.txt")),
            np.loadtxt(f("_strain_energy.txt")),
            np.loadtxt(f("_rxn_force.txt")))


# ─────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────
def _disp_row(gs, row, fig, disp_x, disp_y, label):
    mag = np.sqrt(disp_x**2 + disp_y**2)
    for col, (data, title, cmap) in enumerate([
        (disp_x, f"{label} $u_x$", "RdBu_r"),
        (disp_y, f"{label} $u_y$", "RdBu_r"),
        (mag,    f"{label} $|u|$", "viridis"),
    ], start=1):
        ax   = fig.add_subplot(gs[row, col])
        vmax = max(np.abs(data).max(), 1e-6) if cmap == "RdBu_r" else None
        im   = ax.imshow(data, cmap=cmap, origin="upper",
                         vmin=-vmax if vmax else None, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _scalar_axes(gs, row, fig, se, rf, label, color_se="b", color_rf=None):
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]
    ls = "-o" if color_se == "b" else "--s"

    ax = fig.add_subplot(gs[row, 0:2])
    ax.plot(DISP_VALS, se, ls, color=color_se, linewidth=2, markersize=5, label=label)
    ax.set_xlabel("d", fontsize=10)
    ax.set_ylabel("Strain energy", fontsize=10)
    ax.set_title("Strain energy vs displacement", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[row, 2:])
    rf_labels = ["F_left_x", "F_right_x", "F_bottom_y", "F_top_y"]
    for i, (lbl, c) in enumerate(zip(rf_labels, colors)):
        ax.plot(DISP_VALS, rf[:, i], ls, color=c, linewidth=2, markersize=5,
                label=f"{label} {lbl}")
    ax.set_xlabel("d", fontsize=10)
    ax.set_ylabel("Reaction force", fontsize=10)
    ax.set_title("Reaction forces vs displacement", fontsize=11)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)


def make_single_fig(binary_255, display_img, mode, disp_x, disp_y, se, rf, title):
    fig = plt.figure(figsize=(22, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    if mode == "rgb": ax.imshow(display_img)
    else: ax.imshow(display_img, cmap="gray")
    ax.set_title("Input pattern", fontsize=11)
    ax.axis("off")

    _disp_row(gs, 0, fig, disp_x, disp_y, "")
    _scalar_axes(gs, 1, fig, se, rf, title[:20])
    return fig


def make_ab_comparison(binary_255,
                       a_dx, a_dy, a_se, a_rf,
                       b_dx, b_dy, b_se, b_rf,
                       label_a, label_b):
    """Model A vs Model B comparison figure."""
    a_mag = np.sqrt(a_dx**2 + a_dy**2)
    b_mag = np.sqrt(b_dx**2 + b_dy**2)

    # Resize to same shape for error maps
    h, w = max(a_dx.shape[0], b_dx.shape[0]), max(a_dx.shape[1], b_dx.shape[1])
    a_dx_r = cv2.resize(a_dx, (w, h))
    a_dy_r = cv2.resize(a_dy, (w, h))
    b_dx_r = cv2.resize(b_dx, (w, h))
    b_dy_r = cv2.resize(b_dy, (w, h))
    a_mag_r = np.sqrt(a_dx_r**2 + a_dy_r**2)
    b_mag_r = np.sqrt(b_dx_r**2 + b_dy_r**2)

    fig = plt.figure(figsize=(26, 20))
    fig.suptitle(f"{label_a} vs {label_b}", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(4, 6, figure=fig, hspace=0.5, wspace=0.4)

    # Row 0: input
    ax = fig.add_subplot(gs[0, 0:2])
    ax.imshow(binary_255, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Binary pattern", fontsize=10)
    ax.axis("off")

    # Row 1: displacement fields side by side
    for col, (title, data, cmap) in enumerate([
        (f"{label_a} $u_x$", a_dx_r, "RdBu_r"),
        (f"{label_b} $u_x$", b_dx_r, "RdBu_r"),
        (f"{label_a} $u_y$", a_dy_r, "RdBu_r"),
        (f"{label_b} $u_y$", b_dy_r, "RdBu_r"),
        (f"{label_a} $|u|$", a_mag_r, "viridis"),
        (f"{label_b} $|u|$", b_mag_r, "viridis"),
    ]):
        ax = fig.add_subplot(gs[1, col])
        if cmap == "RdBu_r":
            vmax = max(np.abs(data).max(), 1e-6)
            im   = ax.imshow(data, cmap=cmap, origin="upper", vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(data, cmap=cmap, origin="upper")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 2: error maps A vs B
    err_x   = np.abs(a_dx_r - b_dx_r)
    err_y   = np.abs(a_dy_r - b_dy_r)
    err_mag = np.abs(a_mag_r - b_mag_r)
    ref_x   = np.abs(b_dx_r).mean() + 1e-10
    ref_y   = np.abs(b_dy_r).mean() + 1e-10
    ref_mag = b_mag_r.mean() + 1e-10

    for col, (title, data) in enumerate([
        ("|Diff| $u_x$",        err_x),
        ("Rel. diff $u_x$ (%)", err_x / ref_x * 100),
        ("|Diff| $u_y$",        err_y),
        ("Rel. diff $u_y$ (%)", err_y / ref_y * 100),
        ("|Diff| $|u|$",        err_mag),
        ("Rel. diff $|u|$ (%)", err_mag / ref_mag * 100),
    ]):
        ax = fig.add_subplot(gs[2, col])
        im = ax.imshow(data, cmap="hot_r", origin="upper", vmin=0)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 3: overlaid scalar curves
    ax = fig.add_subplot(gs[3, 0:2])
    ax.plot(DISP_VALS, a_se, "b-o",  linewidth=2, markersize=6, label=label_a)
    ax.plot(DISP_VALS, b_se, "r--s", linewidth=2, markersize=6, label=label_b)
    ax.set_xlabel("d", fontsize=10)
    ax.set_ylabel("Strain energy", fontsize=10)
    ax.set_title("Strain energy: A vs B", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[3, 2:4])
    rf_labels = ["F_left_x", "F_right_x", "F_bottom_y", "F_top_y"]
    colors    = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]
    for i, (lbl, c) in enumerate(zip(rf_labels, colors)):
        ax.plot(DISP_VALS, a_rf[:, i], "-o",  color=c, linewidth=2, markersize=5,
                label=f"A {lbl}", alpha=0.9)
        ax.plot(DISP_VALS, b_rf[:, i], "--s", color=c, linewidth=2, markersize=5,
                label=f"B {lbl}", alpha=0.6)
    ax.set_xlabel("d", fontsize=10)
    ax.set_ylabel("Reaction force", fontsize=10)
    ax.set_title("Reaction forces: A vs B", fontsize=11)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)

    # Error table
    ax = fig.add_subplot(gs[3, 4:])
    ax.axis("off")
    se_diff = np.abs(a_se - b_se)
    rf_diff = np.abs(a_rf - b_rf)
    rows = [
        ["Metric",                   "|Mean diff|",                     "Rel. diff"],
        ["Strain energy",            f"{se_diff.mean():.4f}",           f"{(se_diff/(np.abs(b_se)+1e-10)*100).mean():.1f}%"],
        ["Reaction forces",          f"{rf_diff.mean():.4f}",           f"{(rf_diff/(np.abs(b_rf)+1e-10)*100).mean():.1f}%"],
        ["Displacement u_x (field)", f"{err_x.mean():.4f}",             f"{(err_x/ref_x*100).mean():.1f}%"],
        ["Displacement u_y (field)", f"{err_y.mean():.4f}",             f"{(err_y/ref_y*100).mean():.1f}%"],
        ["Displacement |u| (field)", f"{err_mag.mean():.4f}",           f"{(err_mag/ref_mag*100).mean():.1f}%"],
    ]
    table = ax.table(cellText=rows[1:], colLabels=rows[0],
                     cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#ecf0f1")
    ax.set_title(f"Difference summary ({label_a} vs {label_b})", fontsize=11, pad=20)
    return fig


def make_vs_fea_fig(binary_255, ml_dx, ml_dy, ml_se, ml_rf,
                    fea_dx, fea_dy, fea_se, fea_rf, ml_label):
    """ML vs FEA comparison."""
    fea_dx_r  = cv2.resize(fea_dx, (ml_dx.shape[1], ml_dx.shape[0]))
    fea_dy_r  = cv2.resize(fea_dy, (ml_dy.shape[1], ml_dy.shape[0]))
    ml_mag    = np.sqrt(ml_dx**2  + ml_dy**2)
    fea_mag   = np.sqrt(fea_dx_r**2 + fea_dy_r**2)

    fig = plt.figure(figsize=(26, 20))
    fig.suptitle(f"{ml_label} vs FEA Ground Truth", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(4, 6, figure=fig, hspace=0.5, wspace=0.4)

    ax = fig.add_subplot(gs[0, 0:2])
    ax.imshow(binary_255, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Binary pattern", fontsize=10)
    ax.axis("off")

    for col, (title, data, cmap) in enumerate([
        (f"{ml_label} $u_x$", ml_dx,    "RdBu_r"),
        ("FEA $u_x$",         fea_dx_r, "RdBu_r"),
        (f"{ml_label} $u_y$", ml_dy,    "RdBu_r"),
        ("FEA $u_y$",         fea_dy_r, "RdBu_r"),
        (f"{ml_label} $|u|$", ml_mag,   "viridis"),
        ("FEA $|u|$",         fea_mag,  "viridis"),
    ]):
        ax = fig.add_subplot(gs[1, col])
        if cmap == "RdBu_r":
            vmax = max(np.abs(data).max(), 1e-6)
            im   = ax.imshow(data, cmap=cmap, origin="upper", vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(data, cmap=cmap, origin="upper")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    err_x   = np.abs(ml_dx   - fea_dx_r)
    err_y   = np.abs(ml_dy   - fea_dy_r)
    err_mag = np.abs(ml_mag  - fea_mag)
    for col, (title, data) in enumerate([
        ("|Error| $u_x$",        err_x),
        ("Rel. error $u_x$ (%)", err_x / (np.abs(fea_dx_r).mean()+1e-10) * 100),
        ("|Error| $u_y$",        err_y),
        ("Rel. error $u_y$ (%)", err_y / (np.abs(fea_dy_r).mean()+1e-10) * 100),
        ("|Error| $|u|$",        err_mag),
        ("Rel. error $|u|$ (%)", err_mag / (fea_mag.mean()+1e-10) * 100),
    ]):
        ax = fig.add_subplot(gs[2, col])
        im = ax.imshow(data, cmap="hot_r", origin="upper", vmin=0)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[3, 0:2])
    ax.plot(DISP_VALS, ml_se,  "b-o",  linewidth=2, markersize=6, label=ml_label)
    ax.plot(DISP_VALS, fea_se, "r--s", linewidth=2, markersize=6, label="FEA")
    ax.set_xlabel("d", fontsize=10); ax.set_ylabel("Strain energy", fontsize=10)
    ax.set_title(f"Strain energy: {ml_label} vs FEA", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[3, 2:4])
    rf_labels = ["F_left_x", "F_right_x", "F_bottom_y", "F_top_y"]
    colors    = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]
    for i, (lbl, c) in enumerate(zip(rf_labels, colors)):
        ax.plot(DISP_VALS, ml_rf[:, i],  "-o",  color=c, linewidth=2, markersize=5,
                label=f"{ml_label} {lbl}", alpha=0.9)
        ax.plot(DISP_VALS, fea_rf[:, i], "--s", color=c, linewidth=2, markersize=5,
                label=f"FEA {lbl}", alpha=0.6)
    ax.set_xlabel("d", fontsize=10); ax.set_ylabel("Reaction force", fontsize=10)
    ax.set_title(f"Reaction forces: {ml_label} vs FEA", fontsize=11)
    ax.legend(fontsize=7, loc="upper left", ncol=2); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[3, 4:])
    ax.axis("off")
    se_rel = np.abs(ml_se-fea_se)/(np.abs(fea_se)+1e-10)*100
    rf_rel = np.abs(ml_rf-fea_rf)/(np.abs(fea_rf)+1e-10)*100
    rows = [
        ["Metric",                   "Mean |error|",                        "Mean rel. error"],
        ["Strain energy",            f"{np.abs(ml_se-fea_se).mean():.4f}",  f"{se_rel.mean():.1f}%"],
        ["Reaction forces",          f"{np.abs(ml_rf-fea_rf).mean():.4f}",  f"{rf_rel.mean():.1f}%"],
        ["Displacement u_x (field)", f"{err_x.mean():.4f}",                 f"{(err_x/(np.abs(fea_dx_r).mean()+1e-10)*100).mean():.1f}%"],
        ["Displacement u_y (field)", f"{err_y.mean():.4f}",                 f"{(err_y/(np.abs(fea_dy_r).mean()+1e-10)*100).mean():.1f}%"],
        ["Displacement |u| (field)", f"{err_mag.mean():.4f}",               f"{(err_mag/(fea_mag.mean()+1e-10)*100).mean():.1f}%"],
    ]
    table = ax.table(cellText=rows[1:], colLabels=rows[0],
                     cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False); table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#ecf0f1")
    ax.set_title(f"Error summary ({ml_label} vs FEA)", fontsize=11, pad=20)
    return fig


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def arr_to_bytes(arr, fmt="%.5e"):
    buf = io.BytesIO()
    np.savetxt(buf, arr, fmt=fmt)
    return buf.getvalue()

def fig_to_bytes(f):
    buf = io.BytesIO()
    f.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def get_example_images():
    pattern_dir = EXAMPLE_DIR / "input_patterns"
    results_dir = REPO_DIR / "Results"
    if not pattern_dir.exists(): return []
    available = []
    for txt_file in sorted(pattern_dir.glob("*.txt")):
        name    = txt_file.stem
        has_fea = (results_dir / f"{name}_strain_energy.txt").exists()
        available.append((name, has_fea))
    return available


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("🗂️ Mode")
    app_mode = st.radio("Select mode",
                        ["📤 Upload your image", "🗄️ Dataset browser"], index=0)

    st.divider()
    st.header("🤖 Models")
    use_a = st.checkbox(f"Model A — UNet (Matt)", value=True)
    use_b = st.checkbox(f"Model B — FNO", value=False)

    if not use_a and not use_b:
        st.warning("Select at least one model.")

    if app_mode == "📤 Upload your image":
        st.divider()
        st.header("🖼️ Input type")
        input_mode = st.radio("Select input type",
                              ["RGB image (auto preprocessing)", "Binary image (use as-is)"],
                              index=0)
        if input_mode == "RGB image (auto preprocessing)":
            st.divider()
            st.header("⚙️ Preprocessing")
            param2     = st.slider("param2 (sensitivity)", 5, 40, 12)
            min_dist   = st.slider("min_dist (px)", 10, 50, 25)
            min_radius = st.slider("min_radius (px)", 3, 20, 8)
            max_radius = st.slider("max_radius (px)", 10, 40, 14)
        else:
            param2 = min_dist = min_radius = max_radius = None
            st.info("Binary: Black=fibre (E=10), White=matrix (E=1).")


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
model_a = model_b = stats_a = stats_b = device_a = device_b = None

if use_a:
    with st.spinner("Loading Model A (UNet)..."):
        model_a, stats_a, device_a, err_a = load_model("A")
    if err_a:
        st.error(f"Model A: {err_a}")
        use_a = False
    else:
        st.success(f"Model A loaded ({'GPU' if device_a.type == 'cuda' else 'CPU'})")

if use_b:
    with st.spinner("Loading Model B (FNO)..."):
        model_b, stats_b, device_b, err_b = load_model("B")
    if err_b:
        st.error(f"Model B: {err_b}")
        use_b = False
    else:
        st.success(f"Model B loaded ({'GPU' if device_b.type == 'cuda' else 'CPU'})")

if not use_a and not use_b:
    st.stop()


# ═════════════════════════════════════════════
# SHARED INFERENCE + DISPLAY LOGIC
# ═════════════════════════════════════════════
def run_and_display(binary_255, display_img, mode, pattern_name):
    results = {}

    if use_a:
        with st.spinner("Running Model A..."):
            results["A"] = run_model("A", binary_255, model_a, stats_a, device_a)
        st.success("✅ Model A complete!")

    if use_b:
        with st.spinner("Running Model B..."):
            results["B"] = run_model("B", binary_255, model_b, stats_b, device_b)
        st.success("✅ Model B complete!")

    # FEA button
    st.divider()
    st.subheader("🔬 FEA Ground Truth (optional — 2-5 min)")

    for key in ["fea_results", "fea_pattern"]:
        if key not in st.session_state:
            st.session_state[key] = None

    if st.button("▶ Run FEA", type="primary"):
        save_binary_for_fea(binary_255, pattern_name)
        pb  = st.progress(0)
        stx = st.empty()
        ok, err = run_fea_pipeline(pattern_name, pb, stx)
        if ok:
            st.success("✅ FEA complete!")
            st.session_state.fea_results = load_fea_results(pattern_name)
            st.session_state.fea_pattern = pattern_name
        else:
            st.error(f"FEA failed:\n{err}")

    fea_done = (st.session_state.fea_results is not None and
                st.session_state.fea_pattern == pattern_name)

    st.divider()

    # Build tabs
    tab_names = []
    if use_a: tab_names.append(f"📊 {MODELS['A']['label']}")
    if use_b: tab_names.append(f"📊 {MODELS['B']['label']}")
    if use_a and use_b: tab_names.append("⚖️ A vs B")
    if fea_done:
        if use_a: tab_names.append(f"⚖️ A vs FEA")
        if use_b: tab_names.append(f"⚖️ B vs FEA")
        tab_names.append("🔬 FEA Ground Truth")

    tabs = st.tabs(tab_names)
    tab_idx = 0

    figs = {}

    if use_a:
        a_dx, a_dy, a_se, a_rf = results["A"]
        fig_a = make_single_fig(binary_255, display_img, mode, a_dx, a_dy, a_se, a_rf,
                                MODELS["A"]["label"])
        figs["A"] = fig_a
        with tabs[tab_idx]: st.pyplot(fig_a, use_container_width=True)
        tab_idx += 1

    if use_b:
        b_dx, b_dy, b_se, b_rf = results["B"]
        fig_b = make_single_fig(binary_255, display_img, mode, b_dx, b_dy, b_se, b_rf,
                                MODELS["B"]["label"])
        figs["B"] = fig_b
        with tabs[tab_idx]: st.pyplot(fig_b, use_container_width=True)
        tab_idx += 1

    if use_a and use_b:
        fig_ab = make_ab_comparison(binary_255,
                                    a_dx, a_dy, a_se, a_rf,
                                    b_dx, b_dy, b_se, b_rf,
                                    MODELS["A"]["label"], MODELS["B"]["label"])
        figs["AB"] = fig_ab
        with tabs[tab_idx]: st.pyplot(fig_ab, use_container_width=True)
        tab_idx += 1

    if fea_done:
        fea_dx, fea_dy, fea_se, fea_rf = st.session_state.fea_results
        if use_a:
            fig_a_fea = make_vs_fea_fig(binary_255, a_dx, a_dy, a_se, a_rf,
                                        fea_dx, fea_dy, fea_se, fea_rf, MODELS["A"]["label"])
            figs["A_FEA"] = fig_a_fea
            with tabs[tab_idx]: st.pyplot(fig_a_fea, use_container_width=True)
            tab_idx += 1
        if use_b:
            fig_b_fea = make_vs_fea_fig(binary_255, b_dx, b_dy, b_se, b_rf,
                                        fea_dx, fea_dy, fea_se, fea_rf, MODELS["B"]["label"])
            figs["B_FEA"] = fig_b_fea
            with tabs[tab_idx]: st.pyplot(fig_b_fea, use_container_width=True)
            tab_idx += 1

        # FEA standalone tab
        fig_fea = make_single_fig(binary_255, binary_255, "gray",
                                  fea_dx, fea_dy, fea_se, fea_rf, "FEA Ground Truth")
        figs["FEA"] = fig_fea
        with tabs[tab_idx]: st.pyplot(fig_fea, use_container_width=True)
        tab_idx += 1

    # Downloads
    st.divider()
    st.subheader("📥 Download results")
    dl_cols = st.columns(len(figs))
    for i, (key, fig) in enumerate(figs.items()):
        with dl_cols[i]:
            st.download_button(f"{key} plot (.png)", fig_to_bytes(fig),
                               file_name=f"{key.lower()}_results.png", mime="image/png")


# ═════════════════════════════════════════════
# MODE 1: UPLOAD
# ═════════════════════════════════════════════
if app_mode == "📤 Upload your image":
    is_rgb = input_mode == "RGB image (auto preprocessing)"

    uploaded = st.file_uploader(
        "Upload an RGB image" if is_rgb else "Upload a binary B&W image",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
    )

    if uploaded is not None:
        file_bytes   = np.frombuffer(uploaded.read(), np.uint8)
        pattern_name = Path(uploaded.name).stem.replace(" ", "_")

        if is_rgb:
            img_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            original = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(original, caption="Uploaded image", use_container_width=True,
                     clamp=not is_rgb)
        with col2:
            if is_rgb:
                binary_255, n_circles = preprocess_rgb(
                    original, min_radius=min_radius, max_radius=max_radius,
                    min_dist=min_dist, param2=param2)
                st.metric("Circles detected", n_circles)
            else:
                binary_255 = preprocess_binary_upload(original)
            st.metric("Fibre volume fraction",
                      f"{100*np.sum(binary_255==0)/binary_255.size:.1f}%")
            st.image(binary_255, caption="Binary (400×400)", use_container_width=True, clamp=True)

        st.divider()
        mode_str = "rgb" if is_rgb else "gray"
        run_and_display(binary_255, original, mode_str, pattern_name)

    else:
        st.info("👆 Upload an image to get started.")


# ═════════════════════════════════════════════
# MODE 2: DATASET BROWSER
# ═════════════════════════════════════════════
else:
    st.subheader("🗄️ Dataset Browser — Example Images")
    examples = get_example_images()
    if not examples:
        st.warning(f"No example images found.")
        st.stop()

    n_cols     = min(len(examples), 6)
    thumb_cols = st.columns(n_cols)
    if "browser_selected" not in st.session_state:
        st.session_state.browser_selected = examples[0][0]

    for i, (name, has_fea) in enumerate(examples):
        txt_path   = EXAMPLE_DIR / "input_patterns" / f"{name}.txt"
        binary_255 = load_dataset_txt(txt_path)
        thumb      = cv2.resize(binary_255, (128, 128), interpolation=cv2.INTER_NEAREST)
        with thumb_cols[i % n_cols]:
            st.image(thumb, caption=name, use_container_width=True)
            st.caption("✅ FEA ready" if has_fea else "⏳ No FEA yet")
            if st.button("Select", key=f"btn_{name}"):
                st.session_state.browser_selected = name

    st.divider()
    selected   = st.session_state.browser_selected
    st.markdown(f"### Selected: `{selected}`")

    txt_path   = EXAMPLE_DIR / "input_patterns" / f"{selected}.txt"
    binary_255 = load_dataset_txt(txt_path)

    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(binary_255, caption=f"{selected}", use_container_width=True)
        st.metric("Fibre volume fraction",
                  f"{100*np.sum(binary_255==0)/binary_255.size:.1f}%")

    with col2:
        # Check precomputed FEA
        results_dir = REPO_DIR / "Results"
        if (results_dir / f"{selected}_strain_energy.txt").exists():
            st.session_state.fea_results = load_fea_results(selected)
            st.session_state.fea_pattern = selected
            st.success("✅ Precomputed FEA results loaded!")
        else:
            st.session_state.fea_results = None
            st.session_state.fea_pattern = None

    run_and_display(binary_255, binary_255, "gray", selected)
