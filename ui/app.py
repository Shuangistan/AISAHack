"""
app.py — Mechanical MNIST Demo App (Matt's model)
===================================================
Two main modes:
  1. Upload your own image → ML inference → optional FEA → comparison
  2. Dataset browser → pick example image → ML + precomputed FEA reference

Convention:
  - Dataset images (.txt): values 0/1   (0=fibre E=10, 1=matrix E=1)
  - Custom images (upload): values 0/255 (0=fibre E=10, 255=matrix E=1)
  - FEA input (.txt): MUST be 0/255 for NumpyImageToGmsh.py
  - ML input: values 0/1 float32 (0=fibre, 1=matrix)

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
IMG_SIZE    = 256

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
def load_model():
    try:
        from model import UNetMultiRegression
    except ImportError:
        st.error("model.py not found.")
        st.stop()
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = REPO_DIR / "best_model.pt"
    stats_path = REPO_DIR / "norm_stats.npz"
    if not model_path.exists():
        st.error(f"best_model.pt not found in {REPO_DIR}")
        st.stop()
    if not stats_path.exists():
        st.error(f"norm_stats.npz not found in {REPO_DIR}")
        st.stop()
    model = UNetMultiRegression(in_channels=1).to(device)
    ckpt  = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    stats = np.load(stats_path, allow_pickle=True)
    return model, stats, device


# ─────────────────────────────────────────────
# CORE CONVERSION FUNCTIONS
# ─────────────────────────────────────────────
def to_binary_255(img_gray):
    """Convert any grayscale image to binary uint8 with values 0/255.
    Handles: 0/1 float, 0/255 uint8, anything in between.
    Convention: dark pixels → 0 (fibre E=10), light pixels → 255 (matrix E=1).
    """
    if img_gray.max() <= 1.0:
        # Dataset convention: 0/1 → multiply by 255
        return (img_gray * 255).astype(np.uint8)
    else:
        # Already 0/255 range
        _, binary = cv2.threshold(
            img_gray.astype(np.uint8), 127, 255, cv2.THRESH_BINARY
        )
        return binary


def binary_255_to_ml_input(binary_255, size=IMG_SIZE):
    """Convert 0/255 binary image to ML model input float32 (0/1, size x size).
    Convention: 0=fibre (rigid E=10), 1=matrix (soft E=1).
    """
    resized = cv2.resize(binary_255, (size, size), interpolation=cv2.INTER_AREA)
    # After resize some pixels may be intermediate → re-binarize
    binary_float = (resized >= 128).astype(np.float32)  # 0=fibre, 1=matrix
    return binary_float


def load_dataset_txt(txt_path, size=400):
    """Load a dataset .txt pattern and return binary 0/255 uint8 image."""
    img = np.loadtxt(txt_path)
    binary_255 = to_binary_255(img)
    if binary_255.shape != (size, size):
        binary_255 = cv2.resize(binary_255, (size, size), interpolation=cv2.INTER_NEAREST)
    return binary_255


# ─────────────────────────────────────────────
# PREPROCESSING — Upload modes
# ─────────────────────────────────────────────
def preprocess_rgb(img_rgb, min_radius=8, max_radius=14, min_dist=25, param2=12, size=400):
    """RGB image → HoughCircles → binary 0/255."""
    gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized, (5, 5), 1)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1,
        minDist=min_dist, param1=50, param2=param2,
        minRadius=min_radius, maxRadius=max_radius,
    )
    binary    = np.ones((size, size), dtype=np.uint8) * 255
    n_circles = 0
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        circles = [c for c in circles if
                   c[0] - c[2] > 1 and c[0] + c[2] < size - 1 and
                   c[1] - c[2] > 1 and c[1] + c[2] < size - 1]
        n_circles = len(circles)
        for (x, y, r) in circles:
            cv2.circle(binary, (x, y), r, 0, -1)
    binary_ml = binary_255_to_ml_input(binary)
    return binary, binary_ml, n_circles


def preprocess_binary_upload(img_gray, size=400):
    """Binary uploaded image → use as-is, just convert to 0/255 and resize."""
    binary_255 = to_binary_255(img_gray)
    binary_255 = cv2.resize(binary_255, (size, size), interpolation=cv2.INTER_NEAREST)
    binary_ml  = binary_255_to_ml_input(binary_255)
    return binary_255, binary_ml, -1


# ─────────────────────────────────────────────
# ML INFERENCE
# ─────────────────────────────────────────────
def run_ml_model(binary_ml, model, stats, device):
    """binary_ml: float32 array (IMG_SIZE, IMG_SIZE), values 0/1."""
    inp = torch.from_numpy(binary_ml).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        psi_pred, force_pred, disp_pred = model(inp)
    psi_mean      = np.array(stats["psi_mean"])
    psi_std       = np.array(stats["psi_std"]) + 1e-10
    strain_energy = psi_pred[0].cpu().numpy() * psi_std + psi_mean
    force_mean    = np.array(stats["force_mean"])
    force_std     = np.array(stats["force_std"]) + 1e-10
    rxn_force     = (force_pred[0].cpu().numpy() * force_std + force_mean).reshape(7, 4)
    disp_mean     = float(stats["disp_mean"])
    disp_std      = float(stats["disp_std"]) + 1e-10
    disp_x        = disp_pred[0, 0].cpu().numpy() * disp_std + disp_mean
    disp_y        = disp_pred[0, 1].cpu().numpy() * disp_std + disp_mean
    return disp_x, disp_y, strain_energy, rxn_force


# ─────────────────────────────────────────────
# FEA PIPELINE
# ─────────────────────────────────────────────
def save_binary_for_fea(binary_255, pattern_name):
    """Save binary_255 (0/255) as .txt for NumpyImageToGmsh.py.
    Clears input_patterns/ first so only this image is processed."""
    input_dir = REPO_DIR / "input_patterns"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir()
    # NumpyImageToGmsh expects 0/255 values
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
        return False, f"Mesh generation failed:\n{r.stderr}"

    if not xdmf_path.exists():
        return False, f"XDMF not generated: {xdmf_path}"

    status_text.text("Step 3/3 — Running FEniCS simulation (~2-5 min)...")
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
    return (
        np.loadtxt(f("_pixel_disp_0.5_x.txt")),
        np.loadtxt(f("_pixel_disp_0.5_y.txt")),
        np.loadtxt(f("_strain_energy.txt")),
        np.loadtxt(f("_rxn_force.txt")),
    )


# ─────────────────────────────────────────────
# DATASET BROWSER HELPERS
# ─────────────────────────────────────────────
def get_example_images():
    pattern_dir = EXAMPLE_DIR / "input_patterns"
    results_dir = REPO_DIR / "Results"
    if not pattern_dir.exists():
        return []
    available = []
    for txt_file in sorted(pattern_dir.glob("*.txt")):
        name    = txt_file.stem
        has_fea = (
            (results_dir / f"{name}_strain_energy.txt").exists() and
            (results_dir / f"{name}_rxn_force.txt").exists() and
            (results_dir / f"{name}_pixel_disp_0.5_x.txt").exists()
        )
        available.append((name, has_fea))
    return available


# ─────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────
def make_ml_figure(display_img, binary_255, disp_x, disp_y, strain_energy, rxn_force, mode):
    magnitude = np.sqrt(disp_x**2 + disp_y**2)
    fig = plt.figure(figsize=(22, 10))
    fig.suptitle("ML Prediction — Mechanical Response", fontsize=15, fontweight="bold", y=1.01)
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    if mode == "rgb":
        ax.imshow(display_img)
    else:
        ax.imshow(display_img, cmap="gray")
    ax.set_title("Input pattern", fontsize=11)
    ax.axis("off")

    for col, (data, title, cmap) in enumerate([
        (disp_x,    "Predicted $u_x$ (d=0.5)",    "RdBu_r"),
        (disp_y,    "Predicted $u_y$ (d=0.5)",    "RdBu_r"),
        (magnitude, "Displacement $|u|$ (d=0.5)", "viridis"),
    ], start=1):
        ax   = fig.add_subplot(gs[0, col])
        vmax = max(np.abs(data).max(), 1e-6) if cmap == "RdBu_r" else None
        im   = ax.imshow(data, cmap=cmap, origin="upper",
                         vmin=-vmax if vmax else None, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[1, 0:2])
    ax.plot(DISP_VALS, strain_energy, "b-o", linewidth=2, markersize=6)
    for x, y in zip(DISP_VALS[1:], strain_energy[1:]):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Applied displacement d", fontsize=11)
    ax.set_ylabel("Strain energy (ΔΨ)", fontsize=11)
    ax.set_title("Strain energy vs displacement", fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2:])
    for i, (label, color) in enumerate(zip(
        ["F_left_x", "F_right_x", "F_bottom_y", "F_top_y"],
        ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]
    )):
        ax.plot(DISP_VALS, rxn_force[:, i], "o-", label=label,
                color=color, linewidth=2, markersize=5)
    ax.set_xlabel("Applied displacement d", fontsize=11)
    ax.set_ylabel("Reaction force", fontsize=11)
    ax.set_title("Reaction forces vs displacement", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    return fig


def make_fea_figure(binary_255, fea_disp_x, fea_disp_y, fea_se, fea_rf):
    magnitude = np.sqrt(fea_disp_x**2 + fea_disp_y**2)
    fig = plt.figure(figsize=(22, 10))
    fig.suptitle("FEA Ground Truth — Mechanical Response", fontsize=15, fontweight="bold", y=1.01)
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(binary_255, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Binary pattern", fontsize=11)
    ax.axis("off")

    for col, (data, title, cmap) in enumerate([
        (fea_disp_x, "FEA $u_x$ (d=0.5)",         "RdBu_r"),
        (fea_disp_y, "FEA $u_y$ (d=0.5)",         "RdBu_r"),
        (magnitude,  "Displacement $|u|$ (d=0.5)", "viridis"),
    ], start=1):
        ax   = fig.add_subplot(gs[0, col])
        vmax = max(np.abs(data).max(), 1e-6) if cmap == "RdBu_r" else None
        im   = ax.imshow(data, cmap=cmap, origin="upper",
                         vmin=-vmax if vmax else None, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[1, 0:2])
    ax.plot(DISP_VALS, fea_se, "r-s", linewidth=2, markersize=6)
    for x, y in zip(DISP_VALS[1:], fea_se[1:]):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Applied displacement d", fontsize=11)
    ax.set_ylabel("Strain energy (ΔΨ)", fontsize=11)
    ax.set_title("Strain energy vs displacement", fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2:])
    for i, (label, color) in enumerate(zip(
        ["F_left_x", "F_right_x", "F_bottom_y", "F_top_y"],
        ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]
    )):
        ax.plot(DISP_VALS, fea_rf[:, i], "s-", label=label,
                color=color, linewidth=2, markersize=5)
    ax.set_xlabel("Applied displacement d", fontsize=11)
    ax.set_ylabel("Reaction force", fontsize=11)
    ax.set_title("Reaction forces vs displacement", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    return fig


def make_comparison_figure(binary_255, ml_disp_x, ml_disp_y, ml_se, ml_rf,
                            fea_disp_x, fea_disp_y, fea_se, fea_rf):
    fea_dx  = cv2.resize(fea_disp_x, (ml_disp_x.shape[1], ml_disp_x.shape[0]))
    fea_dy  = cv2.resize(fea_disp_y, (ml_disp_y.shape[1], ml_disp_y.shape[0]))
    ml_mag  = np.sqrt(ml_disp_x**2 + ml_disp_y**2)
    fea_mag = np.sqrt(fea_dx**2    + fea_dy**2)

    fig = plt.figure(figsize=(26, 22))
    fig.suptitle("ML vs FEA — Ground Truth Comparison", fontsize=16, fontweight="bold")
    gs  = gridspec.GridSpec(4, 6, figure=fig, hspace=0.5, wspace=0.4)

    ax = fig.add_subplot(gs[0, 0:2])
    ax.imshow(binary_255, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Binary pattern", fontsize=10)
    ax.axis("off")

    for col, (title, data, cmap) in enumerate([
        ("ML $u_x$",  ml_disp_x, "RdBu_r"),
        ("FEA $u_x$", fea_dx,    "RdBu_r"),
        ("ML $u_y$",  ml_disp_y, "RdBu_r"),
        ("FEA $u_y$", fea_dy,    "RdBu_r"),
        ("ML $|u|$",  ml_mag,    "viridis"),
        ("FEA $|u|$", fea_mag,   "viridis"),
    ]):
        ax = fig.add_subplot(gs[1, col])
        if cmap == "RdBu_r":
            vmax = max(np.abs(data).max(), 1e-6)
            im   = ax.imshow(data, cmap=cmap, origin="upper", vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(data, cmap=cmap, origin="upper")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    err_x   = np.abs(ml_disp_x - fea_dx)
    err_y   = np.abs(ml_disp_y - fea_dy)
    err_mag = np.abs(ml_mag - fea_mag)
    rel_x   = err_x   / (np.abs(fea_dx).mean()  + 1e-10) * 100
    rel_y   = err_y   / (np.abs(fea_dy).mean()  + 1e-10) * 100
    rel_mag = err_mag / (fea_mag.mean()          + 1e-10) * 100

    for col, (title, data) in enumerate([
        ("|Error| $u_x$",        err_x),
        ("Rel. error $u_x$ (%)", rel_x),
        ("|Error| $u_y$",        err_y),
        ("Rel. error $u_y$ (%)", rel_y),
        ("|Error| $|u|$",        err_mag),
        ("Rel. error $|u|$ (%)", rel_mag),
    ]):
        ax = fig.add_subplot(gs[2, col])
        im = ax.imshow(data, cmap="hot_r", origin="upper", vmin=0)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[3, 0:2])
    ax.plot(DISP_VALS, ml_se,  "b-o",  linewidth=2, markersize=6, label="ML prediction")
    ax.plot(DISP_VALS, fea_se, "r--s", linewidth=2, markersize=6, label="FEA ground truth")
    ax.set_xlabel("Applied displacement d", fontsize=10)
    ax.set_ylabel("Strain energy (ΔΨ)", fontsize=10)
    ax.set_title("Strain energy: ML vs FEA", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[3, 2:4])
    labels = ["F_left_x", "F_right_x", "F_bottom_y", "F_top_y"]
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.plot(DISP_VALS, ml_rf[:, i],  "-o",  color=color, linewidth=2,
                markersize=5, label=f"ML {label}", alpha=0.9)
        ax.plot(DISP_VALS, fea_rf[:, i], "--s", color=color, linewidth=2,
                markersize=5, label=f"FEA {label}", alpha=0.6)
    ax.set_xlabel("Applied displacement d", fontsize=10)
    ax.set_ylabel("Reaction force", fontsize=10)
    ax.set_title("Reaction forces: ML vs FEA", fontsize=11)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[3, 4:])
    ax.axis("off")
    se_rel  = np.abs(ml_se - fea_se) / (np.abs(fea_se) + 1e-10) * 100
    rf_rel  = np.abs(ml_rf - fea_rf) / (np.abs(fea_rf) + 1e-10) * 100
    rows = [
        ["Metric",                   "Mean |error|",                         "Mean rel. error"],
        ["Strain energy",            f"{np.mean(np.abs(ml_se-fea_se)):.4f}", f"{np.mean(se_rel):.1f}%"],
        ["Reaction forces",          f"{np.mean(np.abs(ml_rf-fea_rf)):.4f}", f"{np.mean(rf_rel):.1f}%"],
        ["Displacement u_x (field)", f"{np.mean(err_x):.4f}",                f"{np.mean(rel_x):.1f}%"],
        ["Displacement u_y (field)", f"{np.mean(err_y):.4f}",                f"{np.mean(rel_y):.1f}%"],
        ["Displacement |u| (field)", f"{np.mean(err_mag):.4f}",              f"{np.mean(rel_mag):.1f}%"],
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
    ax.set_title("Error summary", fontsize=11, pad=20)
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


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("🗂️ Mode")
    app_mode = st.radio(
        "Select mode",
        ["📤 Upload your image", "🗄️ Dataset browser"],
        index=0,
    )
    if app_mode == "📤 Upload your image":
        st.divider()
        st.header("🖼️ Input type")
        input_mode = st.radio(
            "Select input type",
            ["RGB image (auto preprocessing)", "Binary image (use as-is)"],
            index=0,
        )
        if input_mode == "RGB image (auto preprocessing)":
            st.divider()
            st.header("⚙️ Preprocessing")
            param2     = st.slider("param2 (sensitivity)", 5, 40, 12)
            min_dist   = st.slider("min_dist (px)", 10, 50, 25)
            min_radius = st.slider("min_radius (px)", 3, 20, 8)
            max_radius = st.slider("max_radius (px)", 10, 40, 14)
        else:
            param2 = min_dist = min_radius = max_radius = None
            st.info("Binary image used as-is.\nNo circle detection.\nBlack=fibre (E=10), White=matrix (E=1).")
    st.divider()
    st.markdown("**Model:** Matt's UNetMultiRegression (256×256)")


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
with st.spinner("Loading model..."):
    model, stats, device = load_model()
st.success(f"Model loaded ({'GPU' if device.type == 'cuda' else 'CPU'})")


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
            with st.spinner("Preprocessing..."):
                if is_rgb:
                    binary_255, binary_ml, n_circles = preprocess_rgb(
                        original, min_radius=min_radius, max_radius=max_radius,
                        min_dist=min_dist, param2=param2,
                    )
                    st.metric("Circles detected", n_circles)
                else:
                    binary_255, binary_ml, _ = preprocess_binary_upload(original)
                st.metric("Fibre volume fraction",
                          f"{100*np.sum(binary_255==0)/binary_255.size:.1f}%")
            st.image(binary_255, caption="Binary (400×400, 0/255)",
                     use_container_width=True, clamp=True)

        st.divider()
        with st.spinner("Running ML inference..."):
            ml_disp_x, ml_disp_y, ml_se, ml_rf = run_ml_model(binary_ml, model, stats, device)
        st.success("✅ ML prediction complete!")

        st.divider()
        st.subheader("🔬 FEA Ground Truth (optional — 2-5 min)")

        if "fea_results" not in st.session_state:
            st.session_state.fea_results = None
        if "fea_pattern" not in st.session_state:
            st.session_state.fea_pattern = None

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
        mode_str = "rgb" if is_rgb else "gray"
        fig_ml   = make_ml_figure(original, binary_255, ml_disp_x, ml_disp_y,
                                  ml_se, ml_rf, mode_str)

        if fea_done:
            fea_dx, fea_dy, fea_se, fea_rf = st.session_state.fea_results
            fig_fea = make_fea_figure(binary_255, fea_dx, fea_dy, fea_se, fea_rf)
            fig_cmp = make_comparison_figure(binary_255, ml_disp_x, ml_disp_y, ml_se, ml_rf,
                                             fea_dx, fea_dy, fea_se, fea_rf)
            tab1, tab2, tab3 = st.tabs(["📊 ML Prediction", "🔬 FEA Ground Truth",
                                         "⚖️ ML vs FEA Comparison"])
            with tab1: st.pyplot(fig_ml,  use_container_width=True)
            with tab2: st.pyplot(fig_fea, use_container_width=True)
            with tab3: st.pyplot(fig_cmp, use_container_width=True)
        else:
            tab1, = st.tabs(["📊 ML Prediction"])
            with tab1: st.pyplot(fig_ml, use_container_width=True)

        st.divider()
        st.subheader("📥 Download results")
        cols = st.columns(4)
        with cols[0]: st.download_button("ML strain_energy.txt", arr_to_bytes(ml_se),     file_name="ml_strain_energy.txt")
        with cols[1]: st.download_button("ML rxn_force.txt",     arr_to_bytes(ml_rf),     file_name="ml_rxn_force.txt")
        with cols[2]: st.download_button("ML disp_x.txt",        arr_to_bytes(ml_disp_x), file_name="ml_disp_x.txt")
        with cols[3]: st.download_button("ML plot (.png)",        fig_to_bytes(fig_ml),    file_name="ml_results.png", mime="image/png")

    else:
        st.info("👆 Upload an image to get started.")
        st.markdown("""
        **RGB mode:** Circle detection (HoughCircles) extracts fibres automatically.

        **Binary mode:** Image used as-is — no processing.
        Black pixels = fibres (E=10), white pixels = matrix (E=1).
        """)


# ═════════════════════════════════════════════
# MODE 2: DATASET BROWSER
# ═════════════════════════════════════════════
else:
    st.subheader("🗄️ Dataset Browser — Example Images")
    st.markdown("Select an example image. ML runs instantly. "
                "FEA reference results are loaded from precomputed files (no wait).")

    examples = get_example_images()
    if not examples:
        st.warning(f"No example images found in `{EXAMPLE_DIR / 'input_patterns'}`.")
        st.stop()

    # Thumbnail grid
    st.markdown("**Select an image:**")
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
    binary_ml  = binary_255_to_ml_input(binary_255)

    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(binary_255, caption=f"{selected} (400×400)",
                 use_container_width=True, clamp=True)
        st.metric("Fibre volume fraction",
                  f"{100*np.sum(binary_255==0)/binary_255.size:.1f}%")
    with col2:
        with st.spinner("Running ML inference..."):
            ml_disp_x, ml_disp_y, ml_se, ml_rf = run_ml_model(
                binary_ml, model, stats, device
            )
        st.success("✅ ML prediction complete!")

    results_dir = REPO_DIR / "Results"
    has_fea     = (results_dir / f"{selected}_strain_energy.txt").exists()

    if has_fea:
        fea_dx, fea_dy, fea_se, fea_rf = load_fea_results(selected)
        fig_ml  = make_ml_figure(binary_255, binary_255, ml_disp_x, ml_disp_y,
                                 ml_se, ml_rf, "gray")
        fig_fea = make_fea_figure(binary_255, fea_dx, fea_dy, fea_se, fea_rf)
        fig_cmp = make_comparison_figure(binary_255, ml_disp_x, ml_disp_y, ml_se, ml_rf,
                                          fea_dx, fea_dy, fea_se, fea_rf)
        tab1, tab2, tab3 = st.tabs(["📊 ML Prediction", "🔬 FEA Reference",
                                     "⚖️ ML vs FEA Comparison"])
        with tab1: st.pyplot(fig_ml,  use_container_width=True)
        with tab2: st.pyplot(fig_fea, use_container_width=True)
        with tab3: st.pyplot(fig_cmp, use_container_width=True)
    else:
        st.warning(f"No precomputed FEA results for `{selected}`. "
                   f"Run the FEA batch script to generate them.")
        fig_ml = make_ml_figure(binary_255, binary_255, ml_disp_x, ml_disp_y,
                                ml_se, ml_rf, "gray")
        tab1, = st.tabs(["📊 ML Prediction"])
        with tab1: st.pyplot(fig_ml, use_container_width=True)

    st.divider()
    st.subheader("📥 Download results")
    cols = st.columns(3)
    with cols[0]: st.download_button("ML strain_energy.txt", arr_to_bytes(ml_se),      file_name=f"{selected}_ml_se.txt")
    with cols[1]: st.download_button("ML rxn_force.txt",     arr_to_bytes(ml_rf),      file_name=f"{selected}_ml_rf.txt")
    with cols[2]: st.download_button("ML disp_x.txt",        arr_to_bytes(ml_disp_x),  file_name=f"{selected}_ml_dx.txt")
