"""
app.py — Mechanical MNIST Demo App (Updated)
==============================================
Integrates with the new registry-based model architecture.

Supports multiple models through the MODEL_REGISTRY:
  - UNet (256x256, norm_stats.npz)
  - [Additional models can be added to models/__init__.py]

File structure expected:
    app.py
    models/
      __init__.py        ← MODEL_REGISTRY, CONFIG_REGISTRY
      base.py           ← MechMNISTModel abstract base
      unet.py           ← UNetMultiRegression, UNetConfig
      [other models]
    checkpoints/
      unet/
        config.json     ← Config saved from training
        best_model.pt   ← Model checkpoint
        norm_stats.npz  ← Normalization statistics
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

current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import the new model registry system
from models import get_model, load_config, default_config, MODEL_REGISTRY

REPO_DIR    = parent_dir
EXAMPLE_DIR = REPO_DIR / "example"
DISP_VALS   = [0.0, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5]

# ─────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────
# Define available models using the registry.
# Each model is identified by its model_name (key in MODEL_REGISTRY).
# The UI can support multiple models by adding entries here.

AVAILABLE_MODELS = {
    "unet": {
        "label": "UNet Multi-Regression",
        "description": "256×256 U-Net with auxiliary scalar heads",
        "checkpoint_dir": "experiments/reference_models/unet",
    },
    "unet_small": {
        "label": "UNet Multi-Regression 64x64",
        "description": "64×64 U-net with multi-task learning",
        "checkpoint_dir": "experiments/reference_models/unet_small",
    },
    "fno": {
        "label": "Fourier Neural Operator",
        "description": "64×64 FNO with multi-task learning",
        "checkpoint_dir": "experiments/reference_models/fno",
    },
    "swin": {
        "label": "Swin Transformer",
        "description": "64×64 Swin Transformer with multi-task learning",
        "checkpoint_dir": "experiments/reference_models/swin",
    },
    # Template for adding more models:
    # "fno": {
    #     "label": "Fourier Neural Operator",
    #     "description": "64×64 FNO with multi-task learning",
    #     "checkpoint_dir": "experiments/reference_models/fno",
    # },
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
# LOAD MODEL (using new registry)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model_from_registry(model_name):
    """
    Load a model using the new registry-based system.
    
    Parameters
    ----------
    model_name : str
        Key in MODEL_REGISTRY (e.g., "unet")
    
    Returns
    -------
    tuple
        (model, norm_stats, device, error_msg)
    """
    if model_name not in AVAILABLE_MODELS:
        return None, None, None, f"Model '{model_name}' not in AVAILABLE_MODELS"
    
    if model_name not in MODEL_REGISTRY:
        return None, None, None, f"Model '{model_name}' not in MODEL_REGISTRY"
    
    ckpt_dir = REPO_DIR / AVAILABLE_MODELS[model_name]["checkpoint_dir"]
    config_path = ckpt_dir / "config.json"
    pt_path = ckpt_dir / "best_model.pt"
    stats_path = ckpt_dir / "norm_stats.npz"
    
    # Verify files exist
    if not ckpt_dir.exists():
        return None, None, None, f"Checkpoint directory not found: {ckpt_dir}"
    if not pt_path.exists():
        return None, None, None, f"Model file not found: {pt_path}"
    if not stats_path.exists():
        return None, None, None, f"Normalization stats not found: {stats_path}"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Load config (automatically handles subclass instantiation)
        if config_path.exists():
            cfg = load_config(str(config_path))
        else:
            cfg = default_config(model_name)
        
        # Create model from config
        model = get_model(cfg).to(device)
        
        # Load checkpoint
        ckpt = torch.load(pt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        
        # Load normalization statistics
        # stats = np.load(stats_path, allow_pickle=True).item()
        stats = dict(np.load(stats_path,allow_pickle=True))
        
        return model, stats, device, None
    
    except Exception as e:
        return None, None, None, f"Error loading model: {str(e)}"


# ─────────────────────────────────────────────
# CORE CONVERSION & PREPROCESSING
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


def preprocess_rgb(img_rgb, min_radius=8, max_radius=14, min_dist=25, param2=12, size=400):
    """Circle detection via HoughCircles for fiber reinforced composites."""
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
# ML INFERENCE (unified for all models)
# ─────────────────────────────────────────────
def run_model(model_name, binary_255, model, stats, device, cfg):
    """
    Run inference using the new model architecture.
    
    All models inherit from MechMNISTModel and return:
        {"psi": (B, 7), "force": (B, 28), "disp": (B, 2, H, W)}
    
    Returns
    -------
    tuple
        (disp_x, disp_y, strain_energy, reaction_forces)
    """
    # Resize to model's expected input size
    size = cfg.img_size
    binary_ml = binary_255_to_ml_input(binary_255, size)
    
    inp = torch.from_numpy(binary_ml).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(inp)  # Returns dict with "psi", "force", "disp"
    
    # Extract outputs
    psi = output["psi"][0].cpu().numpy()
    force = output["force"][0].cpu().numpy()
    disp = output["disp"][0].cpu().numpy()
    
    # --- UNIVERSAL DENORMALIZATION (New Architecture and Old Architecture)---
    # 1. Strain Energy (handles new 'psi' keys OR old 'se' keys)
    psi_mean = np.array(stats.get("psi_mean", stats.get("se_mean", 0.0)))
    psi_std = np.array(stats.get("psi_std", stats.get("se_std", 1.0))) + 1e-10
    strain_energy = psi * psi_std + psi_mean
    
    # 2. Reaction Forces (handles new 'force' keys OR old 'rf' keys)
    force_mean = np.array(stats.get("force_mean", stats.get("rf_mean", 0.0)))
    force_std = np.array(stats.get("force_std", stats.get("rf_std", 1.0))) + 1e-10
    reaction_forces = (force * force_std + force_mean).reshape(7, 4)
    
    # 3. Displacement Fields (handles grouped arrays OR separated scalars)
    if "disp_mean" in stats:
        # np.atleast_1d gracefully handles both single numbers and arrays
        d_mean = np.atleast_1d(stats.get("disp_mean", 0.0))
        d_std = np.atleast_1d(stats.get("disp_std", 1.0)) + 1e-10
        
        if len(d_mean) == 1: # If it was saved as a single scalar
            disp_x = disp[0] * d_std[0] + d_mean[0]
            disp_y = disp[1] * d_std[0] + d_mean[0]
        else:                # If it was saved as an array [x_mean, y_mean]
            disp_x = disp[0] * d_std[0] + d_mean[0]
            disp_y = disp[1] * d_std[1] + d_mean[1]
    else:
        # Fallback to the exact keys from the old scripts
        dx_mean = float(stats.get("disp_x_mean", 0.0))
        dx_std = float(stats.get("disp_x_std", 1.0)) + 1e-10
        dy_mean = float(stats.get("disp_y_mean", 0.0))
        dy_std = float(stats.get("disp_y_std", 1.0)) + 1e-10
        
        disp_x = disp[0] * dx_std + dx_mean
        disp_y = disp[1] * dy_std + dy_mean
    # ----------------------------------------
    
    return disp_x, disp_y, strain_energy, reaction_forces


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
# FIGURE GENERATION
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


def make_single_model_tab(binary_255, display_img, mode, model_name, model, stats, device, cfg, pattern_name):
    """Run a model and create its visualization."""
    with st.spinner(f"Running {AVAILABLE_MODELS[model_name]['label']}..."):
        disp_x, disp_y, strain_energy, reaction_forces = run_model(
            model_name, binary_255, model, stats, device, cfg
        )
    st.success(f"✅ {AVAILABLE_MODELS[model_name]['label']} complete!")
    
    fig = make_single_fig(binary_255, display_img, mode, disp_x, disp_y,
                          strain_energy, reaction_forces,
                          AVAILABLE_MODELS[model_name]["label"])
    st.pyplot(fig, use_container_width=True)
    return fig, (disp_x, disp_y, strain_energy, reaction_forces)


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()


def get_example_images():
    """Find example binary patterns in example/input_patterns/."""
    patterns_dir = EXAMPLE_DIR / "input_patterns"
    if not patterns_dir.exists():
        return []
    
    examples = []
    for txt_file in sorted(patterns_dir.glob("*.txt")):
        name = txt_file.stem
        has_fea = (REPO_DIR / "Results" / f"{name}_strain_energy.txt").exists()
        examples.append((name, has_fea))
    
    return examples


# ═════════════════════════════════════════════
# SIDEBAR: MODEL SELECTION
# ═════════════════════════════════════════════
st.sidebar.header("⚙️ Configuration")

st.sidebar.subheader("Model Selection")
selected_models = st.sidebar.multiselect(
    "Which models to run?",
    list(AVAILABLE_MODELS.keys()),
    default=list(AVAILABLE_MODELS.keys())[0] if AVAILABLE_MODELS else [],
    format_func=lambda k: AVAILABLE_MODELS[k]["label"],
)

if not selected_models:
    st.warning("Please select at least one model in the sidebar.")
    st.stop()

# Load selected models
models = {}
configs = {}
device_map = {}

for model_name in selected_models:
    model, stats, device, error = load_model_from_registry(model_name)
    if error:
        st.sidebar.error(f"❌ {model_name}: {error}")
    else:
        models[model_name] = (model, stats)
        device_map[model_name] = device
        cfg = load_config(str(REPO_DIR / AVAILABLE_MODELS[model_name]["checkpoint_dir"] / "config.json")) \
              if (REPO_DIR / AVAILABLE_MODELS[model_name]["checkpoint_dir"] / "config.json").exists() \
              else default_config(model_name)
        configs[model_name] = cfg
        st.sidebar.success(f"✅ {AVAILABLE_MODELS[model_name]['label']}")

if not models:
    st.error("No models could be loaded. Check the sidebar for errors.")
    st.stop()

# Image input mode
st.sidebar.subheader("Input Source")
app_mode = st.sidebar.radio(
    "Select mode:",
    ["📤 Upload your image", "🗄️ Dataset Browser"],
    index=0,
)

input_mode = st.sidebar.radio(
    "Image type:",
    ["RGB image (auto preprocessing)", "Binary B&W image"],
    index=0,
)

if input_mode == "RGB image (auto preprocessing)":
    st.sidebar.subheader("Circle Detection Parameters")
    param2 = st.sidebar.slider("param2 (accumulator)", 5, 30, 12)
    min_dist = st.sidebar.slider("min_dist", 10, 50, 25)
    min_radius = st.sidebar.slider("min_radius", 3, 15, 8)
    max_radius = st.sidebar.slider("max_radius", 10, 30, 14)
else:
    param2 = min_dist = min_radius = max_radius = None


# ═════════════════════════════════════════════
# SHARED INFERENCE + DISPLAY LOGIC
# ═════════════════════════════════════════════
def run_and_display(binary_255, display_img, mode, pattern_name):
    results = {}

    # Run all selected models
    for model_name in selected_models:
        model, stats = models[model_name]
        device = device_map[model_name]
        cfg = configs[model_name]
        
        with st.spinner(f"Running {AVAILABLE_MODELS[model_name]['label']}..."):
            disp_x, disp_y, strain_energy, reaction_forces = run_model(
                model_name, binary_255, model, stats, device, cfg
            )
        st.success(f"✅ {AVAILABLE_MODELS[model_name]['label']} complete!")
        results[model_name] = (disp_x, disp_y, strain_energy, reaction_forces)

    # FEA button
    st.divider()
    st.subheader("🔬 FEA Ground Truth (optional — 2-5 min)")

    if "fea_results" not in st.session_state:
        st.session_state.fea_results = None
    if "fea_pattern" not in st.session_state:
        st.session_state.fea_pattern = None

    if st.button("▶ Run FEA", type="primary"):
        save_binary_for_fea(binary_255, pattern_name)
        pb = st.progress(0)
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

    # Create tabs
    tab_names = []
    tab_data = {}
    
    for model_name in selected_models:
        tab_names.append(f"📊 {AVAILABLE_MODELS[model_name]['label']}")
        tab_data[tab_names[-1]] = ("model", model_name)
    
    if len(selected_models) > 1:
        tab_names.append("⚖️ Model Comparison")
        tab_data[tab_names[-1]] = ("comparison", None)
    
    if fea_done:
        for model_name in selected_models:
            tab_names.append(f"⚖️ {AVAILABLE_MODELS[model_name]['label']} vs FEA")
            tab_data[tab_names[-1]] = ("vs_fea", model_name)
        tab_names.append("🔬 FEA Ground Truth")
        tab_data[tab_names[-1]] = ("fea", None)

    tabs = st.tabs(tab_names)
    
    for tab, tab_name in zip(tabs, tab_names):
        with tab:
            tab_type, data = tab_data[tab_name]
            
            if tab_type == "model":
                model_name = data
                dx, dy, se, rf = results[model_name]
                fig = make_single_fig(binary_255, display_img, mode, dx, dy, se, rf,
                                     AVAILABLE_MODELS[model_name]["label"])
                st.pyplot(fig, use_container_width=True)
            
            elif tab_type == "comparison":
                st.info("Comparison plots for multiple models will be displayed here.")
                # Can be extended to show side-by-side comparisons
            
            elif tab_type == "vs_fea":
                model_name = data
                dx, dy, se, rf = results[model_name]
                fea_dx, fea_dy, fea_se, fea_rf = st.session_state.fea_results
                st.info(f"Comparison between {AVAILABLE_MODELS[model_name]['label']} and FEA")
                # Can be extended to show side-by-side comparisons with error metrics
            
            elif tab_type == "fea":
                fea_dx, fea_dy, fea_se, fea_rf = st.session_state.fea_results
                fig = make_single_fig(binary_255, binary_255, "gray", fea_dx, fea_dy,
                                     fea_se, fea_rf, "FEA Ground Truth")
                st.pyplot(fig, use_container_width=True)


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
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        pattern_name = Path(uploaded.name).stem.replace(" ", "_")

        if is_rgb:
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
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
        st.warning(f"No example images found in {EXAMPLE_DIR / 'input_patterns'}.")
        st.stop()

    n_cols = min(len(examples), 6)
    thumb_cols = st.columns(n_cols)
    if "browser_selected" not in st.session_state:
        st.session_state.browser_selected = examples[0][0]

    for i, (name, has_fea) in enumerate(examples):
        txt_path = EXAMPLE_DIR / "input_patterns" / f"{name}.txt"
        binary_255 = load_dataset_txt(txt_path)
        thumb = cv2.resize(binary_255, (128, 128), interpolation=cv2.INTER_NEAREST)
        with thumb_cols[i % n_cols]:
            st.image(thumb, caption=name, use_container_width=True)
            st.caption("✅ FEA ready" if has_fea else "⏳ No FEA yet")
            if st.button("Select", key=f"btn_{name}"):
                st.session_state.browser_selected = name

    st.divider()
    selected = st.session_state.browser_selected
    st.markdown(f"### Selected: `{selected}`")

    txt_path = EXAMPLE_DIR / "input_patterns" / f"{selected}.txt"
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