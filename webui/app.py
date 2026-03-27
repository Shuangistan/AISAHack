import io
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import requests
import streamlit as st

API_URL = "http://localhost:8000"
REPO_DIR = Path(__file__).parent.resolve().parent
EXAMPLE_DIR = REPO_DIR / "example"
DISP_VALS = [0.0, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5]

AVAILABLE_MODELS = {
    "unet": {"label": "UNet Multi-Regression"},
    "unet_small": {"label": "UNet Multi-Regression 64x64"},
    "fno": {"label": "Fourier Neural Operator"},
    "swin": {"label": "Swin Transformer"},
}

st.set_page_config(page_title="Mechanical MNIST — ML Demo", page_icon="⚙️", layout="wide")
st.title("⚙️ Mechanical MNIST — ML Inference & FEA Comparison")

# ─────────────────────────────────────────────
# PREPROCESSING (Happens instantly in UI)
# ─────────────────────────────────────────────
def to_binary_255(img_gray):
    if img_gray.max() <= 1.0: return (img_gray * 255).astype(np.uint8)
    _, binary = cv2.threshold(img_gray.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    return binary

def load_dataset_txt(txt_path, size=400):
    img = np.loadtxt(txt_path)
    binary_255 = to_binary_255(img)
    if binary_255.shape != (size, size):
        binary_255 = cv2.resize(binary_255, (size, size), interpolation=cv2.INTER_NEAREST)
    return binary_255

def preprocess_rgb(img_rgb, min_radius=8, max_radius=14, min_dist=25, param2=12, size=400):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized, (5, 5), 1)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist, param1=50, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    binary = np.ones((size, size), dtype=np.uint8) * 255
    n_circles = 0
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        circles = [c for c in circles if c[0]-c[2]>1 and c[0]+c[2]<size-1 and c[1]-c[2]>1 and c[1]+c[2]<size-1]
        n_circles = len(circles)
        for (x, y, r) in circles:
            cv2.circle(binary, (x, y), r, 0, -1)
    return binary, n_circles

# ─────────────────────────────────────────────
# HTTP REQUESTS TO BACKEND
# ─────────────────────────────────────────────
def fetch_ml_prediction(model_name, binary_255):
    """Send image to FastAPI and get numpy arrays back."""
    success, encoded_img = cv2.imencode('.png', binary_255)
    file_bytes = encoded_img.tobytes()
    
    response = requests.post(
        f"{API_URL}/predict/{model_name}", 
        files={"file": ("image.png", file_bytes, "image/png")}
    )
    if response.status_code != 200:
        raise Exception(response.json().get("detail", "API Error"))
        
    data = response.json()
    return (
        np.array(data["disp_x"]), 
        np.array(data["disp_y"]), 
        np.array(data["strain_energy"]), 
        np.array(data["reaction_forces"])
    )

def fetch_fea_pipeline(pattern_name, binary_255):
    """Trigger the slow FEniCS pipeline via FastAPI."""
    success, encoded_img = cv2.imencode('.png', binary_255)
    file_bytes = encoded_img.tobytes()
    
    response = requests.post(
        f"{API_URL}/run_fea/{pattern_name}", 
        files={"file": ("image.png", file_bytes, "image/png")}
    )
    if response.status_code != 200:
        raise Exception(response.json().get("detail", "API Error"))
        
    data = response.json()
    return (
        np.array(data["fea_dx"]), 
        np.array(data["fea_dy"]), 
        np.array(data["fea_se"]), 
        np.array(data["fea_rf"])
    )

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
        im   = ax.imshow(data, cmap=cmap, origin="upper", vmin=-vmax if vmax else None, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def _scalar_axes(gs, row, fig, se, rf, label, color_se="b"):
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]
    ls = "-o" if color_se == "b" else "--s"

    ax = fig.add_subplot(gs[row, 0:2])
    ax.plot(DISP_VALS, se, ls, color=color_se, linewidth=2, markersize=5, label=label)
    ax.set_xlabel("d", fontsize=10); ax.set_ylabel("Strain energy", fontsize=10)
    ax.set_title("Strain energy vs displacement", fontsize=11); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[row, 2:])
    rf_labels = ["F_left_x", "F_right_x", "F_bottom_y", "F_top_y"]
    for i, (lbl, c) in enumerate(zip(rf_labels, colors)):
        ax.plot(DISP_VALS, rf[:, i], ls, color=c, linewidth=2, markersize=5, label=f"{label} {lbl}")
    ax.set_xlabel("d", fontsize=10); ax.set_ylabel("Reaction force", fontsize=10)
    ax.set_title("Reaction forces vs displacement", fontsize=11); ax.legend(fontsize=7, loc="upper left", ncol=2); ax.grid(True, alpha=0.3)

def make_single_fig(binary_255, display_img, mode, disp_x, disp_y, se, rf, title):
    fig = plt.figure(figsize=(22, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(display_img) if mode == "rgb" else ax.imshow(display_img, cmap="gray")
    ax.set_title("Input pattern", fontsize=11); ax.axis("off")

    _disp_row(gs, 0, fig, disp_x, disp_y, "")
    _scalar_axes(gs, 1, fig, se, rf, title[:20])
    return fig

# ═════════════════════════════════════════════
# SIDEBAR: MODEL SELECTION
# ═════════════════════════════════════════════
st.sidebar.header("⚙️ Configuration")
selected_models = st.sidebar.multiselect(
    "Which models to run?", list(AVAILABLE_MODELS.keys()),
    default=list(AVAILABLE_MODELS.keys())[0] if AVAILABLE_MODELS else [],
    format_func=lambda k: AVAILABLE_MODELS[k]["label"],
)

if not selected_models:
    st.warning("Please select at least one model.")
    st.stop()

app_mode = st.sidebar.radio("Select mode:", ["📤 Upload your image", "🗄️ Dataset Browser"], index=0)
input_mode = st.sidebar.radio("Image type:", ["RGB image (auto preprocessing)", "Binary B&W image"], index=0)

if input_mode == "RGB image (auto preprocessing)":
    param2 = st.sidebar.slider("param2", 5, 30, 12)
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

    for model_name in selected_models:
        with st.spinner(f"Requesting API for {AVAILABLE_MODELS[model_name]['label']}..."):
            try:
                results[model_name] = fetch_ml_prediction(model_name, binary_255)
                st.success(f"✅ {AVAILABLE_MODELS[model_name]['label']} complete!")
            except Exception as e:
                st.error(f"❌ API Error for {model_name}: {e}")

    st.divider()
    st.subheader("🔬 FEA Ground Truth (optional — 2-5 min)")

    if "fea_results" not in st.session_state: st.session_state.fea_results = None
    if "fea_pattern" not in st.session_state: st.session_state.fea_pattern = None

    if st.button("▶ Run FEA via API", type="primary"):
        with st.spinner("Server is running FEA pipeline... This takes 2-5 minutes."):
            try:
                st.session_state.fea_results = fetch_fea_pipeline(pattern_name, binary_255)
                st.session_state.fea_pattern = pattern_name
                st.success("✅ FEA complete!")
            except Exception as e:
                st.error(f"FEA API Error: {e}")

    fea_done = (st.session_state.fea_results is not None and st.session_state.fea_pattern == pattern_name)
    st.divider()

    # Create tabs
    tab_names = [f"📊 {AVAILABLE_MODELS[m]['label']}" for m in selected_models]
    if fea_done: tab_names.append("🔬 FEA Ground Truth")

    tabs = st.tabs(tab_names)
    
    for i, model_name in enumerate(selected_models):
        if model_name in results:
            with tabs[i]:
                dx, dy, se, rf = results[model_name]
                fig = make_single_fig(binary_255, display_img, mode, dx, dy, se, rf, AVAILABLE_MODELS[model_name]["label"])
                st.pyplot(fig, use_container_width=True)

    if fea_done:
        with tabs[-1]:
            fea_dx, fea_dy, fea_se, fea_rf = st.session_state.fea_results
            fig = make_single_fig(binary_255, binary_255, "gray", fea_dx, fea_dy, fea_se, fea_rf, "FEA Ground Truth")
            st.pyplot(fig, use_container_width=True)

# ═════════════════════════════════════════════
# MODE 1 & 2 ROUTING
# ═════════════════════════════════════════════
if app_mode == "📤 Upload your image":
    is_rgb = input_mode == "RGB image (auto preprocessing)"
    uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp", "bmp"])

    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        pattern_name = Path(uploaded.name).stem.replace(" ", "_")
        
        if is_rgb:
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            original = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        col1, col2 = st.columns([1, 2])
        with col1: st.image(original, caption="Uploaded image", use_container_width=True, clamp=not is_rgb)
        with col2:
            if is_rgb:
                binary_255, n_circles = preprocess_rgb(original, min_radius=min_radius, max_radius=max_radius, min_dist=min_dist, param2=param2)
                st.metric("Circles detected", n_circles)
            else:
                binary_255 = to_binary_255(original)
                binary_255 = cv2.resize(binary_255, (400, 400), interpolation=cv2.INTER_NEAREST)
            st.metric("Fibre volume fraction", f"{100*np.sum(binary_255==0)/binary_255.size:.1f}%")
            st.image(binary_255, caption="Binary (400×400)", use_container_width=True, clamp=True)

        st.divider()
        run_and_display(binary_255, original, "rgb" if is_rgb else "gray", pattern_name)
else:
    st.subheader("🗄️ Dataset Browser — Example Images")
    examples = []
    if EXAMPLE_DIR.exists():
        for txt_file in sorted((EXAMPLE_DIR / "input_patterns").glob("*.txt")):
            examples.append(txt_file.stem)
    
    if not examples:
        st.warning("No examples found.")
        st.stop()

    if "browser_selected" not in st.session_state: st.session_state.browser_selected = examples[0]
    
    cols = st.columns(6)
    for i, name in enumerate(examples):
        txt_path = EXAMPLE_DIR / "input_patterns" / f"{name}.txt"
        binary_255 = load_dataset_txt(txt_path)
        with cols[i % 6]:
            st.image(cv2.resize(binary_255, (128, 128), interpolation=cv2.INTER_NEAREST), caption=name)
            if st.button("Select", key=f"btn_{name}"): st.session_state.browser_selected = name

    st.divider()
    selected = st.session_state.browser_selected
    st.markdown(f"### Selected: `{selected}`")
    binary_255 = load_dataset_txt(EXAMPLE_DIR / "input_patterns" / f"{selected}.txt")
    run_and_display(binary_255, binary_255, "gray", selected)