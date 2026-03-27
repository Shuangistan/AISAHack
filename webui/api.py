import io
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn

# Setup paths
current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from models import get_model, load_config, default_config, MODEL_REGISTRY

REPO_DIR = parent_dir

AVAILABLE_MODELS = {
    "unet": {"checkpoint_dir": "experiments/reference_models/unet"},
    "unet_small": {"checkpoint_dir": "experiments/reference_models/unet_small"},
    "fno": {"checkpoint_dir": "experiments/reference_models/fno"},
    "swin": {"checkpoint_dir": "experiments/reference_models/swin"},
}

app = FastAPI(title="Mechanical MNIST backend", description="FEA Surrogate API")

# Store loaded models in memory
model_cache = {}

def load_model_from_registry(model_name):
    if model_name in model_cache:
        return model_cache[model_name]
        
    if model_name not in AVAILABLE_MODELS or model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found.")

    ckpt_dir = REPO_DIR / AVAILABLE_MODELS[model_name]["checkpoint_dir"]
    config_path = ckpt_dir / "config.json"
    pt_path = ckpt_dir / "best_model.pt"
    stats_path = ckpt_dir / "norm_stats.npz"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        cfg = load_config(str(config_path)) if config_path.exists() else default_config(model_name)
        model = get_model(cfg).to(device)
        
        ckpt = torch.load(pt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        
        stats = dict(np.load(stats_path, allow_pickle=True))
        
        model_cache[model_name] = (model, stats, device, cfg)
        return model_cache[model_name]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

def binary_255_to_ml_input(binary_255, size):
    resized = cv2.resize(binary_255, (size, size), interpolation=cv2.INTER_AREA)
    return (resized >= 128).astype(np.float32)

@app.post("/predict/{model_name}")
async def predict(model_name: str, file: UploadFile = File(...)):
    """Run inference for a specific model."""
    model, stats, device, cfg = load_model_from_registry(model_name)
    
    # Read uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    binary_255 = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # ML Inference
    size = cfg.img_size
    binary_ml = binary_255_to_ml_input(binary_255, size)
    inp = torch.from_numpy(binary_ml).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(inp)
    
    psi = output["psi"][0].cpu().numpy()
    force = output["force"][0].cpu().numpy()
    disp = output["disp"][0].cpu().numpy()

    # Universal Denormalization
    psi_mean = np.array(stats.get("psi_mean", stats.get("se_mean", 0.0)))
    psi_std = np.array(stats.get("psi_std", stats.get("se_std", 1.0))) + 1e-10
    strain_energy = psi * psi_std + psi_mean
    
    force_mean = np.array(stats.get("force_mean", stats.get("rf_mean", 0.0)))
    force_std = np.array(stats.get("force_std", stats.get("rf_std", 1.0))) + 1e-10
    reaction_forces = (force * force_std + force_mean).reshape(7, 4)
    
    if "disp_mean" in stats:
        d_mean = np.atleast_1d(stats.get("disp_mean", 0.0))
        d_std = np.atleast_1d(stats.get("disp_std", 1.0)) + 1e-10
        if len(d_mean) == 1:
            disp_x = disp[0] * d_std[0] + d_mean[0]
            disp_y = disp[1] * d_std[0] + d_mean[0]
        else:
            disp_x = disp[0] * d_std[0] + d_mean[0]
            disp_y = disp[1] * d_std[1] + d_mean[1]
    else:
        dx_mean = float(stats.get("disp_x_mean", 0.0))
        dx_std = float(stats.get("disp_x_std", 1.0)) + 1e-10
        dy_mean = float(stats.get("disp_y_mean", 0.0))
        dy_std = float(stats.get("disp_y_std", 1.0)) + 1e-10
        disp_x = disp[0] * dx_std + dx_mean
        disp_y = disp[1] * dy_std + dy_mean

    # Convert to standard Python lists for JSON serialization
    return {
        "disp_x": disp_x.tolist(),
        "disp_y": disp_y.tolist(),
        "strain_energy": strain_energy.tolist(),
        "reaction_forces": reaction_forces.tolist()
    }

@app.post("/run_fea/{pattern_name}")
async def run_fea(pattern_name: str, file: UploadFile = File(...)):
    """Run the slow FEniCS FEA pipeline."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    binary_255 = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Save for Gmsh
    input_dir = REPO_DIR / "input_patterns"
    if input_dir.exists(): shutil.rmtree(input_dir)
    input_dir.mkdir()
    np.savetxt(input_dir / f"{pattern_name}.txt", binary_255, fmt="%d")

    mesh_dir = REPO_DIR / "mesh_files"
    results_dir = REPO_DIR / "Results"
    mesh_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    xdmf_path = mesh_dir / f"{pattern_name}.xdmf"

    python = sys.executable
    
    # 1. Mesh Geometry
    r1 = subprocess.run([python, "NumpyImageToGmsh.py"], cwd=REPO_DIR, capture_output=True, text=True)
    if r1.returncode != 0: raise HTTPException(status_code=500, detail=f"Gmsh prep failed: {r1.stderr}")
    
    # 2. Meshing
    mesh_script = mesh_dir / f"{pattern_name}.py"
    r2 = subprocess.run([python, str(mesh_script)], cwd=mesh_dir, capture_output=True, text=True)
    if r2.returncode != 0: raise HTTPException(status_code=500, detail=f"Meshing failed: {r2.stderr}")

    # 3. FEniCS
    r3 = subprocess.run([python, "Equibiaxial_Hyperelastic.py", str(xdmf_path)], cwd=REPO_DIR, capture_output=True, text=True)
    if r3.returncode != 0: raise HTTPException(status_code=500, detail=f"FEniCS failed: {r3.stderr}")

    # Load results to send back
    d = results_dir
    def f(s): return d / f"{pattern_name}{s}"
    
    return {
        "fea_dx": np.loadtxt(f("_pixel_disp_0.5_x.txt")).tolist(),
        "fea_dy": np.loadtxt(f("_pixel_disp_0.5_y.txt")).tolist(),
        "fea_se": np.loadtxt(f("_strain_energy.txt")).tolist(),
        "fea_rf": np.loadtxt(f("_rxn_force.txt")).tolist()
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)