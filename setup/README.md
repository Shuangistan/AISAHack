# Requirements Files Guide

This project provides multiple `requirements.txt` files for different use cases. Choose the one that matches your needs.

---

## 📋 Which File to Use?

### **Use Case 1: Just Run the UI (Inference Only)**
```bash
pip install -r requirements-ui-only.txt
streamlit run app.py
```
**When:** You have pre-trained checkpoints and just want to run inference

**What's included:**
- ✅ PyTorch & Streamlit
- ✅ Image processing (OpenCV)
- ✅ Visualization (Matplotlib)
- ❌ Training utilities
- ❌ FEA pipeline

**Smallest footprint** (~500MB)

---

### **Use Case 2: Train Models + Run UI**
```bash
pip install -r requirements-training.txt
python train.py --data_root ./data --epochs 100
streamlit run app.py
```
**When:** You want to train new models and run the UI

**What's included:**
- ✅ PyTorch (with training)
- ✅ Data loading & preprocessing
- ✅ Streamlit UI
- ✅ Training monitoring
- ✅ Visualization
- ❌ FEA pipeline

**Medium footprint** (~800MB)

---

### **Use Case 3: Everything (Full Hackathon Setup)**
```bash
pip install -r requirements.txt
pip install -r requirements-fea.txt

# On Linux, also install FEniCS:
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt-get update
sudo apt-get install fenics python3-ufl

# Now you can:
python train.py --data_root ./data
streamlit run app.py
python Equibiaxial_Hyperelastic.py mesh_files/pattern.xdmf
```
**When:** Full hackathon setup with all features

**What's included:**
- ✅ Training
- ✅ UI inference
- ✅ FEA mesh generation & simulation
- ✅ All visualization tools

**Largest footprint** (~2GB with FEniCS)

---

### **Use Case 4: Specific Features**

#### Just Training:
```bash
pip install -r requirements-training.txt
python train.py --data_root ./data
```

#### UI + FEA:
```bash
pip install -r requirements-ui-only.txt
pip install -r requirements-fea.txt
streamlit run app.py
# FEA pipeline will work when button clicked
```

#### FEA Only (no UI):
```bash
pip install -r requirements-fea.txt
python NumpyImageToGmsh.py
python Equibiaxial_Hyperelastic.py mesh_files/pattern.xdmf
```

---

## 🚀 Installation Steps

### **Quick Setup (5 minutes)**

1. **Clone or download the project**
```bash
cd your-project-directory
```

2. **Choose your requirements file** (see guide above)
```bash
# For UI only:
pip install -r requirements-ui-only.txt

# OR for training + UI:
pip install -r requirements-training.txt

# OR for everything:
pip install -r requirements.txt
```

3. **Verify installation**
```bash
python -c "import torch, streamlit, numpy, cv2, matplotlib; print('✅ All imports successful!')"
```

4. **Run the app**
```bash
streamlit run app.py
```

---

### **Full Setup with FEA (10 minutes)**

1. **Install main requirements**
```bash
pip install -r requirements.txt
pip install -r requirements-fea.txt
```

2. **Install FEniCS (Linux/WSL)**
```bash
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt-get update
sudo apt-get install fenics python3-ufl
```

3. **Verify FEniCS**
```bash
python -c "from fenics import *; print('✅ FEniCS installed!')"
```

4. **Run setup script** (optional, for additional configuration)
```bash
bash setup.sh
```

---

## 📦 What's in Each Requirements File

### `requirements.txt` (Complete)
```
PyTorch           → Neural network training & inference
Streamlit         → Web UI
NumPy             → Numerical computing
OpenCV            → Image processing
Matplotlib        → Visualization & plotting
scikit-learn      → ML utilities
PyGMSH            → Mesh generation
Meshio            → Mesh I/O
gmsh              → Mesh backend
h5py              → HDF5 support
Rich              → Pretty output
```

### `requirements-ui-only.txt` (Minimal)
```
PyTorch           → Model inference
Streamlit         → Web UI
NumPy             → Numerical operations
OpenCV            → Image processing
Matplotlib        → Visualization
h5py              → Optional for FEA output
```

### `requirements-training.txt` (ML Development)
```
PyTorch           → Full ML pipeline
Streamlit         → Visualization
NumPy, Pandas     → Data handling
OpenCV, Pillow    → Image ops
Matplotlib        → Plotting
scikit-learn      → Metrics
tqdm              → Progress bars
Rich              → Output formatting
```

### `requirements-fea.txt` (Simulation)
```
PyGMSH            → Mesh generation
Meshio            → Mesh handling
gmsh              → Mesh engine
h5py              → File format
OpenCV            → Image processing
```

---

## ⚠️ Troubleshooting

### "ModuleNotFoundError: No module named 'streamlit'"
**Fix:** You're using `requirements-ui-only.txt` for training. Use `requirements-training.txt` instead.

### "ModuleNotFoundError: No module named 'torch'"
**Fix:** Run one of the requirements files with pip first.

### "ModuleNotFoundError: No module named 'pygmsh'"
**Fix:** You're using `requirements-ui-only.txt` but trying to run FEA. Run:
```bash
pip install -r requirements-fea.txt
```

### FEniCS import error
**Fix:** FEniCS must be installed via apt (Linux only). Run:
```bash
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt-get update
sudo apt-get install fenics python3-ufl
```

### "CUDA out of memory" during training
**Fix:** Reduce batch size in training:
```bash
python train.py --batch_size 4 --data_root ./data
```

---

## 🔍 Check Your Installation

```bash
# Test all imports
python -c "
import torch
import streamlit
import numpy
import cv2
import matplotlib
import sklearn
import tqdm
from PIL import Image
print('✅ Core dependencies OK')
try:
    import pygmsh
    import meshio
    print('✅ FEA dependencies OK')
except:
    print('⚠️  FEA dependencies not installed (optional)')
"

# Verify GPU (if available)
python -c "import torch; print('GPU available:', torch.cuda.is_available())"

# Verify models
python -c "from models import MODEL_REGISTRY; print('Available models:', list(MODEL_REGISTRY.keys()))"
```

---

## Version Constraints

**Why some packages are pinned (exact versions):**
- `pygmsh==6.1.0` → Known compatible with meshio 5.3.4
- `meshio==5.3.4` → Known compatible with pygmsh 6.1.0
- `numpy<2.0` → Some packages still have issues with NumPy 2.0

**Why others use ranges:**
- `torch>=2.0.0,<3.0` → Allows bug fixes while maintaining compatibility
- `streamlit>=1.28.0` → Later versions have better features

---

## Recommended Setup for Hackathon

```bash
# Full setup (recommended):
pip install -r requirements.txt
pip install -r requirements-fea.txt

# Install FEniCS on Linux:
sudo apt-get update
sudo apt-get install -y fenics python3-ufl

# Verify everything:
python -c "from models import get_model; import streamlit; import torch; print('✅ Ready!')"

# Run:
streamlit run app.py
```

---

## Reference

- **PyTorch:** https://pytorch.org/
- **Streamlit:** https://streamlit.io/
- **OpenCV:** https://opencv.org/
- **Matplotlib:** https://matplotlib.org/
- **FEniCS:** https://fenicsproject.org/ (Linux only, install via apt)

---

## Important Notes

### GPU vs CPU
- If you have NVIDIA GPU: PyTorch will use it automatically
- If CPU only: Everything still works, just slower
- Check: `python -c "import torch; print(torch.cuda.is_available())"`

### FEniCS Limitation
- **Linux only** (Ubuntu recommended)
- **Cannot install via pip** on most systems
- Must use: `sudo apt-get install fenics`
- For WSL2, works fine in Ubuntu environment

### Memory Requirements
- **UI only:** ~500MB
- **Training:** ~2GB
- **Full with FEniCS:** ~3GB
