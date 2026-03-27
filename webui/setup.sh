#!/bin/bash
# =============================================================================
# setup.sh — Full environment setup for Mechanical MNIST FEA pipeline
# Tested on: WSL2 Ubuntu 24.04, March 2026
#
# Usage:
#   bash setup.sh
#
# What this does:
#   1. Installs FEniCS legacy via apt
#   2. Creates a Python venv with system site-packages (for FEniCS access)
#   3. Installs Python dependencies (numpy, opencv, pygmsh, meshio, gmsh, h5py)
#   4. Patches pygmsh/helpers.py for compatibility with meshio 5.x and numpy 2.x
#   5. Clones the Mechanical MNIST Cahn-Hilliard repo
#   6. Patches Equibiaxial_Hyperelastic.py (hardcoded BU path fix)
#   7. Runs a quick smoke test
# =============================================================================

set -e  # Exit on any error

echo "============================================"
echo " Mechanical MNIST FEA - Environment Setup"
echo "============================================"

# -------------------------------------
# 1. System dependencies
# -------------------------------------
echo "[1/7] Installing system packages..."
sudo apt-get update -q
sudo apt-get install -y -q \
    software-properties-common \
    git \
    libglu1-mesa \
    python3-full \
    python3-pip

# FEniCS legacy PPA
sudo add-apt-repository -y ppa:fenics-packages/fenics
sudo apt-get update -q
sudo apt-get install -y -q fenics python3-ufl

echo "      FEniCS installed OK"

# -------------------------------------
# 2. Python virtual environment
# -------------------------------------
echo "[2/7] Creating Python venv with system site-packages..."
python3 -m venv ~/fenics_env --system-site-packages
source ~/fenics_env/bin/activate

# Quick check
python3 -c "from fenics import *; print('      FEniCS import OK')"

# -------------------------------------
# 3. Python packages
# -------------------------------------
echo "[3/7] Installing Python packages..."
pip install -q \
    "numpy<2.0" \
    opencv-python \
    "pygmsh==6.1.0" \
    "meshio==5.3.4" \
    gmsh \
    h5py \
    rich

echo "      Python packages installed OK"

# -------------------------------------
# 4. Patch pygmsh helpers.py
# -------------------------------------
echo "[4/7] Patching pygmsh/helpers.py for meshio 5.x compatibility..."

HELPERS_PATH=$(python3 -c "import pygmsh; import os; print(os.path.join(os.path.dirname(pygmsh.__file__), 'helpers.py'))")

# Check if patch file is available next to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/helpers_patched.py"

if [ -f "$PATCH_FILE" ]; then
    cp "$PATCH_FILE" "$HELPERS_PATH"
    echo "      Patched from helpers_patched.py"
else
    echo "      WARNING: helpers_patched.py not found next to setup.sh"
    echo "      Please manually copy helpers_patched.py to: $HELPERS_PATH"
fi

# -------------------------------------
# 5. Clone repo
# -------------------------------------
echo "[5/7] Cloning Mechanical MNIST Cahn-Hilliard repo..."
cd ~
if [ -d "Mechanical-MNIST-Cahn-Hilliard" ]; then
    echo "      Repo already exists, skipping clone"
else
    git clone https://github.com/elejeune11/Mechanical-MNIST-Cahn-Hilliard.git
fi
cd ~/Mechanical-MNIST-Cahn-Hilliard

# -------------------------------------
# 6. Fix hardcoded BU path in FEA script
# -------------------------------------
echo "[6/7] Fixing hardcoded path in Equibiaxial_Hyperelastic.py..."
sed -i "s|folder_name =  '/projectnb2/lejlab2/Hiba/Equi_Hyper/Results'|folder_name = 'Results'|" Equibiaxial_Hyperelastic.py
# Also handle single space variant
sed -i "s|folder_name = '/projectnb2/lejlab2/Hiba/Equi_Hyper/Results'|folder_name = 'Results'|" Equibiaxial_Hyperelastic.py
echo "      Path fixed OK"

# -------------------------------------
# 7. Smoke test
# -------------------------------------
echo "[7/7] Running smoke test..."
python3 -c "
from fenics import *
import numpy, cv2, pygmsh, meshio, gmsh, h5py
print('      All imports OK')
print('      numpy:', numpy.__version__)
print('      meshio:', meshio.__version__)
print('      pygmsh:', pygmsh.__version__)
"

echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo "To activate the environment in a new terminal:"
echo "  source ~/fenics_env/bin/activate"
echo ""
echo "To preprocess an image and run FEA:"
echo "  cd ~/Mechanical-MNIST-Cahn-Hilliard"
echo "  python3 preprocess_image.py <your_image.png> my_pattern --param2 12 --min-dist 25"
echo "  python3 NumpyImageToGmsh.py"
echo "  cd mesh_files && python3 my_pattern.py && cd .."
echo "  python3 Equibiaxial_Hyperelastic.py mesh_files/my_pattern.xdmf"
echo "  # Results in: Results/my_pattern_*.txt"
