"""
visualize_results.py
=====================
Visualizes FEA results from the Mechanical MNIST pipeline:
  - Original RGB image (optional)
  - Input binary pattern
  - Displacement fields (ux, uy) at d=0.5
  - Displacement magnitude map
  - Strain energy vs displacement curve
  - Reaction forces vs displacement curve

Usage:
    python3 visualize_results.py <results_folder> <pattern_name> [binary_pattern.txt] [original_image.png]

Examples:
    python3 visualize_results.py Results carbon_fibre
    python3 visualize_results.py Results carbon_fibre input_patterns/carbon_fibre.txt
    python3 visualize_results.py Results carbon_fibre input_patterns/carbon_fibre.txt /mnt/c/Users/ac147266/Downloads/hackaton/my_image.png

Output:
    <pattern_name>_results.png

Requirements:
    pip install numpy matplotlib opencv-python
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import os
import cv2


def load_results(results_folder, name):
    def f(suffix):
        return os.path.join(results_folder, f"{name}{suffix}")
    ux = np.loadtxt(f("_pixel_disp_0.5_x.txt"))
    uy = np.loadtxt(f("_pixel_disp_0.5_y.txt"))
    strain_energy = np.loadtxt(f("_strain_energy.txt"))
    rxn_force = np.loadtxt(f("_rxn_force.txt"))
    return ux, uy, strain_energy, rxn_force


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 visualize_results.py <results_folder> <pattern_name> [binary.txt] [original.png]")
        sys.exit(1)

    results_folder = sys.argv[1]
    pattern_name   = sys.argv[2]
    binary_path    = sys.argv[3] if len(sys.argv) > 3 else None
    rgb_path       = sys.argv[4] if len(sys.argv) > 4 else None

    print(f"Loading results for: {pattern_name}")
    ux, uy, strain_energy, rxn_force = load_results(results_folder, pattern_name)

    binary_img = np.loadtxt(binary_path) if binary_path and os.path.exists(binary_path) else None
    rgb_img    = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB) if rgb_path and os.path.exists(rgb_path) else None

    disp_vals = [0.0, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5]
    magnitude = np.sqrt(ux**2 + uy**2)

    # Count top panels
    top_panels = []
    if rgb_img    is not None: top_panels.append(('rgb',    rgb_img))
    if binary_img is not None: top_panels.append(('binary', binary_img))
    top_panels.append(('ux',  ux))
    top_panels.append(('uy',  uy))
    top_panels.append(('mag', magnitude))

    ncols = max(len(top_panels), 4)
    fig = plt.figure(figsize=(4.5 * ncols, 10))
    fig.suptitle(f"FEA Results — {pattern_name}", fontsize=16, fontweight='bold', y=1.01)
    gs = gridspec.GridSpec(2, ncols, figure=fig, hspace=0.4, wspace=0.35)

    # --- Top panels ---
    for col, (kind, data) in enumerate(top_panels):
        ax = fig.add_subplot(gs[0, col])
        if kind == 'rgb':
            ax.imshow(data)
            ax.set_title("Original image\n(RGB)", fontsize=11)
            ax.axis('off')
        elif kind == 'binary':
            ax.imshow(data, cmap='gray', origin='upper')
            ax.set_title("Input pattern\n(binary)", fontsize=11)
            ax.axis('off')
        elif kind == 'ux':
            vmax = np.abs(data).max()
            im = ax.imshow(data, cmap='RdBu_r', origin='upper', vmin=-vmax, vmax=vmax)
            ax.set_title("Displacement $u_x$ (d=0.5)", fontsize=11)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        elif kind == 'uy':
            vmax = np.abs(data).max()
            im = ax.imshow(data, cmap='RdBu_r', origin='upper', vmin=-vmax, vmax=vmax)
            ax.set_title("Displacement $u_y$ (d=0.5)", fontsize=11)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        elif kind == 'mag':
            im = ax.imshow(data, cmap='viridis', origin='upper')
            ax.set_title("Displacement magnitude\n$|u|$ (d=0.5)", fontsize=11)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- Strain energy curve ---
    ax = fig.add_subplot(gs[1, 0:ncols//2])
    ax.plot(disp_vals, strain_energy, 'b-o', linewidth=2, markersize=6)
    ax.set_xlabel("Applied displacement d", fontsize=11)
    ax.set_ylabel("Strain energy (ΔΨ)", fontsize=11)
    ax.set_title("Strain energy vs displacement", fontsize=11)
    ax.grid(True, alpha=0.3)
    for i, (x, y) in enumerate(zip(disp_vals, strain_energy)):
        if i > 0:
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)

    # --- Reaction forces curve ---
    ax = fig.add_subplot(gs[1, ncols//2:])
    labels = ['F_left_x', 'F_right_x', 'F_top_y', 'F_bottom_y']
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.plot(disp_vals, rxn_force[:, i], 'o-', label=label,
                color=color, linewidth=2, markersize=5)
    ax.set_xlabel("Applied displacement d", fontsize=11)
    ax.set_ylabel("Reaction force", fontsize=11)
    ax.set_title("Reaction forces vs displacement", fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    out_path = f"{pattern_name}_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
