#!/usr/bin/env python3
"""
Generate publication-quality benchmark charts for AMD Instinct MI210 SGEMM
comparing custom assembly kernels against rocBLAS across transpose modes.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set high-quality styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#d0d0d0"
plt.rcParams["axes.linewidth"] = 0.9

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.4), dpi=300)

# Color Palette
color_rocblas = "#e74c3c"      # Crimson / Red
color_ours = "#1f77b4"         # Deep Tech Blue
color_unopt = "#95a5a6"        # Cool Gray
color_accent = "#27ae60"       # Emerald Green

# =========================================================================
# Subplot 1: 4K Matrix Evaluation (4096 x 4096 x 4096)
# =========================================================================
categories_4k = ["NN", "NT", "TN", "TT"]
rocblas_4k = [33.60, 33.60, 33.60, 33.60]
ours_4k = [34.54, 34.74, 33.85, 34.06]

x = np.arange(len(categories_4k))
width = 0.35

rects1 = ax1.bar(x - width / 2, rocblas_4k, width, label="rocBLAS Baseline (~33.6 TFLOPS)", color=color_rocblas, alpha=0.9, edgecolor="none", zorder=3)
rects2 = ax1.bar(x + width / 2, ours_4k, width, label="Custom GCN Assembly (sgl_vs2)", color=color_ours, alpha=0.95, edgecolor="none", zorder=3)

# Data labels
for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f"{h:.1f}",
                 xy=(rect.get_x() + rect.get_width() / 2, h),
                 xytext=(0, 4), textcoords="offset points",
                 ha="center", va="bottom", fontsize=10, color="#555555")

for i, rect in enumerate(rects2):
    h = rect.get_height()
    diff = ((h / rocblas_4k[i]) - 1.0) * 100.0
    ax1.annotate(f"{h:.2f}\n(+{diff:.1f}%)",
                 xy=(rect.get_x() + rect.get_width() / 2, h),
                 xytext=(0, 4), textcoords="offset points",
                 ha="center", va="bottom", fontsize=10, fontweight="bold", color="#0b486b")

# Theoretical Peak line
ax1.axhline(45.3, color="#7f8c8d", linestyle="--", linewidth=1.2, zorder=2, label="MI210 Theoretical MFMA Peak (~45.3 TFLOPS)")

ax1.set_title("4K SGEMM (4096×4096×4096)\nMacroTile: b256x128x16 (512 WGs / 104 CUs)", fontsize=13, fontweight="bold", pad=12)
ax1.set_ylabel("Throughput (TFLOPS)", fontsize=11, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels([f"{mode}\n(Col-Major)" for mode in categories_4k], fontsize=11)
ax1.set_ylim(0, 55)
ax1.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=9.5)
ax1.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)

# =========================================================================
# Subplot 2: 2K Matrix Evaluation (2048 x 2048 x 2048)
# =========================================================================
modes_2k = [
    "NN\n(b256x128)\nTail Stall",
    "NN\n(b128x64)\nWGM=8",
    "NT\n(b128x64)\nWGM=8",
    "TN\n(b128x64)\nWGM=8",
    "TT\n(b128x64)\nWGM=8"
]
tflops_2k = [22.68, 32.52, 31.58, 29.69, 30.51]
colors_2k = [color_unopt, color_ours, color_ours, color_ours, color_ours]

bars2 = ax2.bar(modes_2k, tflops_2k, width=0.55, color=colors_2k, alpha=0.95, zorder=3)

# Reference lines
ax2.axhline(35.0, color=color_rocblas, linestyle="-.", linewidth=1.5, zorder=2, label="rocBLAS 2K Reference (~35.0 TFLOPS)")
ax2.axhline(45.3, color="#7f8c8d", linestyle="--", linewidth=1.2, zorder=2, label="Theoretical Peak (~45.3 TFLOPS)")

for i, bar in enumerate(bars2):
    h = bar.get_height()
    pct = (h / 35.0) * 100.0
    if i == 0:
        label_text = f"{h:.2f}\n(64.8% ref)\n[80 CUs Idle]"
        color_text = "#666666"
    else:
        label_text = f"{h:.2f}\n({pct:.1f}% ref)"
        color_text = "#0b486b"
    ax2.annotate(label_text,
                 xy=(bar.get_x() + bar.get_width() / 2, h),
                 xytext=(0, 4), textcoords="offset points",
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=color_text)

# Arrow annotation showing speedup from tailoring tile & WGM
ax2.annotate("+43.4% Occupancy & WGM Gain",
             xy=(1.0, 36.8), xytext=(0.1, 47.0),
             arrowprops=dict(facecolor=color_accent, edgecolor=color_accent, shrink=0.08, width=1.6, headwidth=6.5),
             fontsize=10, fontweight="bold", color="#196f3d",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#eafaf1", edgecolor=color_accent, lw=1.2))

ax2.set_title("2K SGEMM (2048×2048×2048)\nTail Wave Recovery via Tile Scaling (b128x64) & WGM=8", fontsize=13, fontweight="bold", pad=12)
ax2.set_ylabel("Throughput (TFLOPS)", fontsize=11, fontweight="bold")
ax2.set_ylim(0, 55)
ax2.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=9.5)
ax2.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)

# Main Super Title
plt.suptitle("AMD Instinct MI210 (CDNA 2 / gfx90a, 104 CUs): SGEMM Benchmark vs rocBLAS",
             fontsize=15, fontweight="bold", y=0.99)

plt.tight_layout()

# Save PNG and SVG
output_png = "docs/images/mi210_sgemm_benchmark.png"
output_svg = "docs/images/mi210_sgemm_benchmark.svg"
plt.savefig(output_png, bbox_inches="tight")
plt.savefig(output_svg, bbox_inches="tight")
print(f"Generated {output_png} and {output_svg}")
