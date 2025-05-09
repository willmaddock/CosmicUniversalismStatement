import matplotlib.pyplot as plt
import numpy as np

# Define TOM levels and their relative positions
levels = ["-ZTOM", "-YTOM", "-XTOM", "-BTOM", "ATOM", "BTOM", "XTOM", "YTOM", "ZTOM", "ZTOM+1"]
positions = np.logspace(-12, 1, len(levels))  # Log scale positioning

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xscale("log")
ax.set_xlim(min(positions) / 2, max(positions) * 2)
ax.set_ylim(-1, 1)
ax.set_yticks([])
ax.set_xticks(positions)
ax.set_xticklabels(levels, rotation=45, fontsize=10)
ax.set_title("Cosmic Universalism TOM Scale")
ax.grid(True, which="both", linestyle="--", linewidth=0.5)

# Scatter plot for TOM levels
ax.scatter(positions, [0] * len(levels), color="blue", label="TOM Levels", s=100)

# Display interactive plot
plt.show()
