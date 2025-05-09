import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Tom Levels and Corresponding Quantum States
tom_levels = [
    "ztom", "ytom", "xtom", "wtom", "vtom", "utom", "ttom", "stom", "rtom", "qtom",
    "ptom", "otom", "ntom", "mtom", "ltom", "ktom", "jtom", "itom", "htom", "gtom",
    "ftom", "etom", "dtom", "ctom", "btom", "atom"
]

quantum_states = [
    2**(2**20), 2**(2**19), 2**(2**18), 2**(2**17), 2**(2**16),
    2**(2**15), 2**(2**14), 2**(2**13), 2**(2**12), 2**(2**11),
    2**(2**10), 2**(2**9), 2**(2**8), 2**(2**7), 2**(2**6),
    2**(2**5), 2**(2**4), 2**(2**3), 2**(2**2), 2**65536,
    2**16, 2**4, 65536, 16, 4, 2
]

time_values = [
    1, 2.704e-8, 2.704e-7, 2.704e-6, 2.704e-5, 0.0002704, 0.002704, 0.02704, 0.2704, 2.704,
    27.04, 4.506*60, 45.06*60, 7.51*3600, 3.1296*86400, 31.296*86400, 0.8547*31557600, 8.547*31557600, 85.47*31557600,
    427.35*31557600, 4273.5*31557600, 42735*31557600, 427350*31557600, 28e9*31557600, 280e9*31557600, 28e12*31557600
]

# Reverse the lists for visualization (so atom starts first)
tom_levels.reverse()
quantum_states.reverse()
time_values.reverse()

# Convert Tom Levels to Numeric Indices for 3D plot
x_values = np.arange(len(tom_levels))

# Create Figure
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Set initial plot properties
ax.set_xlabel("Tom Level")
ax.set_ylabel("Quantum States (log scale)")
ax.set_zlabel("Time (log scale)")
ax.set_yscale("log")
ax.set_zscale("log")

# Color mapping for smooth transitions
color_map = plt.get_cmap("plasma")
colors = [color_map(i / len(tom_levels)) for i in range(len(tom_levels))]

# Scatter plot initialization
sc = ax.scatter([], [], [], c=[], s=50)

# Animation function
def update(frame):
    """ Updates the scatter plot animation in 3D """
    ax.clear()

    # Set labels and scales
    ax.set_xlabel("Tom Level")
    ax.set_ylabel("Quantum States (log scale)")
    ax.set_zlabel("Time (log scale)")
    ax.set_yscale("log")
    ax.set_zscale("log")

    # Update scatter plot with current frame data
    ax.scatter(x_values[:frame], quantum_states[:frame], time_values[:frame], c=colors[:frame], s=50)

    # Rotate View for Dynamic Effect
    ax.view_init(elev=20, azim=frame * 4)

    # Set tick labels
    ax.set_xticks(x_values[:frame])
    ax.set_xticklabels(tom_levels[:frame], rotation=90)

# Run animation
ani = animation.FuncAnimation(fig, update, frames=len(tom_levels), interval=500, repeat=True)

plt.show()