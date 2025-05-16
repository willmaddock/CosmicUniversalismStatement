import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend, works well for saving figures
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate a 3D box for the Sub-YTOM visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set up the box limits (just for visual reference)
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
ax.set_zlim([0, 10])

# Generate fluctuating energy pockets (bubbles) inside the box
num_bubbles = 20
bubble_size = np.random.uniform(0.5, 1.5, num_bubbles)
bubble_positions = np.random.rand(num_bubbles, 3) * 10  # Random positions within the box

# Create the "energy bubbles" (spheres or ellipsoids)
for i in range(num_bubbles):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = bubble_size[i] * np.outer(np.cos(u), np.sin(v)) + bubble_positions[i, 0]
    y = bubble_size[i] * np.outer(np.sin(u), np.sin(v)) + bubble_positions[i, 1]
    z = bubble_size[i] * np.outer(np.ones_like(u), np.cos(v)) + bubble_positions[i, 2]

    # Add transparency and color for the energy bubble effect
    ax.plot_surface(x, y, z, color=np.random.rand(3, ), alpha=0.5)

# Set up labels for visualization
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.title('Sub-YTOM: Quantum Reset Spark in a Box (Energy Fluctuations)')

# Save the figure instead of displaying it
# Replace the path with an actual file path on your system
plt.savefig('/Users/<your_local>/Desktop/quantum_ctom_visualization.png')  # Replace with your file path

