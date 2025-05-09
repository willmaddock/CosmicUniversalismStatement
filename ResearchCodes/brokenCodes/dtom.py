import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
n_dim = 65536  # dtom: 2^16 = 65,536 dimensions
n_compress = 3  # Compress to 3D

# Initialize compression matrix (3x65,536)
np.random.seed(42)  # For reproducibility
A = np.random.randn(n_compress, n_dim)

# Iteratively compress each basis vector and store the result
states_3d = np.zeros((n_compress, n_dim))
for i in range(n_dim):
    basis_vector = np.zeros(n_dim)
    basis_vector[i] = 1.0  # Sparse representation of the i-th basis vector
    states_3d[:, i] = A @ basis_vector  # Project into 3D

# Perform SVD on the compressed states to get ellipsoid axes
U, S, Vt = np.linalg.svd(states_3d, full_matrices=False)
a, b, c = S

# Generate ellipsoid surface
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = a * np.outer(np.cos(u), np.sin(v))
y = b * np.outer(np.sin(u), np.sin(v))
z = c * np.outer(np.ones_like(u), np.cos(v))

# Plot the ellipsoid
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='purple', alpha=0.6, edgecolor='black')
ax.set_box_aspect([a, b, c])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Symbolic 3D Ellipsoid of DTOM (2^65,536)')
plt.show()