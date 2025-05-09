import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from scipy.linalg import expm

# Set up the backend for non-interactive use (adjust as needed)
matplotlib.use('TkAgg')  # or try 'Agg' if running in a script outside Jupyter

# 1. Expand the number of quantum states (dimensionality)
states_256d = np.eye(256)

# 2. Define a more realistic cosmological Hamiltonian
# Placeholder for a more realistic cosmological Hamiltonian
H_cosmic = np.random.randn(256, 256)
H_cosmic = (H_cosmic + H_cosmic.T) / 2  # Ensure it's Hermitian for quantum evolution

# 3. Define evolution time t and cosmological parameters
t = 0.1  # Normalized time for evolution
anisotropy_factors = np.array([1.021, 1.000, 0.979])  # Example based on real data

# 4. Compute the unitary evolution matrix
U = expm(-1j * H_cosmic * t)

# 5. Apply unitary transformation to quantum states
evolved_states_256d = U @ states_256d

# 6. Apply PCA for dimensionality reduction to 3D
pca = PCA(n_components=3)
states_3d = pca.fit_transform(evolved_states_256d.real)  # Use real part for visualization

# 7. Calculate the semi-axes of the ellipsoid
U, S, Vt = np.linalg.svd(states_3d)
a, b, c = S * anisotropy_factors  # Adjust for anisotropy factors

# 8. Visualize the cosmic structure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Generate ellipsoid surface with slight curvature
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = a * np.outer(np.cos(u), np.sin(v))
y = b * np.outer(np.sin(u), np.sin(v))
z = c * np.outer(np.ones_like(u), np.cos(v))

# Plot surface
ax.plot_surface(x, y, z, color='b', alpha=0.5)
ax.set_box_aspect([a, b, c])  # Adjust axes to reflect anisotropy
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Quantum Evolved 3D Projection of Current CTOM')
plt.show()