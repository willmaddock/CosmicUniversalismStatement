from vpython import sphere, vector, color
import numpy as np


def generate_fractal_3d(x, y, z, depth, max_depth):
    if depth > max_depth:
        return []

    points = [vector(x, y, z)]
    scale_factor = 0.5  # Controls the size of recursive branches

    # Generate new recursive points
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)  # Hexagonal-like expansion
    for angle in angles:
        new_x = x + scale_factor * np.cos(angle)
        new_y = y + scale_factor * np.sin(angle)
        new_z = z + scale_factor * np.sin(angle)  # Add 3D movement
        points.extend(generate_fractal_3d(new_x, new_y, new_z, depth + 1, max_depth))

    return points


# Set up the 3D visualization
points = generate_fractal_3d(0, 0, 0, 0, 4)

# Create spheres at each fractal point
for p in points:
    sphere(pos=p, radius=0.02, color=color.cyan)