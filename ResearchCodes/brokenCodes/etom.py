import matplotlib.pyplot as plt
import numpy as np

def draw_fractal(ax, depth, x, y, size):
    """Recursively draw a fractal pattern."""
    if depth == 0:
        ax.add_patch(plt.Rectangle((x, y), size, size, color='blue'))
    else:
        size /= 2
        draw_fractal(ax, depth - 1, x, y, size)
        draw_fractal(ax, depth - 1, x + size, y, size)
        draw_fractal(ax, depth - 1, x, y + size, size)
        draw_fractal(ax, depth - 1, x + size, y + size, size)

# Plot fractal for gtom (2↑↑4)
fig, ax = plt.subplots(figsize=(6, 6))
draw_fractal(ax, depth=4, x=0, y=0, size=1)
ax.set_title("Fractal Representation of gtom (2↑↑4)")
ax.set_aspect('equal')
ax.axis('off')
plt.show()