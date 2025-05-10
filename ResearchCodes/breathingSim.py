import matplotlib

matplotlib.use('TkAgg')  # Ensures Matplotlib works correctly with PyCharm or other GUIs

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

# Define cosmic time scaling parameters
ZTOM_EXPANSION_FACTOR = 2.8  # Trillion-year scaling for expansion
SUB_ZTOM_COMPRESSION_FACTOR = 0.308  # Trillion-year scaling for compression

# Generate time steps for simulation
time_steps = np.linspace(0, 3.108, 1000)  # Cosmic breath duration in trillion years


# Define expansion and compression functions
def smooth_expansion_curve(t):
    """Sigmoid-based transition for realistic cosmic stretching."""
    return ZTOM_EXPANSION_FACTOR / (1 + np.exp(-t))


def accelerated_compression_curve(t):
    """Exponential compression acceleration for sub-ZTOM collapse."""
    return SUB_ZTOM_COMPRESSION_FACTOR * np.exp(-t ** 2)  # Faster than linear decay


# AI-driven entropy forecasting using Bayesian probability
def estimate_entropy(tensor):
    """Handles small tensor inputs correctly for entropy scaling."""
    if tensor.numel() == 1:  # If tensor has only one element
        return tensor.item()  # Return its value as entropy
    else:
        mean_value = tensor.mean()
        variance = tensor.var(unbiased=False)  # Fixes degrees of freedom warning
        return mean_value / (1 + torch.exp(-variance))  # Bayesian entropy scaling


# Apply entropy mapping to phase shifts
def entropy_phase_shift(t):
    """Adjust transition speed dynamically using AI-driven entropy adjustments."""
    tensor_data = torch.tensor([t], dtype=torch.float32)  # Ensure it's a tensor
    entropy_adjustment = estimate_entropy(tensor_data)
    return entropy_adjustment * t  # No need for `.item()`


# Set up figure for animation
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 3.108)
ax.set_ylim(0, ZTOM_EXPANSION_FACTOR)
ax.set_xlabel("Cosmic Time (Trillion Years)")
ax.set_ylabel("Quantum Scaling")
ax.set_title("CU Cosmic Breathing: AI-Adjusted Expansion & Compression")
ax.grid(True)

# Initialize animation lines
expansion_line, = ax.plot([], [], label="Smooth Expansion", color="blue", linestyle="dashed")
compression_line, = ax.plot([], [], label="Accelerated Compression", color="red", linestyle="solid")


# Animation function with entropy mapping
def animate(frame):
    current_time = time_steps[frame]
    entropy_adjusted_time = entropy_phase_shift(current_time)

    expansion_line.set_data(time_steps[:frame], smooth_expansion_curve(time_steps[:frame]))

    # Fix: Ensure compression curve returns a sequence
    compression_line.set_data(time_steps[:frame], [accelerated_compression_curve(t) for t in time_steps[:frame]])

    return expansion_line, compression_line


# Create animated cosmic breathing cycle
ani = animation.FuncAnimation(fig, animate, frames=len(time_steps), interval=10, blit=True)

plt.legend()
plt.show()
