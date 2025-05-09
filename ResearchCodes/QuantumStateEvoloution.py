import matplotlib
# Use non-interactive backend for matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

# Example function to plot quantum state evolution
def quantum_state_evolution():
    # Time variable (in arbitrary units)
    t = np.linspace(0, 10, 100)

    # Example of a quantum state evolution (simple sinusoidal wave)
    # You can replace this with your own function to represent quantum state changes
    state = np.sin(t)

    # Plotting the evolution of the quantum state
    plt.figure(figsize=(10, 6))
    plt.plot(t, state, label='Quantum State Evolution')
    plt.title('Quantum State Evolution Over Time')
    plt.xlabel('Time (t)')
    plt.ylabel('State Amplitude')
    plt.legend()
    plt.grid(True)

    # Save the plot as a .png file (non-interactive mode)
    plt.savefig('quantum_state_evolution.png')
    print("Plot saved as 'quantum_state_evolution.png'")

# Run the quantum state evolution function
quantum_state_evolution()