import numpy as np
import matplotlib.pyplot as plt

# Ensure Matplotlib uses a compatible backend for visualization
import matplotlib
matplotlib.use('TkAgg')  # Adjust if needed for GUI rendering

# Constants
T_COSMIC_BREATH = 3.108e12  # Total cosmic breath duration in years
NUM_PHASES = 26  # Cosmic alphabet phases

# Define entropy-aware function (adjusted to prevent zero scaling)
def entropy_density(n):
    return np.log(n + 2)  # Ensures no phase collapses to zero

# Define recursive quantum memory transfer
def quantum_memory(prev_phase_entropy, current_phase):
    return prev_phase_entropy * entropy_density(current_phase)

# Initialize phase lifespans
lifespans = np.linspace(T_COSMIC_BREATH / NUM_PHASES, T_COSMIC_BREATH, NUM_PHASES)

# Apply entropy modifications
adjusted_lifespans = [lifespan * entropy_density(i) for i, lifespan in enumerate(lifespans)]

# Ensure first phase starts correctly (avoids zero lifespan issue)
forecasted_lifespans = [adjusted_lifespans[0]]
for i in range(1, NUM_PHASES):
    forecasted_lifespans.append(quantum_memory(forecasted_lifespans[i-1], i))

# Plot entropy stability across all phases
entropy_values = [entropy_density(i) for i in range(NUM_PHASES)]

plt.figure(figsize=(10, 5))
plt.plot(range(NUM_PHASES), entropy_values, marker='o', linestyle='-', color='blue')
plt.xlabel("Phase (A-Z)")
plt.ylabel("Entropy Density")
plt.title("Entropy Stability Across Cosmic Phases")
plt.grid()
plt.show()

# Display results
for i, lifespan in enumerate(forecasted_lifespans):
    print(f"Phase {chr(65+i)}: Lifespan = {lifespan:.2e} years")
