import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Modify the State Expansion to reflect exponential state growth
def state_expansion(n, growth_type='exp', k=2, lam=0.1):
    """State expansion function S(n), adapted for exponential growth of states."""
    if growth_type == 'ztom':
        return 2 ** 1  # -ztom: 2^1 states
    elif growth_type == 'atom':
        return 2 ** 2  # Atom: 2^2 states
    elif growth_type == 'btom':
        return 2 ** 4  # Btom: 2^4 states
    elif growth_type == 'ctom':
        return 2 ** 16  # Ctom: 2^16 states
    elif growth_type == 'dtom':
        # Using logarithmic scaling to prevent overflow (log of the large number)
        return 65536 * np.log(2)  # Logarithmic scaling for 2^65536
    else:
        return 1  # Default for other cases

# Transfinite Intelligence function remains the same
def transfinite_intelligence(n):
    """Quantum intelligence layers Q_∞(n)^{Ω_X(n)} with stepwise growth."""
    if n <= np.e:  # Base level
        return 2 ** n
    elif n <= 10:  # Omega-level transition
        return n ** np.e
    else:  # Beyond omega
        return np.e ** n

# Ordinal scaling to match the exponential growth of states
def ordinal_scaling(n):
    """Ordinal function Ω_X(n)."""
    return np.log(n + 1) + 1

# Free-Will Interaction function remains the same, adding stochastic elements
def free_will_interaction(n, history, noise_scale=0.1):
    """Adaptive Free-Will interaction function F(n)."""
    base_influence = 1 + np.random.normal(0, noise_scale)
    momentum = history.get(n, 1.0)  # Track influence momentum
    return base_influence * momentum

# Compute Theta over a large number of states (n_max)
def compute_theta(n_max=100, growth_type='ztom', k=2, lam=0.1):
    """Computes Θ(E, I, F) sum up to n_max with adaptive free will."""
    theta_values = []
    history = {}

    print(f"Computing Theta for {growth_type} state growth...\n")

    for n in range(1, n_max + 1):
        s_n = state_expansion(n, growth_type, k, lam)
        q_n = transfinite_intelligence(n) ** ordinal_scaling(n)
        f_n = free_will_interaction(n, history)
        theta_n = s_n * q_n * f_n
        theta_values.append(theta_n)

        # Print intermediate results for each step
        print(f"Step {n}: S(n) = {s_n:.2e}, Q_∞(n) = {q_n:.2e}, F(n) = {f_n:.2e}, Θ(n) = {theta_n:.2e}")

        # Update free-will influence with some stochastic perturbation
        if np.random.rand() < 0.2:  # 20% chance of perturbation
            history[n] = history.get(n, 1.0) * 1.2

    cumulative_theta = np.cumsum(theta_values)
    print("\nCumulative Theta (Total Evolution):")
    print(cumulative_theta)

    return cumulative_theta  # Return cumulative sum to visualize the result

# Simulation parameters
n_max = 50
growth_type = 'dtom'  # Let's start with dtom (can switch to others)

# Compute results
theta_values = compute_theta(n_max, growth_type)

# 3D Visualization for the Evolution of Θ
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Define the states and intelligence layers for visualization
n_values = np.arange(1, n_max + 1)
intelligence_values = np.array([transfinite_intelligence(n) for n in n_values])
free_will_values = np.array([free_will_interaction(n, {}) for n in n_values])

ax.plot(n_values, intelligence_values, free_will_values, label=f'{growth_type} Evolution')
ax.set_xlabel('n (Evolution Steps)')
ax.set_ylabel('Intelligence Q_∞(n)')
ax.set_zlabel('Free Will Influence F(n)')
ax.set_title(f'3D Evolution of Θ with {growth_type} State Growth')
ax.legend()
plt.show()