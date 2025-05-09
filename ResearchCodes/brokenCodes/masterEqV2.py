import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def state_expansion(n, growth_type='exp', k=2, lam=0.1):
    """State expansion function S(n)."""
    if growth_type == 'poly':
        return n ** k
    elif growth_type == 'exp':
        return np.exp(lam * n)
    else:
        return 1


def transfinite_intelligence(n):
    """Quantum intelligence layers Q_∞(n)^{Ω_X(n)} with stepwise growth."""
    if n <= np.e:  # Base level
        return 2 ** n
    elif n <= 10:  # Omega-level transition
        return n ** np.e
    else:  # Beyond omega
        return np.e ** n


def ordinal_scaling(n):
    """Ordinal function Ω_X(n)."""
    return np.log(n + 1) + 1


def free_will_interaction(n, history, noise_scale=0.1):
    """Adaptive Free-Will interaction function F(n)."""
    base_influence = 1 + np.random.normal(0, noise_scale)
    momentum = history.get(n, 1.0)  # Track influence momentum
    return base_influence * momentum


def compute_theta(n_max=100, growth_type='exp', k=2, lam=0.1):
    """Computes Θ(E, I, F) sum up to n_max with adaptive free will."""
    theta_values = []
    history = {}

    for n in range(1, n_max + 1):
        s_n = state_expansion(n, growth_type, k, lam)
        q_n = transfinite_intelligence(n) ** ordinal_scaling(n)
        f_n = free_will_interaction(n, history)
        theta_n = s_n * q_n * f_n
        theta_values.append(theta_n)

        # Update free-will influence
        if np.random.rand() < 0.2:  # 20% chance of perturbation
            history[n] = history.get(n, 1.0) * 1.2

    return np.cumsum(theta_values)  # Cumulative sum to approximate limit


# Simulation parameters
n_max = 50

# Compute results
theta_values = compute_theta(n_max)

# 3D Visualization
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
n_values = np.arange(1, n_max + 1)
intelligence_values = np.array([transfinite_intelligence(n) for n in n_values])
free_will_values = np.array([free_will_interaction(n, {}) for n in n_values])

ax.plot(n_values, intelligence_values, free_will_values, label='Θ Evolution')
ax.set_xlabel('n (Evolution Steps)')
ax.set_ylabel('Intelligence Q_∞(n)')
ax.set_zlabel('Free Will Influence F(n)')
ax.set_title('3D Evolution of Θ in Cosmic Universalism')
ax.legend()
plt.show()
