import numpy as np
import matplotlib.pyplot as plt

def state_expansion(n, growth_type='exp', k=2, lam=0.1):
    """State expansion function S(n)."""
    if growth_type == 'poly':
        return n**k
    elif growth_type == 'exp':
        return np.exp(lam * n)
    else:
        return 1

def transfinite_intelligence(n, q_base=2):
    """Quantum intelligence layers Q_∞(n)^{Ω_X(n)}."""
    return q_base ** n

def ordinal_scaling(n):
    """Ordinal function Ω_X(n)."""
    return np.log(n + 1) + 1  # Simulating slow-growing ordinal scaling

def free_will_interaction(n, noise_scale=0.1):
    """Free-will interaction function F(n)."""
    return 1 + np.random.normal(0, noise_scale)

def compute_theta(n_max=100, growth_type='exp', k=2, lam=0.1, q_base=2, noise_scale=0.1):
    """Computes Θ(E, I, F) sum up to n_max."""
    theta_values = []
    for n in range(1, n_max + 1):
        s_n = state_expansion(n, growth_type, k, lam)
        q_n = transfinite_intelligence(n, q_base) ** ordinal_scaling(n)
        f_n = free_will_interaction(n, noise_scale)
        theta_n = s_n * q_n * f_n
        theta_values.append(theta_n)
    return np.cumsum(theta_values)  # Cumulative sum to approximate limit

# Simulation parameters
n_max = 100

theta_values = compute_theta(n_max)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(range(1, n_max + 1), theta_values, label=r'$\Theta(E, I, F)$')
plt.xlabel('n (Evolution Steps)')
plt.ylabel('Θ Value')
plt.title('Evolution of Θ in the CU Master Equation')
plt.legend()
plt.grid()
plt.show()
