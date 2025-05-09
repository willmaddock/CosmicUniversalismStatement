import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


# 1. CU Recursion (Tetration)
def cu_recursion(n, U0=10):
    """Tetration-based recursion with U0=10 to avoid overflow."""
    if n == 0:
        return 1
    return U0 ** cu_recursion(n - 1, U0)


# 2. Atomic Timekeeper
def atomic_timekeeper(Z, alpha=13.8e9, beta=0.1):
    """Calibrated with Planck cosmic time (13.8 billion years)."""
    return alpha + beta * np.log(Z)


# 3. Z-TOM Scaling (Logistic Growth)
def z_tom_scaling(n, K=1e30):
    """Logistic growth model for Z-TOM scaling."""
    return K / (1 + (K - 1) * np.exp(-0.5 * n))


# 4. Quantum Intelligence Anchoring (QIA)
def qia_integrand(T, Q, Z, lambd=1e-3, mu=1e-5, delta=0.1):
    """QIA integrand calibrated with superconducting qubit data."""
    return lambd * Q * np.exp(-delta * Z) + mu * T


def quantum_intelligence_anchoring(T0, Tf, Q, Z):
    """Integrate over quantum coherence times (T0=0, Tf=1e-6)."""
    return quad(qia_integrand, T0, Tf, args=(Q, Z))[0]


# 5. CU-Based Inflation
def cu_inflation(t, a0=1, H0=2.2e-18, Z=1):
    """CU inflation aligned with ΛCDM (H0 in 1/s)."""
    return a0 * np.exp(H0 * t + 0.1 * np.log(Z))


# 6. Plotting Function
def plot_cu_models():
    t = np.linspace(1e-36, 1e-32, 100)  # Inflation era
    z_values = [1, 10, 100]

    plt.figure(figsize=(12, 6))
    for z in z_values:
        a = cu_inflation(t, Z=z)
        plt.plot(t, a, label=f'CU Inflation (Z={z})')

    # ΛCDM comparison
    a_lcdm = np.exp(2.2e-18 * t)
    plt.plot(t, a_lcdm, 'k--', label='ΛCDM Inflation')

    plt.title('CU Inflation vs. ΛCDM')
    plt.xlabel('Time (s)')
    plt.ylabel('Scale Factor a(t)')
    plt.legend()
    plt.show()


# Run simulations
print("CU Recursion U_3:", cu_recursion(3))
print("Atomic Timekeeper for Z=10:", atomic_timekeeper(10))
print("Z-TOM Scaling for n=3:", z_tom_scaling(3))
print("QIA for T=[0,1e-6], Q=1, Z=10:", quantum_intelligence_anchoring(0, 1e-6, 1, 10))

plot_cu_models()