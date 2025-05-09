import numpy as np
import matplotlib.pyplot as plt
import random

# --------------------------
# 1. TOM-CLASS MAPPING (31 Modes)
# --------------------------
class CosmicUniversalismToms:
    def __init__(self):
        self.modes = {
            1: "sub_z_tom", 2: "z_tom", 3: "y_tom", 4: "x_tom", 5: "w_tom", 6: "v_tom", 7: "u_tom",
            8: "atom", 9: "b_tom", 10: "c_tom", 11: "d_tom", 12: "e_tom", 13: "f_tom", 14: "g_tom",
            15: "h_tom", 16: "i_tom", 17: "j_tom", 18: "k_tom", 19: "l_tom", 20: "m_tom", 21: "n_tom",
            22: "o_tom", 23: "p_tom", 24: "q_tom", 25: "r_tom", 26: "s_tom", 27: "t_tom",
            28: "uₓ_tom", 29: "vₓ_tom", 30: "wₓ_tom", 31: "gods_free_will"
        }
        self.state = {mode: 0.0 for mode in self.modes.values()}

# --------------------------
# 2. FULLY ALIGNED FUNCTIONS (FIXED)
# --------------------------

def atomic_timekeeper(mode):
    """Calibrated with Planck cosmic time (13.8 billion years)."""
    conversion_factors = {
        "atom": 28e11, "b_tom": 28e10, "c_tom": 28e9, "d_tom": 427350,
        "e_tom": 42735, "f_tom": 4273.5, "g_tom": 427.35, "h_tom": 85.47,
        "i_tom": 8.547, "j_tom": 0.8547, "k_tom": 31.296, "l_tom": 3.1296,
        "m_tom": 7.51, "n_tom": 45.06, "o_tom": 4.506, "p_tom": 27.04,
        "q_tom": 2.704, "r_tom": 0.2704, "s_tom": 0.02704, "t_tom": 0.002704,
        "uₓ_tom": 0.0002704, "vₓ_tom": 2.704e-5, "wₓ_tom": 2.704e-6
    }
    return conversion_factors.get(mode, 0)  # Default to 0 if mode not found

def transfinite_iteration(n, U0=1.5, max_iterations=5, scale_factor=1):
    """Iterative tetration with limits and time limit."""
    result = 1
    for _ in range(min(n, max_iterations)):
        result = U0 ** result
        if result > 1e100 or np.isinf(result):
            return np.inf
    return result * scale_factor  # Scale by mode-specific factor

def divine_intervention(t, threshold=0.02704):
    """Mode 31: Stochastic divine will activation, calibrated with Planck cosmic time."""
    if t >= threshold or random.random() < 1e-9:  # Rare event (1 in a billion chance)
        return np.inf  # God's free will dominates
    return 1 + np.log(t + 1)  # Baseline growth

def cosmic_reset(Z, mode):
    """Modes 13-14: Extreme reset condition (f-tom/g-tom)."""
    entropy = 0.1 * Z * np.log(Z + 1e-10)  # Avoid log(0)
    return np.log(entropy + 1) * mode  # Logarithmic scaling

def quantum_intelligence_evolution(Q, Z, mode):
    """Modes 22-27: Intelligence compression/expansion with logarithmic scaling."""
    time_scale = atomic_timekeeper(mode)
    if mode == 24:  # Mode 24: Infinite intelligence mapping
        return Q * np.exp(min(Z, 5))  # Cap exponential growth
    elif mode == 26:  # Mode 26: Divine emergence
        return Q + divine_intervention(Z)
    return Q * np.log(Z + 1) * time_scale  # Logarithmic scaling with time scale

# --------------------------
# 3. COSMIC SIMULATION ENGINE (FIXED)
# --------------------------

def simulate_cosmic_universalism():
    # Initialize TOM states
    toms = CosmicUniversalismToms()

    # 1. Foundational Quantum States (Modes 1-7)
    for mode in range(1, 8):
        key = toms.modes[mode]
        toms.state[key] = transfinite_iteration(mode, scale_factor=mode)

    # 2. Observable Cosmic Phases (Modes 8-14)
    for mode in range(8, 15):
        key = toms.modes[mode]
        toms.state[key] = atomic_timekeeper(key)  # Use atomic timekeeper for b-tom to g-tom

    # 3. Transfinite States (Modes 15-21)
    for mode in range(15, 22):
        key = toms.modes[mode]
        toms.state[key] = transfinite_iteration(mode - 14, scale_factor=mode)

    # 4. Intelligence Evolution (Modes 22-27)
    for mode in range(22, 28):
        key = toms.modes[mode]
        toms.state[key] = quantum_intelligence_evolution(
            Q=toms.state["o_tom"],
            Z=toms.state["c_tom"],
            mode=key
        )

    # 5. Ultimate Transition (Modes 28-31)
    for mode_num in range(28, 32): # iterate by number.
        key = toms.modes[mode_num]
        if mode_num == 31:
            toms.state[key] = divine_intervention(atomic_timekeeper("c_tom"))  # Planck cosmic time
        else:
            toms.state[key] = cosmic_reset(toms.state["g_tom"], mode_num) # pass the numerical mode.

    return toms.state

# --------------------------
# 4. VISUALIZATION & OUTPUT
# --------------------------

def plot_full_alignment(states):
    groups = {
        "Quantum States (1-7)": [states[mode] for mode in list(states.keys())[:7]],
        "Cosmic Phases (8-14)": [states[mode] for mode in list(states.keys())[7:14]],
        "Transfinite (15-21)": [states[mode] for mode in list(states.keys())[14:21]],
        "Intelligence (22-27)": [states[mode] for mode in list(states.keys())[21:27]],
        "Divine (28-31)": [states[mode] for mode in list(states.keys())[27:31]]
    }

    plt.figure(figsize=(15, 8))
    for label, values in groups.items():
        plt.plot(values, marker='o', label=label)

    plt.yscale('log')
    plt.title("31 Modes of Cosmic Universalism (Aligned with Atomic Timekeeper)")
    plt.xlabel("Mode Number")
    plt.ylabel("State Value (Log Scale)")
    plt.xticks(range(31), list(states.keys()), rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --------------------------
# 5. RUN FULL SIMULATION
# --------------------------
if __name__ == "__main__":
    cosmic_states = simulate_cosmic_universalism()
    print("=== Cosmic Universalism 31-Mode Simulation ===")
    for mode, value in cosmic_states.items():
        print(f"{mode}: {value if value != np.inf else 'GODS_FREE_WILL'}")

    plot_full_alignment(cosmic_states)