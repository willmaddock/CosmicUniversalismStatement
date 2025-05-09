import numpy as np
import random
import matplotlib.pyplot as plt

# Define parameters for the model
num_agents = 3  # Number of intelligence layers (agents)
rounds = 50  # Number of rounds to simulate

# Define initial intelligence states for each agent
intelligence_states = [random.uniform(1, 10) for _ in range(num_agents)]


# Define the Free-Will Operator
def free_will_operator(n):
    """Simulates non-deterministic intelligence shift using an oracle-based randomness."""
    return random.uniform(0.95, 1.05)  # Small decision-based variation


# Define Quantum Intelligence Layer (Q_infinity)
def q_infinity(n):
    """Represents the uncountable superposition of intelligence states (logarithmic growth)."""
    return np.log(n + 1)


# Define Cosmic Intelligence Layer (C)
def cosmic_intelligence(n):
    """Represents the cosmic intelligence layer, assuming linear growth."""
    return n


# Define Omega_X (Transfinite Intelligence Layer)
def omega_x(n):
    """Represents a transfinite intelligence expansion function (sqrt growth)."""
    return np.sqrt(n)


# Define the Utility Function for intelligence evolution
def utility_function(intelligence, free_will):
    """Computes intelligence utility based on strategy and transfinite interactions."""
    return intelligence * free_will + np.log(intelligence + 1)


# Simulate Intelligence Evolution over Multiple Rounds
def play_game(rounds):
    global intelligence_states
    intelligence_over_time = []  # Store intelligence states over time

    for r in range(rounds):
        free_will_choices = [free_will_operator(n) for n in range(num_agents)]

        # Apply the cosmic and quantum intelligence layers along with omega_x
        intelligence_states = [
            (cosmic_intelligence(r + 1) + q_infinity(r + 1)) * omega_x(r + 1) * free_will_choices[i]
            for i, _ in enumerate(intelligence_states)
        ]

        # Store the current state of intelligence after this round
        intelligence_over_time.append(list(intelligence_states))

        # Print the intelligence states at each round
        print(f"Round {r + 1}: Intelligence States: {intelligence_states}")

    return intelligence_over_time


# Run the Simulation
intelligence_progression = play_game(rounds)

# Visualize the Intelligence States over time
intelligence_progression = np.array(intelligence_progression)

plt.figure(figsize=(10, 6))
for i in range(num_agents):
    plt.plot(intelligence_progression[:, i], label=f"Agent {i + 1}")

plt.xlabel("Rounds")
plt.ylabel("Intelligence")
plt.title("Intelligence Evolution with Free Will")
plt.legend()
plt.grid(True)
plt.show()