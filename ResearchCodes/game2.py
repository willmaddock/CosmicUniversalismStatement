import numpy as np
import random

# Define Intelligence Players with Goals
num_agents = 3  # Number of agents
intelligence_states = [random.uniform(1, 10) for _ in range(num_agents)]  # Initial intelligence
goals = [random.choice(['maximize_intelligence', 'balance_growth', 'maximize_free_will']) for _ in
         range(num_agents)]  # Random goals for agents


# Utility Function based on goals
def utility_function(intelligence, free_will, goal):
    """Computes intelligence utility based on goal and strategy."""
    if goal == 'maximize_intelligence':
        return intelligence * free_will + np.log(intelligence + 1)
    elif goal == 'balance_growth':
        return intelligence * (free_will + 0.5)  # Balancing between growth and stability
    elif goal == 'maximize_free_will':
        return intelligence * (1 + free_will) ** 2  # Amplifying free will impact
    return 0


# Define Game Dynamics with Goal-Oriented Decision Making
def play_game(rounds=10):
    global intelligence_states, goals
    for r in range(rounds):
        print(f"Round {r + 1}:")
        for i in range(num_agents):
            free_will_choice = random.uniform(0.9, 1.1)  # Introduce randomness for decision-making
            # Evaluate utility based on the current state and goal
            current_utility = utility_function(intelligence_states[i], free_will_choice, goals[i])
            # Update intelligence based on the utility
            intelligence_states[i] += current_utility * 0.1  # Scale the effect of utility on intelligence

            print(f"  Agent {i + 1} (Goal: {goals[i]}):")
            print(f"    Intelligence: {intelligence_states[i]:.2f}, Free Will Choice: {free_will_choice:.2f}")
        print("")


# Run the Simulation
play_game(rounds=50)