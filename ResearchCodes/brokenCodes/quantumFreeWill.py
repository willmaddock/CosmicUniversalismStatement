import numpy as np
import matplotlib.pyplot as plt
import random

# Define possible decisions and their probabilities
decisions = ['Option 1', 'Option 2', 'Option 3']
probabilities = [0.3, 0.4, 0.3]  # Probabilities that represent the choices

# Function to simulate free will (quantum collapse)
def simulate_free_will(probabilities):
    return random.choices(decisions, weights=probabilities, k=1)[0]

# Simulate multiple decisions
num_simulations = 1000
decisions_made = [simulate_free_will(probabilities) for _ in range(num_simulations)]

# Count occurrences of each decision
decision_counts = {decision: decisions_made.count(decision) for decision in decisions}

# Plot the evolution of decisions (how many times each choice was made)
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(decision_counts.keys(), decision_counts.values(), color=['red', 'green', 'blue'])

# Labels and title
ax.set_xlabel('Decision')
ax.set_ylabel('Number of Times Chosen')
ax.set_title('Simulating Free Will: Decision Distribution')
plt.show()

# Show the final decision
final_decision = simulate_free_will(probabilities)
print(f"Final decision made by the system: {final_decision}")