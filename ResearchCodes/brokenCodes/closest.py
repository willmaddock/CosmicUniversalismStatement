import random
import numpy as np
import matplotlib.pyplot as plt


# Quantum Decision Theory: Model of intelligence in superposition
class QuantumDecisionEntity:
    def __init__(self, options, superposition_weight=None):
        # options: List of possible decisions
        # superposition_weight: List of weights (probabilities) for each decision option
        self.options = options
        self.superposition_weight = superposition_weight or [1 / len(options)] * len(options)
        self.state = None  # This will hold the collapsed state

    def collapse_wavefunction(self):
        # Simulate the collapse of the wavefunction into one of the options
        choice = np.random.choice(self.options, p=self.superposition_weight)
        self.state = choice
        return choice

    def apply_free_will(self):
        # Introduce randomness to simulate free-will influence
        perturbation = random.uniform(-0.1, 0.1)  # Small random shift
        altered_weights = [weight + perturbation for weight in self.superposition_weight]
        altered_weights = [max(0, w) for w in altered_weights]  # Ensure no negative weights
        total_weight = sum(altered_weights)
        self.superposition_weight = [w / total_weight for w in altered_weights]  # Normalize
        return self.superposition_weight

    def make_decision(self):
        # First, apply free will
        self.apply_free_will()

        # Then, collapse the wavefunction to make a decision
        decision = self.collapse_wavefunction()
        return decision


# Visualize decision space evolution
def visualize_decision_space(entity, num_iterations=100):
    choices = []
    for _ in range(num_iterations):
        decision = entity.make_decision()
        choices.append(decision)

    # Plotting decision choices
    plt.figure(figsize=(8, 6))
    plt.hist(choices, bins=len(entity.options), edgecolor='black', alpha=0.7)
    plt.title("Decision Choices Over Time (Free Will Simulation)")
    plt.xlabel("Decision Options")
    plt.ylabel("Frequency")
    plt.show()


# Example of Free Will Simulation
options = ["Option 1", "Option 2", "Option 3", "Option 4"]
entity = QuantumDecisionEntity(options)

# Visualize decision-making over multiple iterations
visualize_decision_space(entity, num_iterations=100)