from vpython import sphere, vector, color, scene
import numpy as np
import time

# Initialize the scene
scene.background = color.black


# 1. Create an atom qubit (superposition of 2 states)
def create_qubit(center, radius=0.3):
    # Two superposed states: |0⟩ and |1⟩ represented as rotating spheres
    state_0 = sphere(pos=vector(center.x + radius, center.y, center.z), radius=radius, color=color.green, opacity=0.6)
    state_1 = sphere(pos=vector(center.x - radius, center.y, center.z), radius=radius, color=color.blue, opacity=0.6)
    return state_0, state_1


# 2. Create b-tom with 16 superposed states (16 distinct particles)
def create_btom(center, count=16):
    particles = []
    angle_step = 2 * np.pi / count
    for i in range(count):
        angle = angle_step * i
        x = np.cos(angle) * 3  # Make the b-tom particles larger and further apart
        y = np.sin(angle) * 3
        z = np.random.uniform(-1, 1) * 2  # Slightly randomized positions in Z
        particles.append(sphere(pos=vector(center.x + x, center.y + y, center.z + z), radius=0.2, color=color.cyan))
    return particles


# 3. Create c-tom with 2^16 superposed states (for simplicity, representing as many particles as possible)
def create_ctom(center, count=2 ** 5):  # Limiting to 2^5 = 32 particles for performance
    particles = []
    for i in range(count):
        x = np.random.uniform(-8, 8)  # Spread particles out even more for the c-tom scale
        y = np.random.uniform(-8, 8)
        z = np.random.uniform(-8, 8)
        particles.append(sphere(pos=vector(center.x + x, center.y + y, center.z + z), radius=0.05, color=color.red))
    return particles


# Define the center point of the universe
center = vector(0, 0, 0)

# Create the components
qubit_states = create_qubit(center)
btom_states = create_btom(center)
ctom_states = create_ctom(center)

# Adjust scene properties
scene.autoscale = False
scene.center = center
scene.lights = []

# Animation Loop
time_step = 0.05  # Time step for each frame


# Rotate qubit states to represent superposition
def rotate_qubit(state_0, state_1, angle):
    state_0.pos = vector(np.cos(angle) * 0.3, 0, np.sin(angle) * 0.3)
    state_1.pos = vector(np.cos(angle + np.pi) * 0.3, 0, np.sin(angle + np.pi) * 0.3)


while True:
    # Rotate the qubit states to show superposition
    angle = time.time() * 0.5
    rotate_qubit(qubit_states[0], qubit_states[1], angle)

    # Update b-tom states to simulate interaction (slight movement)
    for particle in btom_states:
        particle.pos += vector(np.random.uniform(-0.02, 0.02),
                               np.random.uniform(-0.02, 0.02),
                               np.random.uniform(-0.02, 0.02))

    # Update c-tom states (simulating random particle distribution and interaction)
    for particle in ctom_states:
        particle.pos += vector(np.random.uniform(-0.05, 0.05),
                               np.random.uniform(-0.05, 0.05),
                               np.random.uniform(-0.05, 0.05))

    # Slow down the animation for a smoother experience
    time.sleep(time_step)