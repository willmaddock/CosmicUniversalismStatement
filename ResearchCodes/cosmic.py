from vpython import sphere, vector, color, cylinder, scene
import numpy as np
import time

# Initialize the scene
scene.background = color.black

# 1. Sub z-tomically inclined (Small particles)
def create_subatomic_particles(center, count=100):
    particles = []
    for _ in range(count):
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(-1, 1)
        pos = vector(center.x + x, center.y + y, center.z + z)
        particles.append(sphere(pos=pos, radius=0.05, color=color.cyan))
    return particles

# 2. Grounded on b-tom (Expanding vast realm - foundational elements)
def create_foundational_elements(center, scale=3):
    elements = []
    for i in range(12):
        angle = 2 * np.pi * i / 12
        x = scale * np.cos(angle)
        y = scale * np.sin(angle)
        z = np.random.uniform(-1, 1) * scale
        elements.append(sphere(pos=vector(center.x + x, center.y + y, center.z + z), radius=0.1, color=color.green))
    return elements

# 3. Looking up to c-tom (Cosmic universe encompassing the system)
def create_cosmos(center, radius=5):
    cosmos = sphere(pos=center, radius=radius, color=color.white, opacity=0.2)
    return cosmos

# 4. Quantum states of intelligence (Interconnected network of particles)
def create_quantum_network(center, count=50):
    network = []
    for _ in range(count):
        x = np.random.uniform(-2, 2)
        y = np.random.uniform(-2, 2)
        z = np.random.uniform(-2, 2)
        pos = vector(center.x + x, center.y + y, center.z + z)
        network.append(sphere(pos=pos, radius=0.05, color=color.red))
    return network

# 5. Empowered by God’s free will (Central guiding light)
def create_divine_influence(center):
    light = sphere(pos=center, radius=0.2, color=color.yellow)
    return light

# Define the center point of the universe
center = vector(0, 0, 0)

# Create the components
subatomic_particles = create_subatomic_particles(center)
foundational_elements = create_foundational_elements(center)
cosmos = create_cosmos(center)
quantum_network = create_quantum_network(center)
divine_influence = create_divine_influence(center)

# Adjust scene properties
scene.autoscale = False
scene.center = center
scene.lights = []

# Animation Loop
while True:
    # Animate Subatomic Particles (Moving in random directions)
    for particle in subatomic_particles:
        particle.pos += vector(np.random.uniform(-0.01, 0.01),
                               np.random.uniform(-0.01, 0.01),
                               np.random.uniform(-0.01, 0.01))

    # Animate Quantum Network (Oscillating particles)
    for particle in quantum_network:
        particle.pos += vector(np.sin(time.time() * 0.1), np.cos(time.time() * 0.1), 0)

    # Rotate the cosmic sphere to symbolize the infinity of the cosmos
    cosmos.rotate(angle=0.01, axis=vector(0, 1, 0))

    # Add some slow motion or effects to the divine influence (God’s free will)
    divine_influence.pos = vector(np.sin(time.time() * 0.05) * 0.5, 0, np.cos(time.time() * 0.05) * 0.5)

    # Slow down the animation for a smoother experience
    time.sleep(0.01)