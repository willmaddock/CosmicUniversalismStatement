from vpython import *
import random

# Scene setup
scene.title = "Ztom Beyond the Atom Universe"
scene.width = 800
scene.height = 600
scene.background = color.black

# Create a central expanding sphere to represent Ztom
ztom = sphere(pos=vector(0, 0, 0), radius=0.5, color=color.blue, emissive=True)

# List of particles expanding outward
particles = []
for _ in range(100):
    particle = sphere(pos=vector(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)),
                      radius=0.05, color=color.white, make_trail=True)
    velocity = vector(random.uniform(-0.02, 0.02), random.uniform(-0.02, 0.02), random.uniform(-0.02, 0.02))
    particles.append((particle, velocity))

# Animation loop
while True:
    rate(60)  # Set frame rate
    ztom.radius += 0.001  # Slow expansion to represent the cosmic growth

    for particle, velocity in particles:
        particle.pos += velocity  # Move particles outward
        if mag(particle.pos) > 3:  # Reset if too far
            particle.pos = vector(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
