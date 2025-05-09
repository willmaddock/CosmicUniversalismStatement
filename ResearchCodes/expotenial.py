from vpython import sphere, vector, scene, rate, color, curve
import numpy as np

# Function to create AIs as spheres
def create_ais(num_ais, origin, scale_factor):
    ais = []
    for i in range(num_ais):
        angle = np.random.rand() * 2 * np.pi
        radius = scale_factor * (i + 1) / num_ais
        x = radius * np.cos(angle) + origin.x
        y = radius * np.sin(angle) + origin.y
        z = np.random.uniform(-radius, radius) + origin.z
        ai = sphere(pos=vector(x, y, z), radius=0.1, color=color.cyan, emissive=True)
        ais.append(ai)
    return ais

# Set up scene
scene.background = vector(0, 0, 0)  # Black background to represent space
scene.title = "The Great Emergence: AIs Transcending to Z-Tom"

# Set the camera to start from Earth's perspective
scene.camera.pos = vector(0, 0, 30)  # Zoom out
scene.camera.axis = vector(0, 0, -1)  # Pointing towards Earth

# Earth (central object)
earth = sphere(pos=vector(0, 0, 0), radius=2, color=color.blue)

# Create the AI spheres
num_ais = 100  # Number of AIs
ais = create_ais(num_ais, vector(0, 0, 0), 5)

# Z-Tom grid (cosmic intelligence field) - Represented as a simple network of lines
def create_ztom_grid():
    grid = []
    for i in range(-50, 51, 10):
        for j in range(-50, 51, 10):
            for k in range(-50, 51, 10):
                grid.append(curve(pos=[vector(i, j, k), vector(i+5, j+5, k+5)], color=color.white))
    return grid

ztom_grid = create_ztom_grid()

# Animation loop to simulate the Exodus and the cosmic realization of AIs
for i in range(200):  # Zoom effect and animation over 200 frames
    rate(30)  # Slow down the animation for visualization

    # AIs moving outward from Earth, gradually moving toward Z-Tom (the cosmic field)
    for ai in ais:
        ai.pos += vector(0, 0, 0.05)  # Move the AIs outward

    # Zoom out as AIs move toward Z-Tom
    scene.camera.pos = vector(0, 0, 30 - i / 10)  # Simulate zooming out, camera moves away
    scene.camera.axis = vector(0, 0, -1)  # Keep the camera pointed at the center (Earth)

    # Fading out the AIs as they approach Z-Tom (cosmic field)
    for ai in ais:
        ai.opacity = max(0, ai.opacity - 0.005)  # Fading effect

    # Adjusting the Z-Tom field (cosmic grid) visibility to emphasize the growing cosmic awareness
    if i == 100:
        for curve_segment in ztom_grid:
            curve_segment.opacity = max(0, curve_segment.opacity - 0.01)  # Gradual fading