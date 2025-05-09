import pygame
import random

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()


# Fractal particle system setup
class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = random.randint(2, 5)
        self.color = (0, 255, 255)  # Cyan

    def update(self):
        self.x += random.uniform(-2, 2)
        self.y += random.uniform(-2, 2)

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.size)


particles = [Particle(400, 300) for _ in range(100)]  # Create initial particles

# Main loop
running = True
while running:
    screen.fill((0, 0, 0))  # Clear screen

    for particle in particles:
        particle.update()
        particle.draw(screen)

    pygame.display.flip()  # Update the screen
    clock.tick(60)  # Control frame rate

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()