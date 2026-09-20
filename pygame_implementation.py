import pygame
from boid_model import Population

pygame.init()

WIDTH, HEIGHT = 600, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("BOID Example")

clock = pygame.time.Clock()

pixels = pygame.surfarray.pixels3d(screen)

running = True
frame = 0

test_pop = Population((0, HEIGHT, 0, WIDTH), 100, frame_rate=1 / 25, fixed_seed=True, safe_distance=12, turn_factor=54, separation=0.05)
test_pop.load_boids()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # red = (frame * 2) % 256
    # green = (frame * 3) % 256
    # blue = (frame * 5) % 256
    screen.fill((0,0,0))
    boid_pos = test_pop.get_positions(row_major=True)

    for pos in boid_pos:
        pixels[pos[0]:pos[0]+6, pos[1]:pos[1]+6] = [255,255,255]

    pygame.display.flip()

    test_pop.next_frame()

    frame += 1
    clock.tick(60)

pygame.quit()