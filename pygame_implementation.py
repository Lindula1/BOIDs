import pygame
from boid_model import Population

pygame.init()

WIDTH, HEIGHT = 300, 300
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("BOID Example")

clock = pygame.time.Clock()

pixels = pygame.surfarray.pixels3d(screen)

running = True
frame = 0

boid_size = 2

border_adjust = 2
border = (border_adjust, int(HEIGHT-border_adjust), border_adjust, int(WIDTH-border_adjust))

test_pop = Population(border, 100, frame_rate=1 / 1, fixed_seed=False, safe_distance=5,
                      turn_factor=0.4, separation=0.05, alignment=0.05, visible_distance=30, max_speed=4, min_speed=2,
                      cohesion=0.003, rebound_factor=0.9)
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
        pixels[pos[0]:pos[0]+boid_size, pos[1]:pos[1]+boid_size] = [255,255,255]

    pygame.display.flip()

    mouse_x, mouse_y = pygame.mouse.get_pos()

    test_pop.next_frame(mouse_x, mouse_y, 0.05)

    frame += 1
    clock.tick(120)

pygame.quit()