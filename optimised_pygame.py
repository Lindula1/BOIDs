import pygame
from optimised_boid_model import Population

pygame.init()

# WIDTH, HEIGHT = 1280, 720
WIDTH, HEIGHT = 480, 320
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("BOID Example")

clock = pygame.time.Clock()

pixels = pygame.surfarray.pixels3d(screen)

running = True
frame = 0

boid_size = 5

border_adjust = 2
border = (border_adjust, int(HEIGHT-border_adjust), border_adjust, int(WIDTH-border_adjust))

test_pop = Population(border, 50, 120, random_boid_natures=True, min_speed=2, max_speed=5)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # red = (frame * 2) % 256
    # green = (frame * 3) % 256
    # blue = (frame * 5) % 256
    screen.fill((0,0,0))
    boid_pos = test_pop.get_positions()

    for pos in boid_pos[:]:
        pixels[pos[0]:pos[0]+boid_size, pos[1]:pos[1]+boid_size] = [255,255,255]

    pygame.display.flip()

    # mouse_x, mouse_y = pygame.mouse.get_pos()

    test_pop.next_frame(0.5,0.5,0.05,boid_size + 2.0,20.0,0.2, 0.4)

    frame += 1
    clock.tick(120)

pygame.quit()