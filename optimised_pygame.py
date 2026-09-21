import pygame
from gpu_optimised import Population

pygame.init()

WIDTH, HEIGHT = 1280, 720
# WIDTH, HEIGHT = 480, 320
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("BOID Example")

clock = pygame.time.Clock()

pixels = pygame.surfarray.pixels3d(screen)

running = True
frame = 0

boid_size = 3

border_adjust = 2
border = (border_adjust, int(HEIGHT-border_adjust), border_adjust, int(WIDTH-border_adjust))

max_speed = 6

test_pop = Population(border, 50, 7000, random_boid_natures=True, min_speed=2, max_speed=max_speed)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # red = (frame * 2) % 256
    # green = (frame * 3) % 256
    # blue = (frame * 5) % 256
    screen.fill((255,255,255))
    boid_pos = test_pop.get_positions()
    boid_velocities = test_pop.get_velocities()

    for pos_id in range(boid_pos.shape[0]):
        left_range_1 = boid_pos[pos_id, 0]
        left_range_2 = boid_pos[pos_id, 0] + boid_size
        right_range_1 = boid_pos[pos_id, 1]
        right_range_2 = boid_pos[pos_id, 1] + boid_size
        pixels[left_range_1:left_range_2, right_range_1:right_range_2] = [255*(sum(boid_velocities[pos_id]) / max_speed),200,200]

    pygame.display.flip()

    mouse_x, mouse_y = pygame.mouse.get_pos()

    left_held = pygame.mouse.get_pressed()[0]
    right_held = pygame.mouse.get_pressed()[2]

    point_sign = 0

    if left_held:
        point_sign = 1
    elif right_held:
        point_sign = 2

    test_pop.next_frame(0.07,0.08,0.0005,boid_size + 1.0,30.0,0.2, 0.9, mouse_x, mouse_y, 0.60, point_sign)

    frame += 1
    clock.tick(120)

pygame.quit()