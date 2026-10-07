import pygame
from optimised_boid_model import Population

pygame.init()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

font = pygame.font.Font(None, 22)

def display_text(text : str, center : tuple[int, int]):
    text_surface = font.render(text, True, WHITE)

    text_rect = text_surface.get_rect()
    text_rect.center = center
    return text_surface, text_rect

WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("BOIDs Optimised")

clock = pygame.time.Clock()

running = True

boid_size = 2

border_adjust = 2
border = (border_adjust, int(HEIGHT-border_adjust), border_adjust, int(WIDTH-border_adjust))

separation = 0.05
alignment = 0.07
cohesion = 0.002
safe_distance = boid_size + 1
visible_distance = boid_size * 12
turn_factor = 0.4
rebound_factor = 0.7
min_speed = 3
max_speed = 4

test_pop = Population(border, 50, 1000, min_speed, max_speed)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BLACK)
    boid_pos = test_pop.get_positions()

    pixels = pygame.surfarray.pixels3d(screen)

    for pos in boid_pos:
        pixels[pos[0]:pos[0]+boid_size, pos[1]:pos[1]+boid_size] = WHITE

    del pixels

    t_surf, t_rect = display_text(f"frame rate : {int(clock.get_fps())}     frame time : {clock.get_time()}", (WIDTH-130, HEIGHT-20))
    screen.blit(t_surf, t_rect)

    pygame.display.flip()

    mouse_x, mouse_y = pygame.mouse.get_pos()
    test_pop.next_frame(separation, alignment, cohesion, safe_distance, visible_distance, turn_factor, rebound_factor, mouse_x, mouse_y, 0.05)

    clock.tick(75)

pygame.quit()