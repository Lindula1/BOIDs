import pygame

from gpu_optimised import GPUPopulation


# ------------------------------------------------------------
# Pygame setup
# ------------------------------------------------------------

pygame.init()

WIDTH = 300
HEIGHT = 300

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "GPU Boid Simulation"
)

clock = pygame.time.Clock()

# Direct pixel access.
pixels = pygame.surfarray.pixels3d(screen)


# ------------------------------------------------------------
# Simulation parameters
# ------------------------------------------------------------

boid_size = 2

border_adjust = 2

border = (
    border_adjust,
    HEIGHT - border_adjust,
    border_adjust,
    WIDTH - border_adjust,
)

margin_size = 50

margin = (
    border[0] + margin_size,
    border[1] - margin_size,
    border[2] + margin_size,
    border[3] - margin_size,
)


# ------------------------------------------------------------
# Create GPU population
# ------------------------------------------------------------

test_pop = GPUPopulation(
    border=border,

    population_size=200,

    margin=margin,

    frame_rate=1.0,

    safe_distance=4.0,

    turn_factor=0.4,

    separation=0.05,

    alignment=0.05,

    visible_distance=30.0,

    max_speed=6.0,

    min_speed=2.0,

    cohesion=0.003,

    rebound_factor=0.9,
)


# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------

running = True

while running:

    # --------------------------------------------------------
    # Events
    # --------------------------------------------------------

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

    # --------------------------------------------------------
    # Advance GPU simulation FIRST
    # --------------------------------------------------------

    test_pop.next_frame()

    # --------------------------------------------------------
    # Clear screen
    # --------------------------------------------------------

    screen.fill((0, 0, 0))

    # --------------------------------------------------------
    # Copy positions GPU -> CPU
    # --------------------------------------------------------

    boid_pos = test_pop.get_positions()

    # --------------------------------------------------------
    # Draw
    # --------------------------------------------------------

    for x, y in boid_pos:

        x = int(x)
        y = int(y)

        # Keep drawing inside the surface.
        if (
            0 <= x < WIDTH
            and
            0 <= y < HEIGHT
        ):

            x_end = min(
                x + boid_size,
                WIDTH
            )

            y_end = min(
                y + boid_size,
                HEIGHT
            )

            pixels[
                x:x_end,
                y:y_end
            ] = [255, 255, 255]

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    pygame.display.flip()

    clock.tick(120)


pygame.quit()