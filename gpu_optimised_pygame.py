import pygame

from gpu_optimised import GPUPopulation


# ============================================================
# PYGAME INITIALISATION
# ============================================================

pygame.init()


# ============================================================
# WINDOW
# ============================================================

WIDTH = 800
HEIGHT = 800

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "GPU Boid Simulation"
)

clock = pygame.time.Clock()


# ============================================================
# PIXEL ACCESS
# ============================================================

pixels = pygame.surfarray.pixels3d(
    screen
)


# ============================================================
# BOID SETTINGS
# ============================================================

BOID_COUNT = 2000

BOID_SIZE = 2


# ============================================================
# BORDER
#
# (top, bottom, left, right)
# ============================================================

border_adjust = 2

border = (
    border_adjust,
    HEIGHT - border_adjust,
    border_adjust,
    WIDTH - border_adjust,
)


# ============================================================
# MARGIN
# ============================================================

margin_size = 100

margin = (
    border[0] + margin_size,
    border[1] - margin_size,
    border[2] + margin_size,
    border[3] - margin_size,
)


# ============================================================
# CREATE GPU POPULATION
# ============================================================

test_pop = GPUPopulation(

    border=border,

    population_size=BOID_COUNT,

    margin=margin,

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    frame_rate=1.0,

    # --------------------------------------------------------
    # Neighbour behaviour
    # --------------------------------------------------------

    safe_distance=8.0,

    visible_distance=50.0,

    # --------------------------------------------------------
    # Behaviour strengths
    # --------------------------------------------------------

    separation=0.05,

    alignment=0.05,

    cohesion=0.003,

    # --------------------------------------------------------
    # Movement
    # --------------------------------------------------------

    max_speed=6.0,

    min_speed=2.0,

    # --------------------------------------------------------
    # Borders
    # --------------------------------------------------------

    turn_factor=0.4,

    rebound_factor=0.9,

    # --------------------------------------------------------
    # Mouse
    # --------------------------------------------------------

    mouse_radius=150.0,

    mouse_strength=1.0,

    # --------------------------------------------------------
    # Individual personality
    # --------------------------------------------------------

    random_min=0.0,

    random_max=5.0,
)


# ============================================================
# MAIN LOOP
# ============================================================

running = True

while running:

    # ========================================================
    # EVENTS
    # ========================================================

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            running = False

        # ----------------------------------------------------
        # Escape closes the simulation.
        # ----------------------------------------------------

        elif (
            event.type == pygame.KEYDOWN
            and
            event.key == pygame.K_ESCAPE
        ):

            running = False

    # ========================================================
    # MOUSE POSITION
    # ========================================================

    mouse_x, mouse_y = pygame.mouse.get_pos()

    # ========================================================
    # MOUSE BUTTONS
    # ========================================================

    mouse_buttons = pygame.mouse.get_pressed()

    left_button = mouse_buttons[0]
    right_button = mouse_buttons[2]

    # ========================================================
    # DETERMINE MOUSE MODE
    # ========================================================

    if left_button:

        # Left click = attraction

        mouse_mode = 1

    elif right_button:

        # Right click = repulsion

        mouse_mode = 2

    else:

        # No interaction

        mouse_mode = 0

    # ========================================================
    # SEND MOUSE STATE TO GPU
    # ========================================================

    test_pop.set_mouse(
        mouse_x,
        mouse_y,
        mouse_mode
    )

    # ========================================================
    # UPDATE GPU SIMULATION
    # ========================================================

    test_pop.next_frame()

    # ========================================================
    # CLEAR SCREEN
    # ========================================================

    screen.fill(
        (0, 0, 0)
    )

    # ========================================================
    # GET POSITIONS
    # ========================================================

    boid_positions = (
        test_pop.get_positions()
    )

    # ========================================================
    # DRAW BOIDS
    # ========================================================

    for x, y in boid_positions:

        x = int(x)
        y = int(y)

        if (
            0 <= x < WIDTH
            and
            0 <= y < HEIGHT
        ):

            x_end = min(
                x + BOID_SIZE,
                WIDTH
            )

            y_end = min(
                y + BOID_SIZE,
                HEIGHT
            )

            pixels[
                x:x_end,
                y:y_end
            ] = (
                255,
                255,
                255
            )

    # ========================================================
    # DRAW MOUSE INTERACTION AREA
    # ========================================================

    if mouse_mode != 0:

        if mouse_mode == 1:

            # Attraction
            mouse_colour = (
                0,
                255,
                0
            )

        else:

            # Repulsion
            mouse_colour = (
                255,
                0,
                0
            )

        pygame.draw.circle(
            screen,
            mouse_colour,
            (
                mouse_x,
                mouse_y
            ),
            int(test_pop.mouse_radius),
            1
        )

    # ========================================================
    # DISPLAY
    # ========================================================

    pygame.display.flip()

    # ========================================================
    # FRAME LIMIT
    # ========================================================

    clock.tick(120)


# ============================================================
# CLEANUP
# ============================================================

pygame.quit()