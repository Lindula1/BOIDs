import os
import math

import numpy as np


# ============================================================
# CUDA DLL SETUP
# ============================================================

def setup_cuda_dlls():
    """
    Locate a normal NVIDIA CUDA Toolkit installation on Windows.

    This allows Numba to find nvvm.dll and other CUDA DLLs.
    """

    if os.name != "nt":
        return

    cuda_base = (
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    )

    if not os.path.isdir(cuda_base):
        return

    versions = []

    for name in os.listdir(cuda_base):

        path = os.path.join(cuda_base, name)

        if (
            os.path.isdir(path)
            and name.lower().startswith("v")
        ):
            versions.append((name, path))

    # Newest CUDA version first.
    versions.sort(reverse=True)

    for _, path in versions:

        bin_dir = os.path.join(
            path,
            "bin"
        )

        nvvm_dir = os.path.join(
            path,
            "nvvm",
            "bin"
        )

        if os.path.isdir(bin_dir):

            try:
                os.add_dll_directory(bin_dir)
            except OSError:
                pass

        if os.path.isdir(nvvm_dir):

            try:
                os.add_dll_directory(nvvm_dir)
            except OSError:
                pass

        # Add CUDA directories to PATH.
        os.environ["PATH"] = (
            bin_dir
            + os.pathsep
            + nvvm_dir
            + os.pathsep
            + os.environ.get("PATH", "")
        )

        os.environ.setdefault(
            "CUDA_HOME",
            path
        )

        break


setup_cuda_dlls()


# ============================================================
# NUMBA CUDA
# ============================================================

from numba import cuda


# ============================================================
# CUDA KERNEL
# ============================================================

@cuda.jit
def update_boids(

    positions,
    velocities,

    new_positions,
    new_velocities,

    # --------------------------------------------------------
    # Neighbour distances
    # --------------------------------------------------------

    visible_distance_sq,
    safe_distance_sq,

    # --------------------------------------------------------
    # Base behaviour strengths
    # --------------------------------------------------------

    separation,
    alignment,
    cohesion,

    # --------------------------------------------------------
    # Per-boid random behaviour
    # --------------------------------------------------------

    separation_random,
    alignment_random,
    cohesion_random,

    # --------------------------------------------------------
    # Speed
    # --------------------------------------------------------

    max_speed,
    min_speed,

    # --------------------------------------------------------
    # Border behaviour
    # --------------------------------------------------------

    turn_factor,
    rebound_factor,

    # --------------------------------------------------------
    # Margin
    # --------------------------------------------------------

    margin_top,
    margin_bottom,
    margin_left,
    margin_right,

    # --------------------------------------------------------
    # Border
    # --------------------------------------------------------

    border_top,
    border_bottom,
    border_left,
    border_right,

    # --------------------------------------------------------
    # Simulation timestep
    # --------------------------------------------------------

    frame_rate,

    # --------------------------------------------------------
    # Mouse
    # --------------------------------------------------------

    mouse_x,
    mouse_y,
    mouse_radius_sq,
    mouse_strength,
    mouse_mode,

):
    """
    One CUDA thread handles one boid.

    mouse_mode:

        0 = no mouse interaction
        1 = attraction
        2 = repulsion
    """

    # --------------------------------------------------------
    # Which boid does this thread handle?
    # --------------------------------------------------------

    i = cuda.grid(1)

    n = positions.shape[0]

    if i >= n:
        return

    # ========================================================
    # CURRENT BOID
    # ========================================================

    x = positions[i, 0]
    y = positions[i, 1]

    vx = velocities[i, 0]
    vy = velocities[i, 1]

    # ========================================================
    # ACCUMULATORS
    # ========================================================

    # Separation
    close_x = 0.0
    close_y = 0.0

    # Alignment
    velocity_sum_x = 0.0
    velocity_sum_y = 0.0

    # Cohesion
    position_sum_x = 0.0
    position_sum_y = 0.0

    neighbour_count = 0

    # ========================================================
    # O(N²) NEIGHBOUR SEARCH
    # ========================================================

    for j in range(n):

        if i == j:
            continue

        # ----------------------------------------------------
        # Difference between boids
        # ----------------------------------------------------

        dx = x - positions[j, 0]
        dy = y - positions[j, 1]

        distance_sq = (
            dx * dx
            +
            dy * dy
        )

        # ----------------------------------------------------
        # Outside visible range
        # ----------------------------------------------------

        if distance_sq > visible_distance_sq:
            continue

        # ====================================================
        # SEPARATION
        # ====================================================

        if distance_sq <= safe_distance_sq:

            close_x += dx
            close_y += dy

        # ====================================================
        # ALIGNMENT + COHESION
        # ====================================================

        else:

            velocity_sum_x += velocities[j, 0]
            velocity_sum_y += velocities[j, 1]

            position_sum_x += positions[j, 0]
            position_sum_y += positions[j, 1]

            neighbour_count += 1

    # ========================================================
    # SEPARATION
    # ========================================================

    separation_strength = (
        separation
        *
        separation_random[i]
    )

    vx += (
        close_x
        *
        separation_strength
    )

    vy += (
        close_y
        *
        separation_strength
    )

    # ========================================================
    # ALIGNMENT + COHESION
    # ========================================================

    if neighbour_count > 0:

        inv_count = 1.0 / neighbour_count

        # ----------------------------------------------------
        # Average velocity
        # ----------------------------------------------------

        average_velocity_x = (
            velocity_sum_x
            *
            inv_count
        )

        average_velocity_y = (
            velocity_sum_y
            *
            inv_count
        )

        # ----------------------------------------------------
        # Average position
        # ----------------------------------------------------

        average_position_x = (
            position_sum_x
            *
            inv_count
        )

        average_position_y = (
            position_sum_y
            *
            inv_count
        )

        # ====================================================
        # ALIGNMENT
        # ====================================================

        alignment_strength = (
            alignment
            *
            alignment_random[i]
        )

        vx += (
            average_velocity_x
            -
            vx
        ) * alignment_strength

        vy += (
            average_velocity_y
            -
            vy
        ) * alignment_strength

        # ====================================================
        # COHESION
        # ====================================================

        cohesion_strength = (
            cohesion
            *
            cohesion_random[i]
        )

        vx += (
            average_position_x
            -
            x
        ) * cohesion_strength

        vy += (
            average_position_y
            -
            y
        ) * cohesion_strength

    # ========================================================
    # MOUSE INTERACTION
    # ========================================================

    if mouse_mode != 0:

        mouse_dx = mouse_x - x
        mouse_dy = mouse_y - y

        mouse_distance_sq = (
            mouse_dx * mouse_dx
            +
            mouse_dy * mouse_dy
        )

        # ----------------------------------------------------
        # Mouse is inside interaction radius
        # ----------------------------------------------------

        if (
            mouse_distance_sq > 0.0
            and
            mouse_distance_sq <= mouse_radius_sq
        ):

            mouse_distance = math.sqrt(
                mouse_distance_sq
            )

            # ------------------------------------------------
            # Normalised direction
            # ------------------------------------------------

            direction_x = (
                mouse_dx
                /
                mouse_distance
            )

            direction_y = (
                mouse_dy
                /
                mouse_distance
            )

            # ------------------------------------------------
            # Distance-based falloff
            #
            # 1.0 at mouse
            # 0.0 at edge
            # ------------------------------------------------

            falloff = (
                1.0
                -
                mouse_distance
                /
                math.sqrt(mouse_radius_sq)
            )

            force = (
                mouse_strength
                *
                falloff
            )

            # ------------------------------------------------
            # Attraction
            # ------------------------------------------------

            if mouse_mode == 1:

                vx += (
                    direction_x
                    *
                    force
                )

                vy += (
                    direction_y
                    *
                    force
                )

            # ------------------------------------------------
            # Repulsion
            # ------------------------------------------------

            elif mouse_mode == 2:

                vx -= (
                    direction_x
                    *
                    force
                )

                vy -= (
                    direction_y
                    *
                    force
                )

    # ========================================================
    # MARGIN AVOIDANCE
    # ========================================================

    if x < margin_left:

        vx += turn_factor

    elif x > margin_right:

        vx -= turn_factor

    if y < margin_top:

        vy += turn_factor

    elif y > margin_bottom:

        vy -= turn_factor

    # ========================================================
    # SPEED LIMITING
    # ========================================================

    speed_sq = (
        vx * vx
        +
        vy * vy
    )

    if speed_sq > 0.0:

        speed = math.sqrt(
            speed_sq
        )

        # ----------------------------------------------------
        # Maximum speed
        # ----------------------------------------------------

        if speed > max_speed:

            scale = (
                max_speed
                /
                speed
            )

            vx *= scale
            vy *= scale

        # ----------------------------------------------------
        # Minimum speed
        # ----------------------------------------------------

        elif speed < min_speed:

            scale = (
                min_speed
                /
                speed
            )

            vx *= scale
            vy *= scale

    else:

        # Prevent stationary boids.
        vx = min_speed
        vy = 0.0

    # ========================================================
    # BORDER REBOUND
    # ========================================================

    if x < border_left:

        x = border_left

        vx = (
            abs(vx)
            *
            rebound_factor
        )

    elif x > border_right:

        x = border_right

        vx = (
            -abs(vx)
            *
            rebound_factor
        )

    if y < border_top:

        y = border_top

        vy = (
            abs(vy)
            *
            rebound_factor
        )

    elif y > border_bottom:

        y = border_bottom

        vy = (
            -abs(vy)
            *
            rebound_factor
        )

    # ========================================================
    # POSITION UPDATE
    # ========================================================

    x += (
        vx
        *
        frame_rate
    )

    y += (
        vy
        *
        frame_rate
    )

    # ========================================================
    # WRITE NEW STATE
    # ========================================================

    new_positions[i, 0] = x
    new_positions[i, 1] = y

    new_velocities[i, 0] = vx
    new_velocities[i, 1] = vy


# ============================================================
# GPU POPULATION
# ============================================================

class GPUPopulation:

    def __init__(

        self,

        border,
        population_size,
        margin,

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

        # ----------------------------------------------------
        # Mouse settings
        # ----------------------------------------------------

        mouse_radius=75.0,
        mouse_strength=0.5,

        # ----------------------------------------------------
        # Random behaviour range
        # ----------------------------------------------------

        random_min=0.0,
        random_max=5.0,
    ):

        # ====================================================
        # CUDA CHECK
        # ====================================================

        if not cuda.is_available():

            raise RuntimeError(
                "CUDA is not available. "
                "Check your NVIDIA driver and CUDA Toolkit."
            )

        # ====================================================
        # BASIC PARAMETERS
        # ====================================================

        self.population_size = (
            population_size
        )

        self.frame_rate = np.float32(
            frame_rate
        )

        self.safe_distance = np.float32(
            safe_distance
        )

        self.visible_distance = np.float32(
            visible_distance
        )

        self.separation = np.float32(
            separation
        )

        self.alignment = np.float32(
            alignment
        )

        self.cohesion = np.float32(
            cohesion
        )

        self.max_speed = np.float32(
            max_speed
        )

        self.min_speed = np.float32(
            min_speed
        )

        self.turn_factor = np.float32(
            turn_factor
        )

        self.rebound_factor = np.float32(
            rebound_factor
        )

        # ====================================================
        # MOUSE PARAMETERS
        # ====================================================

        self.mouse_radius = np.float32(
            mouse_radius
        )

        self.mouse_radius_sq = np.float32(
            mouse_radius * mouse_radius
        )

        self.mouse_strength = np.float32(
            mouse_strength
        )

        # ====================================================
        # BORDER
        #
        # (top, bottom, left, right)
        # ====================================================

        self.border_top = np.float32(
            border[0]
        )

        self.border_bottom = np.float32(
            border[1]
        )

        self.border_left = np.float32(
            border[2]
        )

        self.border_right = np.float32(
            border[3]
        )

        # ====================================================
        # MARGIN
        #
        # (top, bottom, left, right)
        # ====================================================

        self.margin_top = np.float32(
            margin[0]
        )

        self.margin_bottom = np.float32(
            margin[1]
        )

        self.margin_left = np.float32(
            margin[2]
        )

        self.margin_right = np.float32(
            margin[3]
        )

        # ====================================================
        # SQUARED DISTANCES
        # ====================================================

        self.visible_distance_sq = np.float32(
            visible_distance
            *
            visible_distance
        )

        self.safe_distance_sq = np.float32(
            safe_distance
            *
            safe_distance
        )

        # ====================================================
        # RANDOM NUMBER GENERATOR
        # ====================================================

        rng = np.random.default_rng()

        # ====================================================
        # INITIAL POSITIONS
        # ====================================================

        positions = np.empty(
            (
                population_size,
                2
            ),
            dtype=np.float32
        )

        positions[:, 0] = rng.uniform(
            self.border_left,
            self.border_right,
            population_size
        )

        positions[:, 1] = rng.uniform(
            self.border_top,
            self.border_bottom,
            population_size
        )

        # ====================================================
        # INITIAL VELOCITIES
        # ====================================================

        velocities = np.empty(
            (
                population_size,
                2
            ),
            dtype=np.float32
        )

        angles = rng.uniform(
            0.0,
            2.0 * np.pi,
            population_size
        )

        speeds = rng.uniform(
            self.min_speed,
            self.max_speed,
            population_size
        )

        velocities[:, 0] = (
            np.cos(angles)
            *
            speeds
        )

        velocities[:, 1] = (
            np.sin(angles)
            *
            speeds
        )

        # ====================================================
        # RANDOM BOID PERSONALITIES
        # ====================================================

        separation_random = rng.uniform(
            random_min,
            random_max,
            population_size
        ).astype(np.float32)

        alignment_random = rng.uniform(
            random_min,
            random_max,
            population_size
        ).astype(np.float32)

        cohesion_random = rng.uniform(
            random_min,
            random_max,
            population_size
        ).astype(np.float32)

        # ====================================================
        # COPY STATE TO GPU
        # ====================================================

        self.positions = (
            cuda.to_device(
                positions
            )
        )

        self.velocities = (
            cuda.to_device(
                velocities
            )
        )

        # ====================================================
        # COPY PERSONALITIES TO GPU
        # ====================================================

        self.separation_random = (
            cuda.to_device(
                separation_random
            )
        )

        self.alignment_random = (
            cuda.to_device(
                alignment_random
            )
        )

        self.cohesion_random = (
            cuda.to_device(
                cohesion_random
            )
        )

        # ====================================================
        # DOUBLE BUFFERS
        # ====================================================

        self.new_positions = (
            cuda.device_array_like(
                self.positions
            )
        )

        self.new_velocities = (
            cuda.device_array_like(
                self.velocities
            )
        )

        # ====================================================
        # CUDA LAUNCH CONFIGURATION
        # ====================================================

        self.threads_per_block = 256

        self.blocks_per_grid = (
            population_size
            +
            self.threads_per_block
            -
            1
        ) // self.threads_per_block

        # ====================================================
        # MOUSE STATE
        # ====================================================

        self.mouse_x = np.float32(0.0)
        self.mouse_y = np.float32(0.0)

        # 0 = off
        # 1 = attraction
        # 2 = repulsion
        self.mouse_mode = np.int32(0)

        # ====================================================
        # CUDA TEST
        # ====================================================

        self._test_cuda()

    # ========================================================
    # CUDA TEST
    # ========================================================

    def _test_cuda(self):

        test = cuda.device_array(
            1,
            dtype=np.float32
        )

        test.copy_to_device(
            np.array(
                [1.0],
                dtype=np.float32
            )
        )

        cuda.synchronize()

        del test

    # ========================================================
    # SET MOUSE
    # ========================================================

    def set_mouse(
        self,
        x,
        y,
        mode=0,
    ):
        """
        Set mouse position and interaction mode.

        mode:
            0 = off
            1 = attract
            2 = repel
        """

        self.mouse_x = np.float32(x)

        self.mouse_y = np.float32(y)

        self.mouse_mode = np.int32(
            mode
        )

    # ========================================================
    # UPDATE
    # ========================================================

    def next_frame(self):

        update_boids[
            self.blocks_per_grid,
            self.threads_per_block
        ](

            self.positions,
            self.velocities,

            self.new_positions,
            self.new_velocities,

            # ------------------------------------------------
            # Distances
            # ------------------------------------------------

            self.visible_distance_sq,
            self.safe_distance_sq,

            # ------------------------------------------------
            # Behaviour
            # ------------------------------------------------

            self.separation,
            self.alignment,
            self.cohesion,

            # ------------------------------------------------
            # Random behaviour
            # ------------------------------------------------

            self.separation_random,
            self.alignment_random,
            self.cohesion_random,

            # ------------------------------------------------
            # Speed
            # ------------------------------------------------

            self.max_speed,
            self.min_speed,

            # ------------------------------------------------
            # Border
            # ------------------------------------------------

            self.turn_factor,
            self.rebound_factor,

            # ------------------------------------------------
            # Margin
            # ------------------------------------------------

            self.margin_top,
            self.margin_bottom,
            self.margin_left,
            self.margin_right,

            # ------------------------------------------------
            # Border
            # ------------------------------------------------

            self.border_top,
            self.border_bottom,
            self.border_left,
            self.border_right,

            # ------------------------------------------------
            # Frame rate
            # ------------------------------------------------

            self.frame_rate,

            # ------------------------------------------------
            # Mouse
            # ------------------------------------------------

            self.mouse_x,
            self.mouse_y,

            self.mouse_radius_sq,

            self.mouse_strength,

            self.mouse_mode,
        )

        # ====================================================
        # WAIT FOR GPU
        # ====================================================

        cuda.synchronize()

        # ====================================================
        # SWAP POSITION BUFFERS
        # ====================================================

        (
            self.positions,
            self.new_positions
        ) = (
            self.new_positions,
            self.positions
        )

        # ====================================================
        # SWAP VELOCITY BUFFERS
        # ====================================================

        (
            self.velocities,
            self.new_velocities
        ) = (
            self.new_velocities,
            self.velocities
        )

    # ========================================================
    # GET POSITIONS
    # ========================================================

    def get_positions(self):

        return (
            self.positions.copy_to_host()
        )

    # ========================================================
    # GET VELOCITIES
    # ========================================================

    def get_velocities(self):

        return (
            self.velocities.copy_to_host()
        )

    # ========================================================
    # GET DEVICE ARRAYS
    # ========================================================

    def get_positions_device(self):

        return self.positions

    def get_velocities_device(self):

        return self.velocities

    # ========================================================
    # GET RANDOM PERSONALITIES
    # ========================================================

    def get_random_behaviour(self):

        separation = (
            self.separation_random.copy_to_host()
        )

        alignment = (
            self.alignment_random.copy_to_host()
        )

        cohesion = (
            self.cohesion_random.copy_to_host()
        )

        return (
            separation,
            alignment,
            cohesion
        )