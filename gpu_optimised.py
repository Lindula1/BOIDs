import os
import math

import numpy as np

# ------------------------------------------------------------
# CUDA DLL setup
# ------------------------------------------------------------

def setup_cuda_dlls():
    """
    Locate a normal NVIDIA CUDA Toolkit installation on Windows.

    This is needed because Windows may not automatically expose
    nvvm.dll to Python.
    """

    if os.name != "nt":
        return

    cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"

    if not os.path.isdir(cuda_base):
        return

    # Find installed CUDA versions.
    versions = []

    for name in os.listdir(cuda_base):
        path = os.path.join(cuda_base, name)

        if os.path.isdir(path) and name.lower().startswith("v"):
            versions.append((name, path))

    # Prefer the newest installed version.
    versions.sort(reverse=True)

    for _, path in versions:
        bin_dir = os.path.join(path, "bin")
        nvvm_dir = os.path.join(path, "nvvm", "bin")

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

        # Also add to PATH for dependencies of nvvm.dll.
        os.environ["PATH"] = (
            bin_dir
            + os.pathsep
            + nvvm_dir
            + os.pathsep
            + os.environ.get("PATH", "")
        )

        # CUDA_HOME helps Numba locate the toolkit.
        os.environ.setdefault("CUDA_HOME", path)

        break


setup_cuda_dlls()


# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

from numba import cuda


# ------------------------------------------------------------
# GPU kernel
# ------------------------------------------------------------

@cuda.jit
def update_boids(
    positions,
    velocities,
    new_positions,
    new_velocities,

    visible_distance_sq,
    safe_distance_sq,

    separation,
    alignment,
    cohesion,

    separation_random,
    alignment_random,
    cohesion_random,

    max_speed,
    min_speed,

    turn_factor,
    rebound_factor,

    margin_top,
    margin_bottom,
    margin_left,
    margin_right,

    border_top,
    border_bottom,
    border_left,
    border_right,

    frame_rate,
):
    """
    One CUDA thread handles one boid.

    positions:
        [N, 2]

    velocities:
        [N, 2]

    new_positions / new_velocities:
        output arrays

    All threads read the OLD state and write to NEW state.
    This prevents race conditions between boids.
    """

    i = cuda.grid(1)

    n = positions.shape[0]

    if i >= n:
        return

    # --------------------------------------------------------
    # Current boid
    # --------------------------------------------------------

    x = positions[i, 0]
    y = positions[i, 1]

    vx = velocities[i, 0]
    vy = velocities[i, 1]

    # --------------------------------------------------------
    # Accumulators
    # --------------------------------------------------------

    close_x = 0.0
    close_y = 0.0

    velocity_sum_x = 0.0
    velocity_sum_y = 0.0

    position_sum_x = 0.0
    position_sum_y = 0.0

    neighbour_count = 0

    # --------------------------------------------------------
    # O(N²) neighbour search
    # --------------------------------------------------------

    for j in range(n):

        if i == j:
            continue

        dx = x - positions[j, 0]
        dy = y - positions[j, 1]

        distance_sq = dx * dx + dy * dy

        # Outside visible range.
        if distance_sq > visible_distance_sq:
            continue

        # ----------------------------------------------------
        # Separation
        # ----------------------------------------------------

        if distance_sq <= safe_distance_sq:

            close_x += dx
            close_y += dy

        # ----------------------------------------------------
        # Alignment + cohesion
        # ----------------------------------------------------

        else:

            velocity_sum_x += velocities[j, 0]
            velocity_sum_y += velocities[j, 1]

            position_sum_x += positions[j, 0]
            position_sum_y += positions[j, 1]

            neighbour_count += 1

    # --------------------------------------------------------
    # Separation
    # --------------------------------------------------------

    vx += close_x * separation * separation_random[i]
    vy += close_y * separation * separation_random[i]

    # --------------------------------------------------------
    # Alignment + cohesion
    # --------------------------------------------------------

    if neighbour_count > 0:

        inv_count = 1.0 / neighbour_count

        average_velocity_x = velocity_sum_x * inv_count
        average_velocity_y = velocity_sum_y * inv_count

        average_position_x = position_sum_x * inv_count
        average_position_y = position_sum_y * inv_count

        # Alignment
        vx += (
                      average_velocity_x - vx
              ) * alignment * alignment_random[i]

        vy += (
                      average_velocity_y - vy
              ) * alignment * alignment_random[i]

        # Cohesion
        vx += (
                      average_position_x - x
              ) * cohesion * cohesion_random[i]

        vy += (
                      average_position_y - y
              ) * cohesion * cohesion_random[i]

    # --------------------------------------------------------
    # Margin avoidance
    # --------------------------------------------------------

    if x < margin_left:
        vx += turn_factor

    elif x > margin_right:
        vx -= turn_factor

    if y < margin_top:
        vy += turn_factor

    elif y > margin_bottom:
        vy -= turn_factor

    # --------------------------------------------------------
    # Speed limiting
    # --------------------------------------------------------

    speed_sq = vx * vx + vy * vy

    if speed_sq > 0.0:

        speed = math.sqrt(speed_sq)

        # Correctly clamp to MAX speed.
        if speed > max_speed:

            scale = max_speed / speed

            vx *= scale
            vy *= scale

        # Correctly maintain MIN speed.
        elif speed < min_speed:

            scale = min_speed / speed

            vx *= scale
            vy *= scale

    else:

        # Avoid a stationary boid.
        vx = min_speed
        vy = 0.0

    # --------------------------------------------------------
    # Border rebound
    # --------------------------------------------------------

    if x < border_left:

        x = border_left
        vx = abs(vx) * rebound_factor

    elif x > border_right:

        x = border_right
        vx = -abs(vx) * rebound_factor

    if y < border_top:

        y = border_top
        vy = abs(vy) * rebound_factor

    elif y > border_bottom:

        y = border_bottom
        vy = -abs(vy) * rebound_factor

    # --------------------------------------------------------
    # Position update
    # --------------------------------------------------------

    x += vx * frame_rate
    y += vy * frame_rate

    # --------------------------------------------------------
    # Write new state
    # --------------------------------------------------------

    new_positions[i, 0] = x
    new_positions[i, 1] = y

    new_velocities[i, 0] = vx
    new_velocities[i, 1] = vy


# ------------------------------------------------------------
# Population
# ------------------------------------------------------------

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
    ):

        # ----------------------------------------------------
        # CUDA check
        # ----------------------------------------------------

        if not cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. "
                "Check your NVIDIA driver and CUDA Toolkit installation."
            )

        # ----------------------------------------------------
        # Basic parameters
        # ----------------------------------------------------

        self.population_size = population_size

        self.frame_rate = np.float32(frame_rate)

        self.safe_distance = np.float32(safe_distance)
        self.visible_distance = np.float32(visible_distance)

        self.separation = np.float32(separation)
        self.alignment = np.float32(alignment)
        self.cohesion = np.float32(cohesion)

        self.max_speed = np.float32(max_speed)
        self.min_speed = np.float32(min_speed)

        self.turn_factor = np.float32(turn_factor)
        self.rebound_factor = np.float32(rebound_factor)

        # ----------------------------------------------------
        # Border format:
        #
        # (top, bottom, left, right)
        # ----------------------------------------------------

        self.border_top = np.float32(border[0])
        self.border_bottom = np.float32(border[1])
        self.border_left = np.float32(border[2])
        self.border_right = np.float32(border[3])

        # ----------------------------------------------------
        # Margin format:
        #
        # (top, bottom, left, right)
        # ----------------------------------------------------

        self.margin_top = np.float32(margin[0])
        self.margin_bottom = np.float32(margin[1])
        self.margin_left = np.float32(margin[2])
        self.margin_right = np.float32(margin[3])

        # ----------------------------------------------------
        # Squared distances
        #
        # Avoid sqrt() in the neighbour search.
        # ----------------------------------------------------

        self.visible_distance_sq = np.float32(
            visible_distance * visible_distance
        )

        self.safe_distance_sq = np.float32(
            safe_distance * safe_distance
        )

        # ----------------------------------------------------
        # Initial CPU state
        # ----------------------------------------------------

        rng = np.random.default_rng()

        positions = np.empty(
            (population_size, 2),
            dtype=np.float32
        )

        velocities = np.empty(
            (population_size, 2),
            dtype=np.float32
        )

        # Random initial positions.
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

        separation_random = rng.uniform(
            0.0,
            5.0,
            population_size
        ).astype(np.float32)

        alignment_random = rng.uniform(
            0.0,
            5.0,
            population_size
        ).astype(np.float32)

        cohesion_random = rng.uniform(
            0.0,
            5.0,
            population_size
        ).astype(np.float32)

        # Random initial directions.
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

        velocities[:, 0] = np.cos(angles) * speeds
        velocities[:, 1] = np.sin(angles) * speeds

        # ----------------------------------------------------
        # Copy state to GPU
        # ----------------------------------------------------

        self.positions = cuda.to_device(positions)
        self.velocities = cuda.to_device(velocities)

        self.separation_random = cuda.to_device(
            separation_random
        )

        self.alignment_random = cuda.to_device(
            alignment_random
        )

        self.cohesion_random = cuda.to_device(
            cohesion_random
        )

        # Double buffers.
        self.new_positions = cuda.device_array_like(
            self.positions
        )

        self.new_velocities = cuda.device_array_like(
            self.velocities
        )

        # ----------------------------------------------------
        # CUDA launch configuration
        # ----------------------------------------------------

        self.threads_per_block = 256

        self.blocks_per_grid = (
            population_size
            + self.threads_per_block
            - 1
        ) // self.threads_per_block

        # ----------------------------------------------------
        # First kernel compilation
        #
        # This gives a clean error during construction rather
        # than halfway through the Pygame loop.
        # ----------------------------------------------------

        self._test_cuda()

    def _test_cuda(self):

        test = cuda.device_array(
            1,
            dtype=np.float32
        )

        test.copy_to_device(
            np.array([1.0], dtype=np.float32)
        )

        # Synchronise to ensure CUDA is functioning.
        cuda.synchronize()

        del test

    # --------------------------------------------------------
    # Advance simulation
    # --------------------------------------------------------

    def next_frame(self):

        update_boids[
            self.blocks_per_grid,
            self.threads_per_block
        ](
            self.positions,
            self.velocities,

            self.new_positions,
            self.new_velocities,

            self.visible_distance_sq,
            self.safe_distance_sq,

            self.separation,
            self.alignment,
            self.cohesion,

            self.separation_random,
            self.alignment_random,
            self.cohesion_random,

            self.max_speed,
            self.min_speed,

            self.turn_factor,
            self.rebound_factor,

            self.margin_top,
            self.margin_bottom,
            self.margin_left,
            self.margin_right,

            self.border_top,
            self.border_bottom,
            self.border_left,
            self.border_right,

            self.frame_rate,
        )

        # Make sure the kernel has completed.
        cuda.synchronize()

        # Swap buffers.
        (
            self.positions,
            self.new_positions
        ) = (
            self.new_positions,
            self.positions
        )

        (
            self.velocities,
            self.new_velocities
        ) = (
            self.new_velocities,
            self.velocities
        )

    # --------------------------------------------------------
    # Get positions for Pygame
    # --------------------------------------------------------

    def get_positions(self):

        return self.positions.copy_to_host()

    # --------------------------------------------------------
    # Optional accessors
    # --------------------------------------------------------

    def get_velocities(self):

        return self.velocities.copy_to_host()

    def get_positions_device(self):

        return self.positions

    def get_velocities_device(self):

        return self.velocities