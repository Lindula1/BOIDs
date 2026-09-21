import numpy as np
import os

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

from numba import cuda

int_dtype = np.int32
float_dtype = np.float32

def is_tuple_4(iterable) -> bool:
    return isinstance(iterable, tuple) and len(iterable) == 4 and all(isinstance(x, int) and not isinstance(x, bool) for x in iterable)


@cuda.jit
def boid_kernel(old_positions, old_velocities, positions, velocities, alignment_randoms, bottom, top, left, right, margin_offset, min_speed, max_speed, separation,
                alignment, cohesion, safe_distance, visible_distance, turn_factor, rebound_factor, point_x, point_y, point_factor, point_sign):

    boid_id = cuda.grid(1)

    if boid_id >= old_positions.shape[0]:
        return

    bx = old_positions[boid_id, 0]
    by = old_positions[boid_id, 1]

    bvx = old_velocities[boid_id, 0]
    bvy = old_velocities[boid_id, 1]

    x_pos_avg = 0.0
    y_pos_avg = 0.0

    x_vel_avg = 0.0
    y_vel_avg = 0.0

    close_dx = 0.0
    close_dy = 0.0

    neighboring_boids = 0

    visible_distance_sq = visible_distance**2

    safe_distance_sq = safe_distance**2

    for other_id in range(old_positions.shape[0]):

        if other_id == boid_id:
            continue

        ox = old_positions[other_id, 0]
        oy = old_positions[other_id, 1]

        dx = bx - ox
        dy = by - oy

        squared_distance = dx**2 + dy**2

        if squared_distance > visible_distance_sq:
            continue

        ovx = old_velocities[other_id, 0]
        ovy = old_velocities[other_id, 1]

        if squared_distance < safe_distance_sq:
            close_dx += dx
            close_dy += dy

        else:
            x_pos_avg += ox
            y_pos_avg += oy

            x_vel_avg += ovx
            y_vel_avg += ovy

            neighboring_boids += 1

    if neighboring_boids > 0:
        x_pos_avg /= neighboring_boids
        y_pos_avg /= neighboring_boids

        x_vel_avg /= neighboring_boids
        y_vel_avg /= neighboring_boids

        nature = alignment_randoms[boid_id]

        bvx += (x_pos_avg - bx) * cohesion+ (x_vel_avg - bvx) * alignment * nature

        bvy += (y_pos_avg - by) * cohesion + (y_vel_avg - bvy) * alignment * nature

    bvx += close_dx * separation
    bvy += close_dy * separation

    if bx < left + margin_offset:
        bvx += turn_factor

    if bx > right - margin_offset:
        bvx -= turn_factor

    if by < bottom + margin_offset:
        bvy += turn_factor

    if by > top - margin_offset:
        bvy -= turn_factor

    if bx < left:
        bvx *= -rebound_factor
        bx = left + 1

    if bx > right:
        bvx *= -rebound_factor
        bx = right - 1

    if by < bottom:
        bvy *= -rebound_factor
        by = bottom + 1

    if by > top:
        bvy *= -rebound_factor
        by = top - 1

    if not point_sign == 0:
        px = point_x - bx
        py = point_y - by
        if point_sign == 1:
            px = bx - point_x
            py = by - point_y


        point_distance_sq = px * px + py * py

        if point_distance_sq < visible_distance_sq*5:
            bvx += px * point_factor
            bvy += py * point_factor

    cx = (right - left) / 2 - bx
    cy = (top - bottom) / 2 - by

    bvx += cx * 0.00002
    bvy += cy * 0.00003

    speed = (bvx**2 + bvy**2) ** 0.5

    if speed > 0.0:
        if speed < min_speed:
            bvx = (bvx / speed) * min_speed
            bvy = (bvy / speed) * min_speed

        elif speed > max_speed:
            bvx = (bvx / speed) * max_speed
            bvy = (bvy / speed) * max_speed

    positions[boid_id, 0] = int(bx + bvx)
    positions[boid_id, 1] = int(by + bvy)

    velocities[boid_id, 0] = bvx
    velocities[boid_id, 1] = bvy


class Population:
    def __init__(self, border: tuple[int, int, int, int], margin_offset: int, boid_count: int,
                 min_speed: float = 2.0, max_speed: float = 4.0, random_boid_natures: bool = False):
        self.boid_count = boid_count
        self.margin_offset = margin_offset

        self.min_speed = min_speed
        self.max_speed = max_speed

        self.random_boid_natures = random_boid_natures

        self._inst_border(border)

        self.positions = np.empty((boid_count, 2), dtype=int_dtype)

        self.velocities = np.empty((boid_count, 2), dtype=float_dtype)

        self.alignment_randoms = np.ones(boid_count, dtype=float_dtype)

        self._inst_values(random_boid_natures)

        self.d_positions = cuda.to_device(self.positions)

        self.d_velocities = cuda.to_device(self.velocities)

        self.d_alignment_randoms = cuda.to_device(self.alignment_randoms)

        self.d_new_positions = cuda.device_array_like(self.positions)

        self.d_new_velocities = cuda.device_array_like(self.velocities)

        self.threads_per_block = 256

        self.blocks_per_grid = (boid_count + self.threads_per_block - 1) // self.threads_per_block

    def _inst_values(self, randomise: bool):
        bottom, top, left, right = self.border

        rng = np.random.default_rng()

        self.positions[:, 0] = rng.uniform(left + self.margin_offset, right - self.margin_offset, self.boid_count)

        self.positions[:, 1] = rng.uniform(bottom + self.margin_offset, top - self.margin_offset, self.boid_count)

        self.velocities[:, 0] = rng.uniform(-self.min_speed, self.min_speed, self.boid_count)

        self.velocities[:, 1] = rng.uniform(-self.min_speed, self.min_speed, self.boid_count)

        if randomise:
            self.alignment_randoms[:] = rng.uniform(0.9, 1.1, size=self.boid_count)

    def _inst_border(self, border):

        self.border = border

        if not is_tuple_4(self.border):

            raise Exception("Invalid bounding box layout. ","Must be in form tuple(-y,+y,-x,+x)")

        v_abs = [abs(i) for i in self.border[0:2]]

        h_abs = [abs(i) for i in self.border[2:4]]

        if not max(v_abs) or max(v_abs) <= self.margin_offset or not max(h_abs) or max(h_abs) <= self.margin_offset:
            raise Exception("Inappropriate bounding box size.\n[-value,+value] where +value and -value cannot be 0 and abs(+value) or abs(-value) greater than margin")

    def next_frame(self, separation: float, alignment: float, cohesion: float, safe_distance: float, visible_distance: float, turn_factor: float,
                   rebound_factor: float, point_x: int, point_y: int, point_factor: float, point_sign : int):

        bottom, top, left, right = self.border

        boid_kernel[self.blocks_per_grid, self.threads_per_block](self.d_positions, self.d_velocities, self.d_new_positions, self.d_new_velocities,
                  self.d_alignment_randoms, bottom, top, left, right, self.margin_offset, self.min_speed, self.max_speed, separation,
                  alignment, cohesion, safe_distance, visible_distance, turn_factor, rebound_factor, point_x, point_y, point_factor, point_sign)

        cuda.synchronize()

        self.d_positions, self.d_new_positions = self.d_new_positions, self.d_positions

        self.d_velocities, self.d_new_velocities, = self.d_new_velocities, self.d_velocities

    def get_positions(self):
        self.d_positions.copy_to_host(self.positions)
        return self.positions

    def get_velocities(self):
        self.d_velocities.copy_to_host(self.velocities)
        return self.velocities


if __name__ == "__main__":
    test_pop = Population(border=(0, 200, 0, 200), margin_offset=50, boid_count=1000, min_speed=2.0, max_speed=4.0, random_boid_natures=True)
    positions = test_pop.get_positions()