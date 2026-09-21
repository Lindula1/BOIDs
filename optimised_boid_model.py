import numpy as np

rng = np.random.default_rng()
int_dtype = np.int32
float_dtype = np.float16

def is_tuple_4(iterable) -> bool:
    return isinstance(iterable, tuple) and len(iterable) == 4 and all(isinstance(x, int) and not isinstance(x, bool) for x in iterable)

class Population:
    def __init__(self, border: tuple[int, int, int, int], margin_offset : int, boid_count: int,
                 min_speed : float = 2.0, max_speed : float = 4.0, random_boid_natures : bool = False):
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

    def _inst_values(self, randomise : bool):
        bottom, top, left, right = self.border
        self.positions[:, 0] = rng.uniform(left + self.margin_offset, right - self.margin_offset, self.boid_count)
        self.positions[:, 1] = rng.uniform(bottom + self.margin_offset, top - self.margin_offset, self.boid_count)
        self.velocities[:, 0] = rng.uniform(-self.min_speed, self.min_speed, self.boid_count)
        self.velocities[:, 1] = rng.uniform(-self.min_speed, self.min_speed, self.boid_count)

        if randomise:
            self.alignment_randoms[:] = rng.uniform(0.9, 1.1, size=self.boid_count)

    def _inst_border(self, border):
        self.border = border

        if not is_tuple_4(self.border):
            raise Exception("Invalid bounding box layout. Must be in form tuple(-y,+y,-x,+x)")

        v_abs, h_abs = [abs(i) for i in self.border[0:2]], [abs(i) for i in self.border[2:4]]
        if not max(v_abs) or max(v_abs) <= self.margin_offset or not max(h_abs) or max(h_abs) <= self.margin_offset:
            raise Exception("Inappropriate bounding box size.\n[-value,+value] where +value and -value cannot be 0 and abs(+value) or abs(-value) greater than 50")

    def bucket_boids(self):
        self.positions = np.sort(self.positions)

    def next_frame(self, separation : float, alignment : float, cohesion : float, safe_distance : float, visible_distance : float, turn_factor : float, rebound_factor : float, point_x : int, point_y : int, point_factor : float):
        old_positions = self.positions.copy()
        old_velocities = self.velocities.copy()
        for boid_id in range(self.positions.shape[0]):
            x_pos_avg = y_pos_avg = x_vel_avg = y_vel_avg = close_dx = close_dy = float_dtype(0)
            neighboring_boids = int_dtype(0)

            bx, by = old_positions[boid_id]
            bvx, bvy = old_velocities[boid_id]

            for other_id in range(self.positions.shape[0]):

                if other_id == boid_id:
                    continue

                ox, oy = old_positions[other_id]

                dx, dy = bx - ox, by - oy
                squared_distance = dx**2 + dy**2

                if squared_distance > visible_distance**2:
                    continue


                ovx, ovy = old_velocities[other_id]

                if squared_distance < safe_distance**2:
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

                bvx = bvx + (x_pos_avg - bx) * cohesion + (x_vel_avg - bvx) * alignment * self.alignment_randoms[boid_id]
                bvy = bvy + (y_pos_avg - by) * cohesion + (y_vel_avg - bvy) * alignment * self.alignment_randoms[boid_id]

            bvx += close_dx * separation
            bvy += close_dy * separation

            bottom, top, left, right = self.border
            if bx < left + self.margin_offset:
                bvx += turn_factor
            if bx > right - self.margin_offset:
                bvx -= turn_factor
            if by < bottom + self.margin_offset:
                bvy += turn_factor
            if by > top - self.margin_offset:
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

            speed = np.sqrt(bvx ** 2 + bvy ** 2, dtype=float_dtype)

            px = bx - point_x
            py = by - point_y

            if (px**2 + py**2) < visible_distance**2:
                bvx += px * point_factor
                bvy += py * point_factor

            if speed < self.min_speed:
                bvx = (bvx / speed) * self.min_speed
                bvy = (bvy / speed) * self.min_speed
            if speed > self.max_speed:
                bvx = (bvx / speed) * self.max_speed
                bvy = (bvy / speed) * self.max_speed

            self.positions[boid_id, :] = np.array([bx + bvx, by + bvy], dtype=int_dtype)
            self.velocities[boid_id, :] = np.array([bvx, bvy], dtype=float_dtype)


    def get_positions(self):
        return self.positions

if __name__ == "__main__":
    test_pop = Population((0,200,0,200),50,20, random_boid_natures=True)
    test_pop.next_frame(0.05, 0.05, 0.0005, 2, 20, 0.2, 0.7)
    print(test_pop.get_positions())