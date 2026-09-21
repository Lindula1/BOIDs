"""
Author : Lindula
Version : 0.1.0
Description :
BOID data structure designed with pixel coordinates in mind
Inspired by:
https://people.ece.cornell.edu/land/courses/ece4760/labs/s2021/Boids/Boids.html
"""
import math
import numpy as np

class Boid:
    def __init__(self, *args : int, **kwargs):
        """
        :param args: pass in x y z etc.. coordinates
        """
        self.dimension = len(args)

        self.pixel_position = list(args)
        self.velocity_vector = [0 for _ in range(self.dimension)]

        self.max_speed = kwargs.get("max_speed")
        self.min_speed = kwargs.get("min_speed")

    def set_position(self, *args : int):
        self.pixel_position[0 : len(args)-1] = args

    def get_position(self) -> list[int]:
        return self.pixel_position

    def set_velocity(self, *args : float, **kwargs):
        raise NotImplementedError

    def get_velocity(self) -> list[float]:
        raise NotImplementedError

    def get_abs_velocity(self):
        return sum([_**2 for _ in self.velocity_vector])**0.5

    def frame_updates(self):
        raise NotImplementedError

    def update_position(self, frame_rate : float) -> bool:
        if len(self.velocity_vector) != len(self.pixel_position):
            raise Exception("HOW?")

        vx, vy = self.get_velocity()
        speed = self.get_abs_velocity()

        if speed == 0:
            self.frame_updates()
            return False

        if speed >= self.max_speed:
            self.set_velocity((vx / speed) * self.min_speed, (vy / speed) * self.min_speed)
        if speed < self.min_speed:
            self.set_velocity((vx / speed) * self.min_speed, (vy / speed) * self.min_speed)

        self.set_position(*[math.floor(self.pixel_position[i] + self.velocity_vector[i] * frame_rate) for i in range(len(self.pixel_position))])

        self.frame_updates()

        return True

    # Debugging
    def __str__(self):
        return f"Pos : {self.get_position()}, Velocity : {self.get_abs_velocity() : .3f}"


# noinspection DuplicatedCode
class Boid2D(Boid):
    def __init__(self, x : int, y : int, max_speed : float, min_speed : float):
        super().__init__(x, y, max_speed=max_speed, min_speed=min_speed)
        # Separation parameters
        self._close_distance = [0 for _ in range(len(self.get_velocity()))]
        self._velocity_avg = [0 for _ in range(len(self.get_velocity()))]
        self._neighbouring_boids = 0
        self._position_avg = [0 for _ in range(len(self.get_position()))]

    def get_position(self) -> list[int]:
        return self.pixel_position

    def set_position(self, x : int, y : int):
        """
        :param x: pixel x location
        :param y: pixel y location
        :return:
        """
        self.pixel_position = [x, y]

    def set_velocity(self, x : float, y : float):
        self.velocity_vector = [x, y]

    def get_velocity(self) -> list[float]:
        return self.velocity_vector

    def get_distance(self, other : Boid) -> float:
        return sum([(self.get_position()[i] - other.get_position()[i])**2 for i in range(len(self.get_position()))])**0.5


    def at_safe_distance(self, other: Boid, safe_distance : float, distance_to_other : float) -> bool:
        vector_to_other = [self.get_position()[i] - other.get_position()[i] for i in range(len(self.get_position()))]

        if distance_to_other > safe_distance: # Neighbouring
            self._neighbouring_boids += 1
            self._velocity_avg = [self._velocity_avg[i] + other.get_velocity()[i] for i in range(len(self._velocity_avg))]
            self._position_avg = [self._position_avg[i] + other.get_position()[i] for i in range(len(self._position_avg))]
            return True
        self._close_distance = [self._close_distance[i] + vector_to_other[i] for i in range(len(self._close_distance))]
        return False

    def _average_velocity(self) -> bool:
        if self._neighbouring_boids < 1:
            return False

        self._velocity_avg = [self._velocity_avg[i] / self._neighbouring_boids for i in range(len(self._velocity_avg))]
        return True

    def _average_position(self) -> bool:
        if self._neighbouring_boids < 1:
            return False

        self._position_avg = [self._position_avg[i] / self._neighbouring_boids for i in range(len(self._position_avg))]
        return True

    def apply_cohesion(self, centering_factor : float):
        if not self._average_position():
            return None

        px, py = self.get_position()
        vx, vy = self.get_velocity()
        apx, apy = self._position_avg

        self.set_velocity(vx + (apx - px)*centering_factor, vy + (apy - py)*centering_factor)
        return None

    def apply_alignment(self, matching_factor : float):
        if not self._average_velocity():
            return None

        vx, vy = self.get_velocity()
        avx, avy = self._velocity_avg

        self.set_velocity(vx + (avx - vx)*matching_factor, vy + (avy - vy)*matching_factor)
        return None

    def apply_separation(self, avoid_factor : float):
        vx, vy = self.get_velocity()
        dx, dy = self._close_distance

        if dx == 0 and dy == 0:
            return None

        self.set_velocity(vx + dx * avoid_factor, vy + dy * avoid_factor)

        return None

    def margin_avoidance(self, turn_factor : float, margin : tuple[int,int,int,int]):
        top, bottom, left, right = margin
        x, y = self.get_position()
        vx, vy = self.get_velocity()
        if x < left:
            self.set_velocity(vx + turn_factor, vy)
        if x > right:
            self.set_velocity(vx - turn_factor, vy)
        if y > bottom:
            self.set_velocity(vx, vy - turn_factor)
        if y < top:
            self.set_velocity(vx, vy + turn_factor)

    def border_rebound(self, rebound_factor : float, border : tuple[int,int,int,int]):
        top, bottom, left, right = border
        x, y = self.get_position()
        vx, vy = self.get_velocity()
        if x < left:
            self.set_velocity(-vx * rebound_factor, vy)
            self.set_position(left+1, y)
        if x > right:
            self.set_velocity(-vx * rebound_factor, vy)
            self.set_position(right-1, y)
        if y > bottom:
            self.set_velocity(vx, -vy * rebound_factor)
            self.set_position(x, bottom-1)
        if y < top:
            self.set_velocity(vx, -vy * rebound_factor)
            self.set_position(x, top+1)

    def point_repel(self, point_x : int, point_y : int, visible_distance : float, repel_factor : float):
        x, y = self.get_position()
        if ((x - point_x)**2 + (y-point_y)**2)**0.5 > visible_distance:
            return None

        vx, vy = self.get_velocity()
        velocity_from_point = [vx + (x - point_x)*repel_factor, vy +(y - point_y)*repel_factor]
        self.set_velocity(*velocity_from_point)
        return None

    def frame_updates(self):
        self._close_distance = [0 for _ in range(len(self.get_velocity()))]
        self._velocity_avg = [0 for _ in range(len(self.get_velocity()))]
        self._position_avg = [0 for _ in range(len(self.get_position()))]
        self._neighbouring_boids = 0



if __name__ == "__main__":
    test = Boid2D(1, 2)
    test.set_velocity(23.5, 13.5)
    print(test)