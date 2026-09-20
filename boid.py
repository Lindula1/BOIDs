"""
Author : Lindula
Version : 0.1.0
Description :
BOID data structure designed with pixel coordinates in mind
Inspired by:
https://people.ece.cornell.edu/land/courses/ece4760/labs/s2021/Boids/Boids.html
"""
import math

class Boid:
    def __init__(self, *args : int, **kwargs):
        """
        :param args: pass in x y z etc.. coordinates
        """
        self.pixel_position = list(args)
        self.velocity_vector = [0 for _ in range(len(args))]
        self.max_speed = kwargs.get("max_speed")

    def set_position(self, *args : int):
        self.pixel_position[0 : len(args)-1] = args

    def get_position(self) -> list[int]:
        return self.pixel_position

    def set_velocity(self, *args : float, **kwargs):
        raise NotImplementedError

    def get_velocity(self, **kwargs) -> list[float]:
        """
        :param kwargs:
        :return: pixels per frame
        """
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
        min_speed = self.max_speed**0.5

        if speed == 0:
            return False

        if speed >= self.max_speed:
            self.set_velocity((vx / speed)*self.max_speed, (vy/speed)*min_speed)
        if speed < min_speed:
            self.set_velocity((vx / speed) * min_speed, (vy / speed) * min_speed)
        self.set_position(*[math.ceil(self.pixel_position[i] + self.velocity_vector[1-i] * frame_rate) for i in range(len(self.pixel_position))])
        self.frame_updates()

        return True

    # Debugging
    def __str__(self):
        return f"Pos : {self.get_position()}, Velocity : {self.get_abs_velocity() : .3f}"

class Boid2D(Boid):
    def __init__(self, x : int, y : int, max_speed : float):
        super().__init__(y, x, max_speed=max_speed)
        # Separation parameters
        self._close_distance = [0 for _ in range(len(self.get_velocity()))]

    def set_position(self, y : int, x : int):
        """
        :param y: pixel y location
        :param x: pixel x location
        :return:
        """
        self.pixel_position = [y, x]

    def set_velocity(self, *args, cartesian : bool = True):
        """
        :param args: velocity passed in as r & theta (radians) or vx and vy IN ORDER
        :param cartesian:
        :return: None
        """
        if cartesian:
            velocity = [args[0], args[1]]
        else:
            velocity = [args[0]*math.cos(args[1]), args[0]*math.sin(args[1])]

        self.velocity_vector = velocity

    def get_velocity(self, cartesian : bool = True) -> list[float]:
        if not cartesian:
            y, x = self.velocity_vector
            return [self.get_abs_velocity(), float(math.atan(y/x))]

        return self.velocity_vector

    def safe_distance(self, other: Boid, safe_distance : float) -> bool:
        vector_to_other = [self.get_position()[i] - other.get_position()[i] for i in range(len(self.get_position()))]
        distance_to_other = sum([_**2 for _ in vector_to_other])**0.5
        if distance_to_other >= safe_distance:
            return True
        self._close_distance = [vector_to_other[i] + self._close_distance[i] for i in range(len(self._close_distance))]
        return False

    def apply_separation(self, avoid_factor : float):
        vx, vy = self.get_velocity()
        dy, dx = self._close_distance
        print(dy, dx)

        self.set_velocity(vx+dx*avoid_factor, vy+dy*avoid_factor)

    def margin_avoidance(self, turn_factor : float, margin : tuple[int,int,int,int]):
        top, bottom, left, right = margin
        y, x = self.get_position()
        vx, vy = self.get_velocity()
        if x < left:
            self.set_velocity(vx + turn_factor, vy)
        if x > right:
            self.set_velocity(vx - turn_factor, vy)
        if y > bottom:
            self.set_velocity(vx, vy - turn_factor)
        if y < top:
            self.set_velocity(vx, vy - turn_factor)

    def frame_updates(self):
        self._close_distance = [0 for _ in range(len(self.get_velocity()))]

if __name__ == "__main__":
    test = Boid2D(1, 2)
    test.set_velocity(23.5, 13.5)
    print(test)