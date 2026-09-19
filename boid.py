"""
Author : Lindula
Version : 0.1.0
Description :
Boid data structure designed with pixel coordinates in mind
https://people.ece.cornell.edu/land/courses/ece4760/labs/s2021/Boids/Boids.html
"""
import math

class Boid:
    def __init__(self, *args : int):
        """
        :param args: pass in x y z etc.. coordinates
        """
        self.pixel_position = args
        self.velocity_vector = None

    def set_position(self, *args : int):
        self.pixel_position[0 : len(args)-1] = args

    def get_position(self) -> tuple[int,...]:
        return self.pixel_position

    def set_velocity(self, *args : float, **kwargs):
        raise NotImplementedError

    def get_velocity(self, **kwargs) -> list[float,...]:
        """
        :param kwargs:
        :return: pixels per frame
        """
        raise NotImplementedError

    def get_abs_velocity(self):
        return sum([_**2 for _ in self.velocity_vector])**0.5

    def factor_updates(self):
        raise NotImplementedError

    def update_position(self):
        if len(self.velocity_vector) != len(self.pixel_position):
            raise Exception("HOW?")
        self.velocity_vector = [self.velocity_vector[i] * self.velocity_vector[i] for i in range(len(self.velocity_vector))]
        self.factor_updates()

    # Debugging
    def __str__(self):
        return f"Pos : {self.get_position()}, Velocity : {self.get_abs_velocity() : .3f}"

class Boid2D(Boid):
    def __init__(self, x : int, y : int):
        super().__init__(y, x)
        # Separation parameters
        self._close_distance = None

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

    def separate_boids(self, other: Boid, separation: float):
        self._close_distance =

    def factor_updates(self):
        self._close_distance = None

if __name__ == "__main__":
    test_boid = Boid2D(1, 2)
    test_boid.set_velocity(23.5, 13.5)
    print(test_boid)