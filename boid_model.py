import math
import random
from boid import Boid2D, Boid

class Population:
    def __init__(self, border : tuple[int, int, int, int], boid_count : int, **kwargs):
        """
        :param separation:
        :type separation:
        :param cohesion:
        :type cohesion:
        :param alignment:
        :type alignment:
        :param max_speed:
        :type max_speed:
        :param boid_count:
        :type boid_count:
        :param border: [-y,+y,-x,+x]
        :type border:
        :param margin:
        :type margin:
        """
        self.boid_count = boid_count
        self.border = border

        v_abs, h_abs = [abs(i) for i in border[0:2]], [abs(i) for i in border[2:4]]
        if not max(v_abs) or not max(h_abs):
            raise Exception("Incorrect bounding box dimensions")

        if kwargs.get("fixed_seed"):
            random.seed(69)

        self.separation = 0.7 if kwargs.get("separation") is None else kwargs.get("separation")
        self.cohesion = 0.7 if kwargs.get("cohesion") is None else kwargs.get("cohesion")
        self.alignment = 0.7 if kwargs.get("alignment") is None else kwargs.get("alignment")

        self.max_speed = 4.5 if kwargs.get("max_speed") is None else kwargs.get("max_speed")
        self.frame_rate = 1/12 if kwargs.get("frame_rate") is None else kwargs.get("frame_rate")

        self.safe_distance = math.floor(max([abs(i) for i in border])*0.12) if kwargs.get("safe_distance") is None else kwargs.get("safe_distance")

        self.margin = kwargs.get("separation")
        if self.margin is None or not type(self.margin) is tuple[int,int,int,int]:
            self.margin = tuple(int(_*0.75) for _ in self.border)
        self.boid_list = None

    def load_boids(self, starting_positions : list[list[int]] | None = None):
        if starting_positions is None:
            starting_positions = [[random.randint(self.border[0], self.border[1]), random.randint(self.border[2], self.border[3])] for _ in range(self.boid_count)]

        if len(starting_positions) != self.boid_count:
            raise Exception("Incorrect number of starting positions for number of boids")

        self.boid_list = []
        for i in range(self.boid_count):
            self.boid_list.append(Boid2D(*starting_positions[i], max_speed=self.max_speed))

    def next_frame(self) -> None:
        for boid in self.boid_list:
            for other in self.boid_list:
                if other is boid:
                    continue
                boid.safe_distance(other, self.safe_distance)
            boid.apply_separation(self.separation)
            boid.update_position(self.frame_rate)

    def get_positions(self) -> tuple[list[int], list[int]]:
        y, x = zip(*[b.get_position() for b in self.boid_list])
        return x, y


if __name__ == "__main__":
    test_pop = Population((0,50,0,50), 25, frame_rate=1/5)
    test_pop.load_boids()