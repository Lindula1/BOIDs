import math
import random
from boid import Boid2D, Boid

def is_tuple_4(iterable) -> bool:
    return isinstance(iterable, tuple) and len(iterable) == 4 and all(isinstance(x, int) and not isinstance(x, bool) for x in iterable)


class Population:
    def __init__(self, border : tuple[int, int, int, int], boid_count : int, **kwargs):
        """
        :param boid_count:
        :type boid_count: int
        :param border: [-y,+y,-x,+x]
        :type border: tuple
        :param margin: [-y,+y,-x,+x]
        :type margin: tuple
        """
        self.boid_count = boid_count
        if boid_count <= 0:
            raise Exception("Invalid boid count", boid_count)

        self._inst_border(border)

        self._inst_margin(kwargs)

        if kwargs.get("fixed_seed"):
            random.seed(69)

        self.separation = 0.05 if kwargs.get("separation") is None else kwargs.get("separation")
        self.cohesion = 0.05 if kwargs.get("cohesion") is None else kwargs.get("cohesion")
        self.alignment = 0.05 if kwargs.get("alignment") is None else kwargs.get("alignment")

        self.max_speed = 4.0 if kwargs.get("max_speed") is None else kwargs.get("max_speed")
        self.min_speed = 2.0 if kwargs.get("min_speed") is None else kwargs.get("min_speed")
        self.frame_rate = 1/12 if kwargs.get("frame_rate") is None else kwargs.get("frame_rate")

        self.safe_distance = max([abs(i) for i in border])*0.10 if kwargs.get("safe_distance") is None else kwargs.get("safe_distance")
        self.visible_distance = max([abs(i) for i in border])*0.10 if kwargs.get("visible_distance") is None else kwargs.get("visible_distance")
        self.turn_factor = 0.2 if kwargs.get("turn_factor") is None else kwargs.get("turn_factor")
        self.rebound_factor = 0.7 if kwargs.get("rebound_factor") is None else kwargs.get("rebound_factor")

        self.boid_list = None

    def _inst_margin(self, kwargs) -> None:
        self.margin = kwargs.get("margin")
        if self.margin is None:
            self.margin = tuple(int(self.border[_] + 50 * (-1) ** _) for _ in range(len(self.border)))
            return None

        if not is_tuple_4(self.margin):
            raise Exception("Invalid margin layout. Must be in the form tuple(-y,+y,-x,+x)")

        v_abs, h_abs = [abs(i) for i in self.margin[0:2]], [abs(i) for i in self.margin[2:4]]
        if not max(v_abs) or not max(h_abs):
            raise Exception("Inappropriate margin size")

        return None

    def _inst_border(self, border):
        self.border = border

        if not is_tuple_4(self.border):
            raise Exception("Invalid bounding box layout. Must be in the form tuple(-y,+y,-x,+x)")

        v_abs, h_abs = [abs(i) for i in self.border[0:2]], [abs(i) for i in self.border[2:4]]
        if not max(v_abs) or max(v_abs) <= 50 or not max(h_abs) or max(h_abs) <= 50:
            raise Exception("Inappropriate bounding box size.\n[-value,+value] where +value and -value cannot be 0 and abs(+value) or abs(-value) greater than 50")

    def load_boids(self, starting_positions : list[list[int]] | None = None):
        if starting_positions is None:
            starting_positions = [[random.randint(self.margin[2], self.margin[3]), random.randint(self.margin[0], self.margin[1])] for _ in range(self.boid_count)]

        if len(starting_positions) != self.boid_count:
            raise Exception("Incorrect number of starting positions for number of boids")

        self.boid_list = []
        for i in range(self.boid_count):
            self.boid_list.append(Boid2D(*starting_positions[i], max_speed=self.max_speed, min_speed=self.min_speed))
            self.boid_list[i].set_velocity(*[random.random()*self.max_speed*-1**random.randint(0,1) for i in range(len(starting_positions[i]))])

    def next_frame(self) -> None:
        for boid in self.boid_list:
            for other in self.boid_list:
                if other is boid:
                    continue
                distance = boid.get_distance(other)
                boid.at_safe_distance(other, self.safe_distance, distance)
                boid.is_neighbouring(other, self.safe_distance, self.visible_distance, distance)
            boid.apply_separation(self.separation)
            boid.apply_alignment(self.alignment)
            boid.apply_cohesion(self.cohesion)
            boid.margin_avoidance(self.turn_factor, self.margin)
            boid.border_rebound(self.rebound_factor, self.border)
            boid.update_position(self.frame_rate)

    def get_positions(self, row_major : bool = False) -> tuple[list[int], list[int]] | list[list[int]]:
        if row_major:
            return [b.get_position() for b in self.boid_list]

        x, y = zip(*[b.get_position() for b in self.boid_list])
        return x, y


if __name__ == "__main__":
    test_pop = Population((0,100,0,100), 25, frame_rate=1/5)
    test_pop.load_boids()