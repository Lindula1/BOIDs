from boid import Boid2D, Boid

class Population:
    def __init__(self, separation, cohesion, alignment, max_speed : float):
        self.separation = separation
        self.cohesion = cohesion
        self.alignment = alignment
        self.max_speed = max_speed