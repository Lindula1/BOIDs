import pygame
from matplotlib import pyplot as plt
from boid_model import Population
import random

if __name__ == "__main__":
    test_pop = Population((0, 50, 0, 50), 25, frame_rate=1 / 5, fixed_seed=True)
    test_pop.load_boids()

    boid_x, boid_y = test_pop.get_positions()

    plt.subplot(1,2,1)
    plt.scatter(boid_x, boid_y, marker='o')

    test_pop.next_frame()
    boid_x, boid_y = test_pop.get_positions()

    plt.subplot(1,2,2)
    plt.scatter(boid_x, boid_y, marker='o')
    plt.show()
    # print(boid_positions)