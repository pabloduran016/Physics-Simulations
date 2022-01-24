"""
Script to
Written by Pablo Duran (https://github.com/pabloduran016)
Needed pygame and pymunk libraries
"""
import pygame as pg
import pymunk as pk
import pymunk.pygame_util


# Constants
WIDTH, HEIGHT = SIZE = 800, 800
FPS = 60
WHITE = 255, 255, 255, 255
TITLE = ''
GRAVITY = 0, 1


# Class to carry out the simulation
class Simulation:
    running: bool = True
    space: pk.Space
    draw_options: pk.pygame_util.DrawOptions

    def __init__(self):
        pg.init()
        pg.display.set_caption(TITLE)
        self.screen = pg.display.set_mode(SIZE)
        self.clock = pg.time.Clock()
        self.initialize_physics()

    def initialize_physics(self) -> None:
        self.space = pk.Space()
        self.space.gravity = GRAVITY
        self.draw_options = pk.pygame_util.DrawOptions(self.screen)

    def start(self) -> None:  # method to start the simulation
        self.run()

    def events(self) -> None:  # method to check events
        events = pg.event.get()
        for event in events:
            if event.type == pg.QUIT:
                self.running = False

    def update(self) -> None:  # method to run one step further of the simulation
        self.space.step(.5)  # step further into the simulation
        self.space.step(.5)  # two small steps instead of one bigger increases precision
        pass

    def draw(self) -> None:  # method to draw the current state of the simulation
        self.screen.fill(WHITE)
        self.space.debug_draw(self.draw_options)

    def run(self) -> None:  # method to show the simulation
        while self.running:
            self.clock.tick(FPS)
            self.events()
            self.update()
            self.draw()
            pg.display.flip()


if __name__ == '__main__':
    s = Simulation()
    # start simulation
    s.start()
