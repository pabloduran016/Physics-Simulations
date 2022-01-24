"""
Script to run a double pendulum simulation.
Written by Pablo Duran (https://github.com/pabloduran016)
Needed pygame and pymunk libraries
"""
import pygame as pg
import pymunk as pk
from pymunk import Vec2d as Vec
import pymunk.pygame_util
from math import pi, cos, sin
from abc import ABC
from typing import Tuple, Union
import json
import random

# Constants
WIDTH, HEIGHT = SIZE = 800, 800
FPS = 60
WHITE = 255, 255, 255, 255
BLACK = 0, 0, 0, 255
TITLE = 'Double Pendulum'
GRAVITY = 0, .5

# VALUES FOR CUSTOMIZATION
AIR_FRICTION_FACTOR = 1  # A value of 1 will cause no friction

FILE_NAME = 'data.json'

MASS_1 = 1
MASS_2 = 1

RADIUS_1 = 10
RADIUS_2 = 10

SHAPE = 'poly'  # change between poly, rect, circle

BOX_RADIUS = 350
BOX_BORDER = 3
BOX_CENTER = WIDTH / 2, HEIGHT / 2
BOX_ELASTICITY = .99999  # relates to bounciness (1.0 maximum bounciness, 0.0 no bounciness) NOT RECOMMENDED TO USE 1.0
# WHICH CAUSES PROBLEMS DUE TO INACCURACIES IN THE SIMULATION

BOB_ELASTICITY = .99999  # relates to bounciness (1.0 maximum bounciness, 0.0 no bounciness) ...

PIVOT_INITIAL_POSITION = WIDTH / 2, HEIGHT / 2

LENGTH_1 = 200
LENGTH_2 = 150

INITIAL_ANGLE_1 = pi + .111  # initial angle in radians
INITIAL_ANGLE_2 = 0

INITIAL_ANGULAR_VELOCITY_1 = 0
INITIAL_ANGULAR_VELOCITY_2 = 0

# Calculation of the positions from polar coordinates to cartesian coordinates
INITIAL_POSITION_1 = PIVOT_INITIAL_POSITION[0] + LENGTH_1 * sin(INITIAL_ANGLE_1), \
                     PIVOT_INITIAL_POSITION[1] + LENGTH_1 * cos(INITIAL_ANGLE_1)
INITIAL_POSITION_2 = INITIAL_POSITION_1[0] + LENGTH_2 * sin(INITIAL_ANGLE_2), \
                     INITIAL_POSITION_1[1] + LENGTH_2 * cos(INITIAL_ANGLE_2)


def rounded(value: Union[Tuple, float], decimals: int = 0) -> Union[Tuple, float]:
    if type(value) == tuple or type(value) == Vec:
        new = []
        for val in value:
            new.append(round(val * 10 ** decimals) / (10 ** decimals))
        return tuple(new)
    elif type(value) == float:
        return round(value * 10 ** decimals) / (10 ** decimals)
    else:
        raise TypeError(f'Expected type tuple, vec or float, got {type(value)}')


# You might want to create different boxes (circular, triangular, rectangular ...)
class Box(ABC):
    """Abstrat class for a box. All types of boxes should inherit from this class"""


class CircularBox(Box):
    """
    Class to create a Circular Box to enclose the double pendulum.
    """

    def __init__(self, simulation, radius: float, center: Tuple[float, float], border: float = 1,
                 elasticity: float = 0, n_segments: int = 50):
        """
        :param simulation: reference to the simulation so adding objects to the space can be done
        :param radius: Radius fo the circular Box
        :param center: center of the box in cartesian coordinates (the origin is in the top lef coordinates and the y
        axis increases in value from top to bottom)
        :param border: border of the bounding box, the border is applied radially
        :param elasticity: elasticity of the box (0 means no bounce, 1 means perfect bounce. NOT RECOMMENDED TO USE 1)
        :param n_segments: number of segments the cicular box is made up of (resolution)
        """
        self.simulation = simulation  # variable to store a reference of the simulation
        self.body = pk.Body(body_type=pk.Body.STATIC)
        self.body.position = center
        self.shapes = []
        # construct a polygon which aproximates to a circunference
        vec = radius * Vec(1, 0).rotated_degrees(random.randint(0, 360))
        angular_dif = 360 / n_segments
        for i in range(n_segments):
            shape = pk.Segment(self.body, vec, vec.rotated_degrees(angular_dif), border)
            shape.elasticity = elasticity
            self.shapes.append(shape)
            vec = vec.rotated_degrees(angular_dif)
        self.simulation.space.add(self.body, *self.shapes)


class PolygonalBox(CircularBox):
    """
    Class to create a Polygonal Box (regular polygon) to enclose the double pendulum.
    """

    def __init__(self, simulation, n_sides: int, radius: float, center: Tuple[float, float], border: float = 1,
                 elasticity: float = 0):
        """
        :param simulation: reference to the simulation so adding objects to the space can be done
        :param n_sides: number of sides of the regular polygon
        :param radius: Radius fo the concentric circle of the polygon
        :param center: center of the box in cartesian coordinates (the origin is in the top lef coordinates and the y
        axis increases in value from top to bottom)
        :param border: border of the bounding box, the border is applied radially
        :param elasticity: elasticity of the box (0 means no bounce, 1 means perfect bounce. NOT RECOMMENDED TO USE 1)
        """
        # A polygon is just a circle with less resolution
        super().__init__(simulation, radius, center, border, elasticity, n_segments=n_sides)


class RectangularBox(Box):
    """
    Class to create a Polygonal Box (regular polygon) to enclose the double pendulum.
    """

    def __init__(self, simulation, width: float, height: float, center: Tuple[float, float], border: float = 1,
                 elasticity: float = 0):
        """
        :param simulation: reference to the simulation so adding objects to the space can be done
        :param width: width of the box
        :param height: height of the box
        :param center: center of the box in cartesian coordinates (the origin is in the top lef coordinates and the y
        axis increases in value from top to bottom)
        :param border: border of the bounding box, the border is applied radially
        :param elasticity: elasticity of the box (0 means no bounce, 1 means perfect bounce. NOT RECOMMENDED TO USE 1)
        """
        self.simulation = simulation  # variable to store a reference of the simulation
        self.body = pk.Body(body_type=pk.Body.STATIC)
        self.body.position = center
        # Pymunk doesn't work with concave shapes so it is necessary to create multiple shapes, one for each wall (4)
        self.shapes = [pk.Segment(self.body, a, b, border / 2) for a, b in [
            [(width / 2, height / 2), (width / 2, - height / 2)],
            [(width / 2, - height / 2), (- width / 2, - height / 2)],
            [(- width / 2, - height / 2), (- width / 2, height / 2)],
            [(- width / 2, height / 2), (width / 2, height / 2)],
        ]]
        for shape in self.shapes:
            shape.elasticity = elasticity
        self.simulation.space.add(self.body, *self.shapes)


class Pendulum:
    """
    Class to store all the pendulum information
    """

    def __init__(self, simulation, bob_elasticity: float = 0):
        """
        :param simulation: reference to the simulation so adding objects to the space can be done
        :param bob_elasticity: elasticity of the bobs (0 means no bounce, 1 means perfect bounce.
        NOT RECOMMENDED TO USE 1)
        """
        self.simulation = simulation  # variable to store a reference of the simulation

        self.pivot = pk.Body(body_type=pk.Body.STATIC)  # body for the pivot
        self.pivot.position = PIVOT_INITIAL_POSITION

        moment_1 = pk.moment_for_circle(mass=MASS_1, inner_radius=0, outer_radius=RADIUS_1)  # Calculation of the moment
        self.bob_1_body = pk.Body(mass=MASS_1, moment=moment_1)  # body for the first bob
        self.bob_1_body.position = Vec(*INITIAL_POSITION_1)
        # Calculating the velocity vec from the angular accleration and the radius
        self.bob_1_body.velocity = (self.bob_1_body.position - self.pivot.position).normalized().rotated(pi) * \
                                   INITIAL_ANGULAR_VELOCITY_1 * LENGTH_1
        self.bob_1_shape = pk.Circle(body=self.bob_1_body, radius=RADIUS_1)  # shape for the first bob
        self.bob_1_shape.elasticity = bob_elasticity

        # same procedure for the second bob
        moment_2 = pk.moment_for_circle(mass=MASS_2, inner_radius=0, outer_radius=RADIUS_2)
        self.bob_2_body = pk.Body(mass=MASS_2, moment=moment_2)  # body for the second bob
        self.bob_2_body.position = Vec(*INITIAL_POSITION_2)
        self.bob_2_body.velocity = (self.bob_2_body.position - self.pivot.position).normalized().rotated(pi) * \
                                   INITIAL_ANGULAR_VELOCITY_2 * LENGTH_2
        self.bob_2_shape = pk.Circle(body=self.bob_2_body, radius=RADIUS_2)  # shape for the second bob
        self.bob_2_shape.elasticity = bob_elasticity

        self.rod_1 = pk.PinJoint(self.pivot, self.bob_1_body)  # rod joining the pivot with the first bob
        self.rod_2 = pk.PinJoint(self.bob_1_body, self.bob_2_body)  # rod joining the first bob with the socnd one

        self.simulation.space.add(self.pivot, self.bob_1_body, self.bob_1_shape, self.bob_2_body, self.bob_2_shape,
                                  self.rod_1, self.rod_2)  # adding all the elements to the _physics space

    def update(self) -> None:  # method to call on every step of the simulation
        # Apply friction by multiplying the current velocity by a factor
        self.bob_1_body.velocity *= AIR_FRICTION_FACTOR
        self.bob_2_body.velocity *= AIR_FRICTION_FACTOR


class Simulation:
    """
    Class to carry out a simulation without graphics for determinate amount of time with a determinate step
    """
    running: bool = True
    space: pk.Space
    pendulum: Pendulum
    box: Box
    shape: str = SHAPE

    def __init__(self, box_radius):
        """:param box_radius: Radius of the box"""
        # STRUCTURE FOR THE DATA DICTIONARY
        # {
        #     "bob_1": {
        #         "position_cartesian": List[Tuple[float, float]],       <List of 2d vectors with x and y>
        #         "velocity_cartesian": List[Tuple[float, float]],       <List of 2d vectors with x and y>
        #         "position_polar": List[Tuple[float, float]],           <List of pairs of distance and angle (radians)>
        #         "velocity_polar": List[Tuple[float, float]],           <List of pairs of speed and angle (radians)>
        #         "kinetic_energy": List[float],                         <List of float values>
        #         "potential_energy": List[float],                       <List of float values>
        #         "mechanical_energy": List[float],                      <List of float values>
        #     },
        #     "bob_2": {
        #         "position_cartesian": List[Tuple[float, float]],
        #         "velocity_cartesian": List[Tuple[float, float]],
        #         "position_polar": List[Tuple[float, float]],
        #         "velocity_polar": List[Tuple[float, float]],
        #         "kinetic_energy": List[float],
        #         "potential_energy": List[float],
        #         "mechanical_energy": List[float],
        #     },
        # }
        self.data = {
            "bob_1": {
                "position_cartesian": [],
                "velocity_cartesian": [],
                "position_polar": [],
                "velocity_polar": [],
                "kinetic_energy": [],
                "potential_energy": [],
                "mechanical_energy": [],
            },
            "bob_2": {
                "position_cartesian": [],
                "velocity_cartesian": [],
                "position_polar": [],
                "velocity_polar": [],
                "kinetic_energy": [],
                "potential_energy": [],
                "mechanical_energy": [],
            },
        }  # DICTIONARY TO STORE ALL THE VALUES
        self.space = pk.Space()
        self.space.gravity = GRAVITY
        self.pendulum = Pendulum(self, BOB_ELASTICITY)
        self.box_radius = box_radius

        if self.shape == 'rect':
            self.box = RectangularBox(self, width=self.box_radius * 2, height=self.box_radius * 2, center=BOX_CENTER,
                                      border=BOX_BORDER, elasticity=BOX_ELASTICITY)
        elif self.shape == 'circle':
            self.box = CircularBox(self, radius=self.box_radius, center=BOX_CENTER, border=BOX_BORDER,
                                   elasticity=BOX_ELASTICITY)
        elif self.shape == 'poly':
            self.box = PolygonalBox(self, n_sides=5, radius=self.box_radius, center=BOX_CENTER, border=BOX_BORDER,
                                    elasticity=BOX_ELASTICITY)
        else:
            raise ValueError(f'Unknown shape, expected poly, rect or circle, got {self.shape}')

    def simulate(self, time: float, measurements_per_second: float) -> None:  # method to start the simulation
        assert measurements_per_second < 60, f'The maximum amount of measurements per second is 60, ' \
                                             f'got {measurements_per_second}'
        # The amount of mesurements needed to take is the amount of time multiplied by the amount of
        # measurements per second
        for rep in range(round(time * measurements_per_second)):
            # Save the current state of the simulation
            self.update_data()
            # Fast forward the simulation as much time as needed. Every iteration is 1 / 60 of a second
            for i in range(round(60 / measurements_per_second)):
                self.run()
            if not self.running:
                break

    def reset_data(self):
        self.data = {
            "bob_1": {
                "position_cartesian": [],
                "velocity_cartesian": [],
                "position_polar": [],
                "velocity_polar": [],
                "kinetic_energy": [],
                "potential_energy": [],
                "mechanical_energy": [],
            },
            "bob_2": {
                "position_cartesian": [],
                "velocity_cartesian": [],
                "position_polar": [],
                "velocity_polar": [],
                "kinetic_energy": [],
                "potential_energy": [],
                "mechanical_energy": [],
            },
        }

    def update_data(self):
        # print(self.pendulum.bob_1_body.position)
        # self.data['bob_1']["position_cartesian"].append(rounded(self.pendulum.bob_1_body.position, decimals=2))
        for bob_key, (bob, pivot) in zip(self.data,
                                         [(self.pendulum.bob_1_body, self.pendulum.pivot),
                                          (self.pendulum.bob_2_body, self.pendulum.bob_1_body)]):
            distance = (bob.position - pivot.position)
            k_energy = bob.mass * (bob.velocity.length ** 2) / 2
            p_energy = bob.mass * self.space.gravity.length * (HEIGHT - bob.position.y)
            m_energy = k_energy + p_energy
            for key, value in zip(self.data[bob_key].keys(), [
                    tuple(bob.position), tuple(bob.velocity), (distance.length, distance.angle),
                    (bob.velocity.length, bob.velocity.angle), k_energy, p_energy, m_energy]):
                self.data[bob_key][key].append(rounded(value))

    def save_data(self, filename):
        with open(filename, 'w') as f:
            f.write(json.dumps(self.data))

    def run(self) -> None:  # method to run the simulation.
        self.space.step(.5)  # step further into the simulation ( 1/60 seconds)
        self.space.step(.5)  # two small steps instead of one bigger increases precision
        self.pendulum.update()


class GraphicalSimulation(Simulation):
    """
    Class to carry out a graphical simulation
    """
    draw_options: pk.pygame_util.DrawOptions

    def __init__(self, box_radius):
        """:param box_radius: Radius of the box"""
        super().__init__(box_radius)
        pg.init()
        pg.display.set_caption(TITLE)
        self.screen = pg.display.set_mode(SIZE)
        self.font = pg.font.SysFont(name='bold', size=25)
        self.clock = pg.time.Clock()
        self.draw_options = pk.pygame_util.DrawOptions(self.screen)

    def start(self, **kwargs) -> None:  # method to start the simulation
        if 'save_data' in kwargs.keys():
            self.run_with_saving(**kwargs)
        else:
            self.run()

    def reset(self, box_radius=BOX_RADIUS) -> None:
        super().__init__(box_radius)
        self.start()

    def events(self) -> None:  # method to check events
        events = pg.event.get()
        for event in events:
            if event.type == pg.QUIT:
                self.running = False
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    self.reset()
                if event.key == pg.K_p:
                    self.shape = 'poly'
                    self.reset()
                if event.key == pg.K_c:
                    self.shape = 'circle'
                    self.reset()
                if event.key == pg.K_r:
                    self.shape = 'rect'
                    self.reset()

        keys = pg.key.get_pressed()
        if keys[pg.K_UP]:
            self.reset(self.box_radius + 3)
        if keys[pg.K_DOWN]:
            self.reset(self.box_radius - 3)

    def update(self) -> None:  # method to run one step further of the simulation
        self.space.step(.5)  # step further into the simulation
        self.space.step(.5)  # two small steps instead of one bigger increases precision
        self.pendulum.update()
        pass

    def draw(self) -> None:  # method to draw the current state of the simulation
        self.screen.fill(WHITE)
        self.space.debug_draw(self.draw_options)
        self.screen.blit(self.font.render('Press space to restart', True, BLACK), (10, 10))
        self.screen.blit(self.font.render('Press c to use a circle', True, BLACK), (10, 30))
        self.screen.blit(self.font.render('Press p to use a polygon', True, BLACK), (10, 50))
        self.screen.blit(self.font.render('Press r to use a rectangle', True, BLACK), (10, 70))
        self.screen.blit(self.font.render('Press up arrow to increase the radius', True, BLACK), (450, 10))
        self.screen.blit(self.font.render('Press down arrow to decrease the radius', True, BLACK), (450, 30))

    def run(self) -> None:  # method to show the simulation
        while self.running:
            self.clock.tick(FPS)
            self.events()
            self.update()
            self.draw()
            pg.display.flip()

    # method to show the simulation and save the data
    def run_with_saving(self, save_data: bool, filename: str, time: Union[str, float],
                        measurements_per_second: float) -> None:
        reps = 0
        frames_between_measurement = round(FPS / measurements_per_second)
        assert save_data
        if time == 'undefined':
            undefined = True
            time = 0
        else:
            undefined = False
        assert type(time) == float or type(time) == int
        while self.running and ((pg.time.get_ticks() / 1000) < time or undefined):
            self.clock.tick(FPS)
            self.events()
            self.update()
            self.draw()
            if reps == frames_between_measurement:
                self.update_data()
                reps = 0
            pg.display.flip()
            reps += 1
        self.save_data(filename)


if __name__ == '__main__':
    s = GraphicalSimulation(BOX_RADIUS)
    # start simulation
    s.start(save_data=True, filename=FILE_NAME, time='undefined', measurements_per_second=5)
