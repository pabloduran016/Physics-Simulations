"""
Script to simulate rings and rods
Written by Pablo Duran (https://github.com/pabloduran016)
Needed pygame, pymunk, PyOpenGL, PyOpenGL-accelerate, numpy libraries
"""
import json
import time
from abc import ABC
from dataclasses import dataclass
from typing import Optional, Callable, Union, List, Dict
from typing import Tuple

import numpy as np
import pygame as pg
import OpenGL.GL as gl
import OpenGL.GLU as glu
from numpy import cross, transpose, array, sign, arccos, sin, cos
from numpy.linalg import norm
from pygame.locals import *


colors = {
    'WHITE': (1, 1, 1),
    'BLACK': (0, 0, 0),
    'RED': (1, 0, 0),
    'GREEN': (0, 1, 0),
    'BLUE': (0, 0, 1),
}

ARRAY_KEYS = {"vertices"}
POSITION_KEYS = {"vertices", "edges", "surfaces", "normals"}
COLOR_KEYS = {"edge_color", "surface_color"}

Vec3d = Union[Tuple[float, float, float], np.ndarray, List[float]]
Vertices = Union[np.ndarray, Tuple[Vec3d, ...]]
Edge = Tuple[int, int]
ColorType = Tuple[float, float, float]
Surface = Tuple[int, int, int, int]
Normal = Tuple[float, float, float]
ShapeType = Dict[str, Union[Vec3d, Vertices, Tuple[Edge], Tuple[Surface], ColorType]]


class Shapes:
    @staticmethod
    def cube(vertices: Vertices, edges: Tuple[Edge, ...], surfaces: Tuple[Surface, ...],
             surface_colors: Tuple[ColorType, ...]) -> None:
        gl.glBegin(gl.GL_QUADS)
        # Draw surfaces
        for i, surface in enumerate(surfaces):
            gl.glColor3fv(surface_colors[i])
            for vertex in surface:
                gl.glVertex3fv(vertices[vertex])
        gl.glEnd()

        # Draw vertex and edges
        gl.glBegin(gl.GL_LINES)
        for edge in edges:
            for vertex in edge:
                gl.glVertex3fv(vertices[vertex])
        gl.glEnd()

    @staticmethod
    def custom(vertices: Vertices, edges: Tuple[Edge, ...], surfaces: Tuple[Surface, ...], edge_color: ColorType = None,
               surface_color: ColorType = None, position: Vec3d = None, orientation: Vec3d = None, transform: Vec3d = None,
               normals: Tuple[Normal, ...] = None, draw_edges: bool = False, **_) -> None:
        if transform is not None:
            vertices = apply_transform(vertices, transform)
        if orientation is not None:
            rot_m = get_rotation_matrix(norm(orientation), orientation)
            vertices = apply_rot_matrix(vertices, rot_m)  # rotate the ring to the desired orientation
        if position is not None:
            vertices = apply_position_change(vertices, position)
        gl.glBegin(gl.GL_QUADS)
        # Draw surfaces
        if surface_color is not None:
            gl.glColor3fv(surface_color)
        for i, surface in enumerate(surfaces):
            if normals is not None:
                gl.glNormal3fv(normals[i])
            for vertex in surface:
                gl.glVertex3fv(vertices[vertex])
        gl.glEnd()
        # Draw vertex and edges
        if draw_edges:
            gl.glBegin(gl.GL_LINES)
            if edge_color is not None:
                gl.glColor3fv(edge_color)
            for i, edge in enumerate(edges):
                for j, vertex in enumerate(edge):
                    gl.glVertex3fv(vertices[vertex])
            gl.glEnd()

    @staticmethod
    def ring(vertices: Vertices, edges: Tuple[Edge, ...], surfaces: Tuple[Surface, ...], edge_color: ColorType = None,
             surface_color: ColorType = None, position: Vec3d = None, orientation: Vec3d = None, transform: Vec3d = None,
             normals: Tuple[Normal, ...] = None, draw_edges: bool = False, **_) -> None:
        if transform is not None:
            vertices = apply_transform(vertices, transform)
        gl.glPushMatrix()
        gl.glTranslatef(*position)
        gl.glRotatef(from_rad_to_deg(norm(orientation)), *orientation)
        gl.glBegin(gl.GL_QUADS)
        # Draw surfaces
        if surface_color is not None:
            gl.glColor3fv(surface_color)
        for i, surface in enumerate(surfaces):
            if normals is not None:
                gl.glNormal3fv(normals[i])
            for vertex in surface:
                v = vertices[vertex]
                gl.glVertex3fv(v)
        gl.glEnd()
        # Draw vertex and edges
        if draw_edges:
            gl.glBegin(gl.GL_LINES)
            if edge_color is not None:
                gl.glColor3fv(edge_color)
            for i, edge in enumerate(edges):
                for j, vertex in enumerate(edge):
                    v = vertices[vertex]
                    gl.glVertex3fv(v)
            gl.glEnd()
        gl.glPopMatrix()

    @staticmethod
    def rod(vertices: Vertices, edges: Tuple[Edge, ...], surfaces: Tuple[Surface, ...], edge_color: ColorType = None,
            surface_color: ColorType = None, normals: Tuple[Normal, ...] = None, position_1: Vec3d = None,
            position_2: Vec3d = None, draw_edges: bool = False, **_) -> None:

        assert type(position_1) == np.ndarray
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE, surface_color)
        rod = position_1 - position_2
        z = np.array([0.0, 0.0, 1.0])
        # the rotation axis is the cross product between Z and rod
        axis = np.cross(z, rod)
        length = (np.dot(rod, rod)) ** .5
        # get the angle using a dot product
        angle = 180.0 / np.pi * np.arccos(np.dot(z, rod) / length)

        gl.glPushMatrix()
        gl.glTranslatef(*position_1)

        gl.glRotatef(angle, *axis)
        vertices = apply_transform(vertices, [1, 1, np.linalg.norm(rod)])
        gl.glBegin(gl.GL_QUADS)
        # Draw surfaces
        if surface_color is not None:
            gl.glColor3fv(surface_color)
        for i, surface in enumerate(surfaces):
            if normals is not None:
                gl.glNormal3fv(normals[i])
            for vertex in surface:
                gl.glVertex3fv(vertices[vertex])
        gl.glEnd()
        # Draw vertex and edges
        if draw_edges:
            gl.glBegin(gl.GL_LINES)
            if edge_color is not None:
                gl.glColor3fv(edge_color)
            for i, edge in enumerate(edges):
                for j, vertex in enumerate(edge):
                    gl.glVertex3fv(vertices[vertex])
            gl.glEnd()
        gl.glPopMatrix()

    @staticmethod
    def ground(vertices: Vertices, edges: Tuple[Edge, ...], color: ColorType) -> None:
        gl.glBegin(gl.GL_QUADS)
        # Draw surfaces
        gl.glColor3fv(color)
        for edge in edges:
            for vertex in edge:
                gl.glVertex3fv(vertices[vertex])
        gl.glEnd()

        # Draw vertex and edges
        gl.glBegin(gl.GL_LINES)
        for edge in edges:
            for vertex in edge:
                gl.glVertex3fv(vertices[vertex])
        gl.glEnd()

    @staticmethod
    def sphere(position: Vec3d, radius: float = 1.5):
        gl.glPushMatrix()
        gl.glTranslatef(*position)
        gl.glColor3f(1, 0, 1)
        glu.gluSphere(glu.gluNewQuadric(), radius, 16, 16)  # quad, radius, slices (longitude), stacks (latitude)
        gl.glPopMatrix()

    @staticmethod
    def line(position_a: Vec3d, position_b: Vec3d):
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3fv(position_a)
        gl.glVertex3fv(position_b)
        gl.glEnd()


def timer(function: Callable):
    def wrapper(*args, **kwargs):
        now = time.time()
        function(*args, **kwargs)
        print(f'Took: {(time.time() - now) * 1000:.02f}ms')

    return wrapper


def apply_position_change(vertices: Vertices, position: Vec3d) -> Vertices:
    assert type(position) == np.ndarray
    return np.array([vertex + position for vertex in vertices])


def load_shape_from_json(json_dir: str) -> ShapeType:
    """
    Function to load a shape from a .json file.
    :param json_dir: path to the json file
    :return: Dictionary with all the info about the shape
    """
    with open(json_dir, 'r') as f:
        result = json.load(f)  # Load the rod.json file into a variable
    for key, value in result.items():  # convert the ilst in the variable to tuples for optimization
        if key in COLOR_KEYS:
            result[key] = colors[result[key]]
            continue
        elif key in ARRAY_KEYS:
            result[key] = np.array(result[key])
        elif key in POSITION_KEYS:
            for index, item in enumerate(value):
                value[index] = item
            result[key] = result[key]
    if 'initial_transform' in result:
        result['vertices'] = apply_transform(result['vertices'], result['initial_transform'])
    if 'rotation' in result:
        result['vertices'] = apply_rot_matrix(result['vertices'], get_rotation_matrix(result['rotation']['angle'],
                                                                                      result['rotation']['axis']))
    return result


def from_rad_to_deg(angle: float) -> float:
    """
    Function to translate from radians to degrees
    :param angle: angle in rads
    :return: angle in degrees
    """
    return angle * 180 / np.pi


def angle_between(vec_a: Vec3d, vec_b: Vec3d) -> float:
    """
    Function to return the angle between to vectors
    :param vec_a: First vector
    :param vec_b: Secon vector
    :return: angle in rads
    """
    return np.arccos((np.dot(vec_a, vec_b)) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


def load_shape_from_prefab(prefab: ShapeType) -> ShapeType:
    return prefab.copy()


def get_rotation_matrix(a: float, axis: Vec3d) -> np.ndarray:
    """
    Function to generate a rotation matrix by a given angle and an axis of rotation
    :param a: angle in radians
    :param axis: axis of rotation in the form of a vector (x, y, z)
    :return Rotation matrix 3x3 in the form of a numpy array
    """
    if np.linalg.norm(axis) == 0:
        return np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]]).astype('float64')
    elif np.linalg.norm(axis) != 1:
        axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    rot_matrix = (
        np.array(
            [[cos(a) + x ** 2 * (1 - cos(a)), x * y * (1 - cos(a)) - z * sin(a), x * z * (1 - cos(a)) + y * sin(a)],
             [y * x * (1 - cos(a)) + z * sin(a), cos(a) + y ** 2 * (1 - cos(a)), y * z * (1 - cos(a)) - x * sin(a)],
             [z * x * (1 - cos(a)) - y * sin(a), z * y * (1 - cos(a)) + x * sin(a), cos(a) + z ** 2 * (1 - cos(a))]])
    )
    return rot_matrix


def apply_rot_matrix(vertices: Vertices, rotation_matrix: np.ndarray) -> Vertices:
    """
    Function to rotate a vertices marix by a fiven rotation matrix
    :param vertices: List of vertices in format (x, y, z)
    :param rotation_matrix: Rotaino matrix in the format of a numppy array
    :return: List of rotated vertices in format (x, y, z)
    """
    result = vertices @ rotation_matrix
    return result


def apply_transform(vertices: Vertices, factor: Vec3d) -> Vertices:
    """
    Function to rotate a vertices marix by a fiven rotation matrix
    :param vertices: List of vertices in format (x, y, z)
    :param factor: 3d Vector indicating the multiplication factors of each axis
    :return: List of rotated vertices in format (x, y, z)
    """
    assert type(vertices) == np.ndarray
    result = vertices * factor
    return result


_R = 8  # Amount of parameter of each ring
_P = 11  # Amount of parameters of each link

RINGS_AND_LINKS_JSON_DIR = 'rings_and_rods/objects_data/rings_and_links.json'


class PhysicObject(ABC):
    """Base class for all physical objects"""


@dataclass
class Ring(PhysicObject):
    pos: np.ndarray
    vel: np.ndarray
    ori: np.ndarray
    ori_vel: np.ndarray
    mass: float
    radius: float
    center_mass: Optional[np.ndarray] = None
    rotation_matrix: np.ndarray = np.zeros((3, 3))


class BaseLink(PhysicObject):
    """Class refering to all types of links"""


@dataclass
class Link:
    """Class refering to a link"""
    theta: float
    theta_vel: float
    mass: float
    rod_l_0: float
    spring_c: float
    torque_c: float
    friction_c: float
    pos: np.ndarray = np.zeros((3,))
    force: np.ndarray = np.zeros((3,))
    rotation_matrix: np.ndarray = np.zeros((3, 3))
    torque_vector: np.ndarray = np.zeros((3,))


class EmptyLink(BaseLink):
    """Class refering to an empty link"""


def load_rings_and_links_from_json(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, 'r') as f:
        result = json.load(f)
    rings = array([Ring(*ring) for ring in result['rings']])
    links_l = []
    for row in result['links']:
        links_l.append([Link(*link) if link != '_' else EmptyLink() for link in row])
    links = array(links_l)
    for ring in rings:
        ring.pos = array(ring.pos).astype('float64')
        ring.vel = array(ring.vel).astype('float64')
        ring.ori = array(ring.ori).astype('float64')
        ring.ori_vel = array(ring.ori_vel).astype('float64')

    return rings, links


class Physics:
    time_step: float
    n_rings: int  # Number of rings in the simulation

    _links: np.ndarray
    _rings: np.ndarray

    def __init__(self, time_step: float, n_rings: int = 2):
        self.time_step = time_step
        self.n_rings = n_rings
        self._rings, self._links = load_rings_and_links_from_json(RINGS_AND_LINKS_JSON_DIR)

    @property
    def links(self):
        """
        - Links between rings, the rods, are defined by an N_RINGSxN_RINGS matrix
            - Rows represent the ring in question
            - Columns represent which ring is being connected to
            - The ones which represent no connection are object of class EmptyLink
            - Represnts objects of class Link with P parameters whic are the following:
                - theta: float: angle around the ring the connection is being made at, measured in radians
                - theta_vel: float: angular velocity, measured in radians per second
                - mass: float: effective point-mass at that end of the rod
                - rod_l_0: float: rod rest length
                - spring_c: float: rod spring constant
                - torque_c: float: link torque constant
                - friction_c: float: link friction coeficient
                - pos: np.ndarray: position of the link
                - force: np.ndarray: links force vector
                - rotation_matrix: np.ndarray: rotation matrix
                - torque_vector: np.ndarray: links torque vector
        """
        return self._links

    @property
    def rings(self):
        """
        - Rings defined as an N_RINGS matrix
        - Rings are objects of class Ring with R number of attributes
            - pos: np.ndarray: specify the rings centre (x, y, z)
            - vel: np.ndarray: specify the rings velocity (x', y', z')
            - ori: np.ndarray: specify the orientation of the ring as a rotation matrix (a, b, c)
            - ori_vel: np.ndarray: specify the rotation velocity as a rotation matrix (a', b', c')
            - mass: float: specifies the ring's mass
            - radius: float: specifies the ring's radius
            - center_mass: np.ndarray: specifies the ring's center of mass
            - rotation_matrix: np.ndarray: rotation matrix
        """
        return self._rings

    def _update(self) -> None:  # method to run one step further of the simulation
        self._update_rods()
        self._update_links()
        self._update_rings()

    def _update_rods(self) -> None:
        """Method to update the rods in the simulation"""
        # Iterate thorugh every row in the links array
        for i, row in enumerate(self.links):
            # Iterate through every connection of the row
            for j, link in enumerate(row):
                if isinstance(link, EmptyLink):
                    # Means there is not a connection
                    continue  # Dont process this link
                ring = self.rings[i]
                phi = norm(ring.ori)  # magnitude of the orientation vector

                # 1) Convert the orientation vector of the ring into a ring rotation matrix
                ring.rotation_matrix = get_rotation_matrix(phi, ring.ori)

                # 2) Convert theta into a link rotation matrix
                link.rotation_matrix = get_rotation_matrix(link.theta, [0, 1, 0])

                # 3) Updte links position
                link.pos = ring.rotation_matrix @ link.rotation_matrix @ array([ring.radius, 0, 0]) + ring.pos

        # Second iteration
        for i, row in enumerate(self.links):
            for j, link in enumerate(row):
                if isinstance(link, EmptyLink):
                    # Means there is not a connection
                    continue  # Dont process this link
                # 4) Get rod lenght, link_l
                pos_f = self.links[j, i].pos
                pos_i = link.pos
                w = pos_f - pos_i
                rod_l = norm(w)  # magnitude of the w vector

                # 5) Get the force vector
                link.force = (link.spring_c * (rod_l - link.rod_l_0) * (w / rod_l)).astype('float64')

    def _update_links(self) -> None:
        """Method to update the links of the simulation"""
        # Iterate thorugh every ring in the links array
        for i, row in enumerate(self.links):
            # Iterate through every connection of the ring
            for j, link in enumerate(row):
                if isinstance(link, EmptyLink):
                    # Means there is not a connection
                    continue  # Dont process this link
                ring = self.rings[i]

                # 1) Calculate the tangent vector
                q = ring.rotation_matrix @ link.rotation_matrix @ transpose(array([0, 1, 0]))

                # 2) Get the angle, gamma, betweem q and f
                if not norm(link.force):  # if the link force is 0
                    link.torque_vector = np.zeros((3,))  # the link torque is an array if zeros
                else:
                    gamma = arccos((q @ link.force) / (norm(q) * norm(link.force)))
                    # 3) Get the link's torque, link_t
                    link.torque_vector = (cross(q, link.force) / norm(cross(q, link.force))) * link.torque_c * \
                                         (np.pi / 2 - gamma)

                # 4) Get the angular acceleration
                f_a = transpose(link.rotation_matrix) @ transpose(ring.rotation_matrix) @ link.force
                f_y = array([0., 1., 0.]) @ f_a
                f_xz = (array([1., 1.]) @ (array([[1., 0., 0.],
                                                  [0., 0., 1.]]) @ f_a) ** 2) ** .5

                # Checking equality of floating values is problematic thats why we use |val - checker| < epsilon
                if abs(link.theta_vel - 0) < .0001 and norm(f_y) <= f_xz * link.friction_c:
                    theta_acc = 0
                elif abs(link.theta_vel - 0) < .0001:
                    theta_acc = (f_y - (sign(f_y) * f_xz) * link.friction_c) * ring.radius / link.mass
                else:
                    theta_acc = (f_y - (sign(link.theta_vel) * f_xz) * link.friction_c) * ring.radius / link.mass

                # 5) Update the link states, theta and theta_vel
                link.theta_vel += theta_acc * self.time_step
                link.theta += link.theta_vel

    def _update_rings(self) -> None:
        """Method to update the simulation's rings"""
        # Iterate thorugh every ring in the links array
        for i, row in enumerate(self.links):
            ring = self.rings[i]
            # 1) Calculate the center of mass
            m = ring.mass + sum((link.mass for link in row if not isinstance(link, EmptyLink)))
            ring_p_m = (ring.mass / m) * ring.pos + np.sum([((link.mass / m) * link.pos) for link in row
                                                        if not isinstance(link, EmptyLink)]).astype('float64')
            # 2) Sum of torques around ring_p_m, ring_t_m
            ring_t_m: np.ndarray = np.sum([(link.torque_vector + cross((link.pos - ring_p_m).astype('float64'), link.force))
                                            for link in row if not isinstance(link, EmptyLink)]).astype('float64')
            # 3) Sum forces around ring_p_m, ring_f_m
            ring_f_m: np.ndarray = np.sum([link.force for link in row if not isinstance(link, EmptyLink)]).astype('float64')

            # 4) Update Rings position, and velocity
            acc = ring_f_m / ring.mass
            ring.vel += acc * self.time_step
            ring.pos += ring.vel

            # 5) Update rings orientation
            ori_acc = ring_t_m / ring.mass
            ring.ori_vel += ori_acc * self.time_step
            ring.ori += ring.ori_vel

    def run(self) -> None:
        """Method to show the simulation"""
        self._update()


# Constants
# Settings
FPS = 60
WIDTH, HEIGHT = SIZE = 1200, 900

# Simulation
FOV = 60  # field of view
ZNEAR, ZFAR = .1, 150
CAM_INTIAL_POSITION = 0, 0, -30

VELOCITY = .1
ROT_VEL = .1

# Ring
RINGS: List[ShapeType] = []
RING_PATH = 'rings_and_rods/gbg/ring.json'
RING_PATH_LOW = 'rings_and_rods/objects_data/ring_low.json'
RING_PREFAB = load_shape_from_json(RING_PATH_LOW)

# Rod
RODS: List[List[Optional[ShapeType]]] = []
ROD_PATH = 'rings_and_rods/gbg/rod.json'
ROD_PATH_LOW = 'rings_and_rods/objects_data/rod_low.json'
ROD_PREFAB = load_shape_from_json(ROD_PATH_LOW)

# Cube
CUBE_PATH = 'rings_and_rods/gbg/cube.json'
CUBE = load_shape_from_json(CUBE_PATH)

# Ground
GROUND_PATH = 'rings_and_rods/objects_data/ground.json'
GROUND = load_shape_from_json(GROUND_PATH)


class Simulation:
    _running: bool = True
    _physics: Physics
    _button_down = False

    def __init__(self):
        pg.init()
        self._screen = pg.display.set_mode(SIZE, DOUBLEBUF | OPENGL)
        self._clock = pg.time.Clock()
        self._font = pg.font.SysFont('Arial', 30)
        glu.gluPerspective(FOV, (WIDTH / HEIGHT), ZNEAR, ZFAR)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glTranslatef(*CAM_INTIAL_POSITION)
        gl.glEnable(gl.GL_DEPTH_TEST)

        gl.glLight(gl.GL_LIGHT0, gl.GL_POSITION, (-5, 5, 5, 2))  # point light from the left, top, front
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, (0, 0, 0, 1))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, (1, 1, 1, 1))

    def start(self, time_step: float, n_rings: int = 2):
        self._physics = Physics(time_step=time_step, n_rings=n_rings)
        self._load_rings()
        self._load_links()
        self._run()

    def _load_rings(self):
        for ring in self._physics.rings:
            r = load_shape_from_prefab(RING_PREFAB)
            r['position'] = ring.pos  # reference to the position object
            r['transform'] = [ring.radius * 2, ring.radius * 2, 1]
            r['orientation'] = ring.ori  # reference to the orientation object
            RINGS.append(r)

    def _load_links(self):
        for i, row in enumerate(self._physics.links):
            RODS.append([])
            for j, link in enumerate(row):
                if isinstance(link, EmptyLink):
                    RODS[i].append(None)
                    continue
                if j < len(RODS):
                    if i < len(RODS[j]):
                        RODS[i].append(None)
                        continue
                rod = load_shape_from_prefab(ROD_PREFAB)
                rod['position_1'] = self._physics.links[i, j].pos
                rod['position_2'] = self._physics.links[j, i].pos
                RODS[i].append(rod)

    def _run(self):
        while self._running:
            self._clock.tick(FPS)
            # print(self._clock.get_fps())
            self._events()
            self._update()
            self._draw()
        pg.quit()

    def _events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self._running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_LEFT:
                    gl.glTranslatef(-0.5, 0, 0)
                elif event.key == pg.K_RIGHT:
                    gl.glTranslatef(0.5, 0, 0)
                elif event.key == pg.K_UP:
                    gl.glTranslatef(0, 1, 0)
                elif event.key == pg.K_DOWN:
                    gl.glTranslatef(0, -1, 0)
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_LEFT:
                    gl.glTranslatef(-1, 0, 0)
                if event.key == pg.K_RIGHT:
                    gl.glTranslatef(1, 0, 0)
                if event.key == pg.K_UP:
                    gl.glTranslatef(0, 1, 0)
                if event.key == pg.K_DOWN:
                    gl.glTranslatef(0, -1, 0)
            elif event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 4:
                    gl.glScalef(1.2, 1.2, 1.2)
                elif event.button == 5:
                    gl.glScalef(1 / 1.2, 1 / 1.2, 1 / 1.2)
            elif event.type == pg.MOUSEMOTION:
                if self._button_down:
                    gl.glRotatef(event.rel[1] * ROT_VEL, 1, 0, 0)
                    gl.glRotatef(event.rel[0] * ROT_VEL, 0, 1, 0)
        keys = pg.key.get_pressed()
        if keys[pg.K_w]:
            gl.glTranslatef(0, 0, VELOCITY)
        if keys[pg.K_a]:
            gl.glTranslatef(VELOCITY, 0, 0)
        if keys[pg.K_s]:
            gl.glTranslatef(0, 0, -VELOCITY)
        if keys[pg.K_d]:
            gl.glTranslatef(-VELOCITY, 0, 0)
        mouse = pg.mouse.get_pressed(3)
        if mouse[0] == 1:
            self._button_down = True
        elif mouse[0] == 0:
            self._button_down = False

    def _update(self):
        self._physics.run()
        self._update_rings()
        self._update_rods()

    def _update_rings(self):
        """Function to update the rings in the visual simulation"""
        for i, ring in enumerate(self._physics.rings):
            r = RINGS[i]
            r['position'] = ring.pos
            r['orientation'] = ring.ori

    def _update_rods(self):
        """Function to update the rods in the visual simulation"""
        for i, row in enumerate(self._physics.links):
            for j, link in enumerate(row):
                if RODS[i][j] is None:
                    continue
                rod = RODS[i][j]
                rod['position_1'] = self._physics.links[i, j].pos
                rod['position_2'] = self._physics.links[j, i].pos

    def _draw(self):
        gl.glPushMatrix()
        gl.glRotatef(-90, 1, 0, 0)
        gl.glClearColor(*colors['WHITE'], 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)  # Clear surface

        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)

        # Ground
        # Shapes.custom(**GROUND)
        for row in RODS:
            for rod in row:
                if rod is None:
                    continue
                Shapes.rod(**rod)
        for ring in RINGS:
            Shapes.ring(**ring)
        # print(f'FPS: {self._clock.get_fps()}')
        gl.glDisable(gl.GL_LIGHT0)
        gl.glDisable(gl.GL_LIGHTING)
        gl.glDisable(gl.GL_COLOR_MATERIAL)
        gl.glPopMatrix()
        pg.display.flip()


if __name__ == '__main__':
    s = Simulation()
    s.start(time_step=1 / FPS)
