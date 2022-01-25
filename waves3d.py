"""
Script to
Written by Pablo Duran (https://github.com/pabloduran016)
"""
import pygame as pg
from typing import TypeVar, List, Union
from colors import *
import numpy as np
import numpy.typing as npt
import OpenGL.GL as gl
import OpenGL.GLU as glu
from numpy import cos, sin, pi
from numpy.linalg import norm

# Constants
WIDTH, HEIGHT = SIZE = 800, 800
FPS = 60
TITLE = ''

_Vec_T = TypeVar('_Vec_T')

Vec3d = Union[Tuple[float, float, float], npt.NDArray[np.float64], List[float]]
Vertices = Union[npt.NDArray[np.int64], Tuple[Vec3d, ...]]
Edge = Union[npt.NDArray[np.int64], Tuple[int, int]]
Edges = Union[Tuple[Edge, ...], npt.NDArray[int]]
Surface = Union[npt.NDArray[np.int64], Tuple[int, int, int, int]]
Surfaces = Union[Tuple[Surface, ...], npt.NDArray[int]]
Normal = Vec3d
Normals = Union[npt.NDArray[np.float64], Normal]


def apply_position_change(vertices: Vertices, position: Vec3d) -> Vertices:
    assert type(position) == np.ndarray
    return np.array([vertex + position for vertex in vertices])


def apply_rot_matrix(vertices: Vertices, rotation_matrix: np.ndarray) -> Vertices:
    """
    Function to rotate a vertices marix by a fiven rotation matrix
    :param vertices: List of vertices in format (x, y, z)
    :param rotation_matrix: Rotaino matrix in the format of a numppy array
    :return: List of rotated vertices in format (x, y, z)
    """
    result = vertices @ rotation_matrix
    return result


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


def draw_shape(vertices: Vertices, edges: Edges, surfaces: Surfaces, edge_color: GLColor = None,
               surface_color: GLColor = None, normals: Normals = None, draw_edges: bool = False) -> None:
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


class Grid:
    def __init__(self, size: float, n: int):
        # self.width = width
        # self.height = height
        # self.n = n
        self.size = size
        self.n = n
        self.vertices: npt.NDArray[np.float64] = np.zeros((n, n, 3))
        self.cell_size = size / n

        edges = []
        surfs = []
        for k in range(n):
            for i in range(n):
                prev_i = k*n + i - 1
                prev_k = (k-1)*n + i
                prev_ik = (k-1)*n + i - 1
                here = k*n + i
                self.vertices[i, k, :] = [i - n/2, 0, k - n/2]
                if i > 0:
                    edges.append([prev_i, here])
                if k > 0:
                    edges.append([prev_k, here])
                if k != 0 and i != 0:
                    surfs.append([prev_ik, prev_k, here, prev_i])

        self.edges: Edges = np.array(edges)
        self.surfaces: Surfaces = np.array(surfs)
        self.normals: Normals = np.empty((n * n, 3))

    def update(self, _dt: float):
        vers = self.vertices.reshape((self.n*self.n, 3))
        for j in range(self.n):
            for i in range(self.n):
                if i != 0 and j != 0:
                    prev_x = j * self.n + i - 1
                    prev_y = (j - 1) * self.n + i
                    here = j * self.n + i
                    a = vers[prev_x] - vers[here]
                    b = vers[prev_y] - vers[here]
                    self.normals[j*self.n + i, :] = b @ a

    def draw(self):
        draw_shape(
            vertices=self.cell_size * self.vertices.reshape((self.n*self.n, 3)),
            edges=self.edges,
            surfaces=self.surfaces,
            edge_color=None,
            surface_color=gl_color(CELESTE),
            normals=self.normals,
            draw_edges=False,
        )


class Wave2d:
    def __init__(self, amp: float, per: float, l: float, phase: float):
        """
        Parameters:
            per: Period of the  wave
            l: Wave length
            phase: Phase
        """
        self.amp = amp
        self.a_freq = 2*pi/per
        self.k: Vec3d = np.array((2*pi/l, 2*pi/l))
        self.phase = phase

    def wave_func(self, x: float, z: float, t: float) -> float:
        # y(x, z, t) = Atrig(kx*x)trig(kz*z)trig(ωt)
        kx, kz = self.k
        a = self.amp*np.cos(kx*x)*np.cos(kz*z)*np.cos(self.a_freq*t + self.phase)
        # b = self.amp*np.cos(self.a_freq*t - self.k*x + self.phase)
        return a


GRID_SIZE = 20
GRID_N = 20

WAVE_AMP = 5
WAVE_T = 9e3
WAVE_WL = 2e-16
WAVE_PHASE = 0

FOV = 60  # field of view
ZNEAR, ZFAR = .1, 150
CAM_INITIAL_POSITION = 0, -30, -20
CAM_INITIAL_ANGLE = 60

VELOCITY = .1
ROT_VEL = .1


class Simulation3D:
    _button_down = False
    running: bool = True

    def __init__(self):
        pg.init()
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode(SIZE, pg.DOUBLEBUF | pg.OPENGL)
        self.font = pg.font.SysFont('Arial', 30)
        glu.gluPerspective(FOV, (WIDTH / HEIGHT), ZNEAR, ZFAR)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glRotatef(CAM_INITIAL_ANGLE, 1, 0, 0)
        gl.glTranslatef(*CAM_INITIAL_POSITION)
        gl.glEnable(gl.GL_DEPTH_TEST)

        gl.glLight(gl.GL_LIGHT0, gl.GL_POSITION, (-5, 5, 5, 2))  # point light from the left, top, front
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, (0, 0, 0, 1))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, (1, 1, 1, 1))

        self.grid = Grid(GRID_SIZE, GRID_N)
        self.wave = Wave2d(WAVE_AMP, WAVE_T, WAVE_WL, WAVE_PHASE)

    def start(self) -> None:  # method to start the simulation
        self.run()

    def run(self) -> None:  # method to show the simulation
        while self.running:
            dt = self.clock.tick(FPS)
            self.events()
            self.update(dt)
            self.draw()
            pg.display.flip()

    def events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            # if event.type == pg.KEYDOWN:
            #     if event.key == pg.K_LEFT:
            #         gl.glTranslatef(-0.5, 0, 0)
            #     elif event.key == pg.K_RIGHT:
            #         gl.glTranslatef(0.5, 0, 0)
            #     elif event.key == pg.K_UP:
            #         gl.glTranslatef(0, 1, 0)
            #     elif event.key == pg.K_DOWN:
            #         gl.glTranslatef(0, -1, 0)
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

    t: float = 0.

    def update(self, dt: float):
        self.t += dt
        x, y, _ = self.grid.vertices.shape
        for i in range(x):
            for j in range(y):
                verx, _, verz = self.grid.vertices[i, j]
                # verx = (self.grid.vertices[i, j, 0]*self.grid.cell_size) - self.grid.n*self.grid.cell_size/2
                # verz = (self.grid.vertices[i, j, 2]*self.grid.cell_size) - self.grid.n*self.grid.cell_size/2
                very = self.wave.wave_func(verx, verz, self.t)
                self.grid.vertices[i, j] = [verx, very, verz]
        self.grid.update(dt)

    def draw(self):
        gl.glPushMatrix()
        gl.glClearColor(*gl_color(WHITE), 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)  # Clear surface

        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)

        self.grid.draw()

        gl.glDisable(gl.GL_LIGHT0)
        gl.glDisable(gl.GL_LIGHTING)
        gl.glDisable(gl.GL_COLOR_MATERIAL)
        gl.glPopMatrix()


if __name__ == '__main__':
    Simulation3D().start()
