"""
Script to
Written by Pablo Duran (https://github.com/pabloduran016)
"""
from glumpy import app, gloo, gl, glm
from OpenGL import GL

from typing import TypeVar, List, Union

from numpy import pi

from colors import *
import numpy as np
import numpy.typing as npt


# Constants
WIDTH, HEIGHT = SIZE = 800, 800
FPS = 60
TITLE = ''


# def draw_shape(vertices: Vertices, edges: Edges, surfaces: Surfaces, edge_color: GLColor = None,
#                surface_color: GLColor = None, normals: Normals = None, draw_edges: bool = False) -> None:
#     gl.glBegin(gl.GL_QUADS)
#     # Draw surfaces
#     if surface_color is not None:
#         gl.glColor3fv(surface_color)
#     for i, surface in enumerate(surfaces):
#         if normals is not None:
#             gl.glNormal3fv(normals[i])
#         for vertex in surface:
#             gl.glVertex3fv(vertices[vertex])
#     gl.glEnd()
#     # Draw vertex and edges
#     if draw_edges:
#         gl.glBegin(gl.GL_LINES)
#         if edge_color is not None:
#             gl.glColor3fv(edge_color)
#         for i, edge in enumerate(edges):
#             for j, vertex in enumerate(edge):
#                 gl.glVertex3fv(vertices[vertex])
#         gl.glEnd()


def load_shader(path: str):
    with open(path, 'r') as f:
        return f.read()


VERTEX_SHADER = load_shader('./vertex.vert')

FRAGMENT_SHADER = load_shader('./fragment.frag')


class Grid:
    def __init__(self, size: float, n: int, amp: float, per: float, phase: float, wl: float):
        self.size = size
        self.n = n
        self.cell_size = size / n

        self.grid = gloo.Program(VERTEX_SHADER, FRAGMENT_SHADER)

        self.grid['amp'] = amp
        self.grid['a_freq'] = 2*pi/per
        self.grid['phase'] = phase
        self.grid['kx'] = 2*pi/wl
        self.grid['kz'] = 2*pi/wl

        cell_s = size / n

        self.vertices = np.zeros((n, n), [('position', np.float32, 3)])
        self.triangles = np.zeros((n-1, n-1, 2, 3), dtype=np.uint32)
        self.edges = np.zeros((n-1, n, 2, 2), dtype=np.uint32)
        for i in range(n):
            for k in range(n):
                self.vertices[i, k]['position'] = [(i - (n-1) / 2)*cell_s, 0, (k - (n-1) / 2)*cell_s]
                if i != n-1 and k != n-1:
                    self.triangles[i, k, 0, :] = [i*n + k,   i*n + k+1,   (i+1)*n + k]
                    self.triangles[i, k, 1, :] = [i*n + k+1, (i+1)*n + k, (i+1)*n + k + 1]

                    self.edges[i, k, 0, :] = [i*n + k, i*n + k+1]
                    self.edges[i, k, 1, :] = [i*n + k, (i+1)*n + k]

                elif k == n-1 and i != n-1:
                    self.edges[i, k, 0, :] = [k*n + i, k*n + i+1]
                    self.edges[i, k, 1, :] = [i*n + k, (i+1)*n + k]

        self.vertices = self.vertices.reshape((n*n)).view(gloo.VertexBuffer)
        self.triangles = self.triangles.reshape(((n-1)**2)*2*3).view(gloo.IndexBuffer)
        self.edges = self.edges.reshape((n-1)*n*2*2).view(gloo.IndexBuffer)

        self.grid['color'] = gl_color(CELESTE)

        self.grid.bind(self.vertices)
        model = np.eye(4, dtype=np.float32)
        self.grid['model'] = model
        view = np.eye(4, dtype=np.float32)
        glm.rotate(view, 45, 0, 1, 0)
        glm.rotate(view, 20, 1, 0, 0)
        glm.translate(view, 0, 0, -self.size*1.5)
        self.grid['view'] = view

    def update(self, t: float):
        self.grid['t'] = t*1000
        view = np.eye(4, dtype=np.float32)
        glm.rotate(view, t*20, 0, 1, 0)
        glm.rotate(view, 20, 1, 0, 0)
        glm.translate(view, 0, 0, -self.size*1.5)
        self.grid['view'] = view
        pass

    def draw(self):
        # Filled cube
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        self.grid['color'] = gl_color(CELESTE)
        self.grid.draw(gl.GL_TRIANGLES, self.triangles)

        # Outlined cube
        # gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
        # gl.glEnable(gl.GL_BLEND)
        # gl.glDepthMask(gl.GL_FALSE)
        # self.grid['color'] = gl_color(BLACK)
        # self.grid.draw(gl.GL_LINES, self.edges)
        # gl.glDepthMask(gl.GL_TRUE)


GRID_SIZE = 20
GRID_N = 100

WAVE_AMP = 5
WAVE_T = 8e3
WAVE_WL = 6
WAVE_PHASE = 0

FOV = 60  # field of view
ZNEAR, ZFAR = .1, 150
# CAM_INITIAL_POSITION = 0, -30, -20
# CAM_INITIAL_ANGLE = 60
#
# VELOCITY = .1
# ROT_VEL = .1


class Simulation3D:
    _button_down = False
    running: bool = True

    def __init__(self):
        self.window = app.Window(width=WIDTH, height=HEIGHT, color=gl_color(WHITE))
        self.window.event(self.on_draw)
        self.window.event(self.on_resize)
        self.window.event(self.on_init)

        self.grid = Grid(GRID_SIZE, GRID_N, WAVE_AMP, WAVE_T, WAVE_PHASE, WAVE_WL)

    def on_init(self):
        gl.glEnable(gl.GL_DEPTH_TEST)

    def on_resize(self, width, height):
        ratio = width / height
        self.grid.grid['projection'] = glm.perspective(FOV, ratio, ZNEAR, ZFAR)
        # self.grid.grid['projection'] = glm.ortho(-2, 2, -2, 2, .1, 100)

    def start(self) -> None:
        app.run(framerate=FPS)

    t: float = 0.
    def update(self, dt: float):
        self.t += dt
        self.grid.update(self.t)

    def on_draw(self, dt: float):
        self.update(dt)
        self.window.clear()
        self.grid.draw()


if __name__ == '__main__':
    Simulation3D().start()
