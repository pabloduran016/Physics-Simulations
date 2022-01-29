"""
Script to
Written by Pablo Duran (https://github.com/pabloduran016)
"""
from dataclasses import dataclass
from typing import List

from glumpy import app, gloo, gl, glm, key
from numpy import pi

from colors import *
import numpy as np


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


WAVE_VERTEX_SHADER = load_shader('waves3d/wave.vert')
WAVE_FRAGMENT_SHADER = load_shader('waves3d/wave.frag')
PLANE_VERTEX_SHADER = load_shader('waves3d/plane.vert')
PLANE_FRAGMENT_SHADER = load_shader('waves3d/plane.frag')


def load_wave_vertex_shader(waves: List['Wave3d']):
    ind = WAVE_VERTEX_SHADER.find('wave_func')
    enter = WAVE_VERTEX_SHADER[ind:].find('{') + ind + 1
    close = WAVE_VERTEX_SHADER[enter:].find('}') + enter

    string = " + \n".join([f"({w.amp} * cos(d * {2*pi/w.wl} - {2*pi/w.per} * t + {w.phase}))" for w in waves])

    body = f"""
    float d = sqrt(pos.x*pos.x + pos.z*pos.z);
    float y = ({string});
    return vec3(pos.x, y, pos.z);
"""

    return WAVE_VERTEX_SHADER[:enter] + body + WAVE_VERTEX_SHADER[close:]

@dataclass
class Wave3d:
    amp: float
    per: float
    wl: float
    phase: float


class Grid:
    def __init__(self, size: float, n: int, waves: List[Wave3d]):
        self.size = size
        self.n = n
        self.cell_size = size / n

        v_shader = load_wave_vertex_shader(waves)
        self.grid = gloo.Program(v_shader, WAVE_FRAGMENT_SHADER)

        self.grid['size'] = size

        # self.grid['height'] = h

        vertices, edges, triangles = self.gen_vertices(n, size)

        self.vertices = vertices.view(gloo.VertexBuffer)
        self.triangles = triangles.view(gloo.IndexBuffer)
        self.edges = edges.view(gloo.IndexBuffer)
        self.grid['color'] = gl_color(CELESTE)
        self.grid.bind(self.vertices)

    def gen_vertices(self, n: int, cell_s: float) -> Tuple[np.ndarray, ...]:
        raise NotImplementedError

    def change(self, model=None, trans=None, rot=None) -> None:
        if trans is not None:
            self.grid['translation'] = trans
        if model is not None:
            self.grid['model'] = model
        if rot is not None:
            self.grid['rotation'] = rot

    def update(self, dt: float):
        self.grid['t'] += dt*1000
        pass

    def draw(self):
        # Filled
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        self.grid['color'] = gl_color(CELESTE)
        self.grid.draw(gl.GL_TRIANGLES, self.triangles)

        # Outlined
        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glEnable(gl.GL_BLEND)
        gl.glDepthMask(gl.GL_FALSE)
        self.grid['color'] = gl_color(BLACK)
        self.grid.draw(gl.GL_LINES, self.edges)
        gl.glDepthMask(gl.GL_TRUE)


class SquaredGrid(Grid):
    def gen_vertices(self, n: int, size: float) -> Tuple[np.ndarray, ...]:
        cell_s = size / n
        # TODO: Use the height to create a water cube
        vertices = np.zeros((n, n), [('position', np.float32, 3)])
        triangles = np.zeros((n - 1, n - 1, 2, 3), dtype=np.uint32)
        edges = np.zeros((n - 1, n, 2, 2), dtype=np.uint32)
        for i in range(n):
            for k in range(n):
                vertices[i, k]['position'] = [(i - (n - 1) / 2) * cell_s, 0, (k - (n - 1) / 2) * cell_s]
                if i != n - 1 and k != n - 1:
                    triangles[i, k, 0, :] = [i * n + k, i * n + k + 1, (i + 1) * n + k]
                    triangles[i, k, 1, :] = [i * n + k + 1, (i + 1) * n + k, (i + 1) * n + k + 1]

                if i != n - 1 and k != n - 1:
                    edges[i, k, 0, :] = [i * n + k, i * n + k + 1]
                    edges[i, k, 1, :] = [i * n + k, (i + 1) * n + k]

                elif k == n - 1 and i != n - 1:
                    edges[i, k, 0, :] = [k * n + i, k * n + i + 1]
                    edges[i, k, 1, :] = [i * n + k, (i + 1) * n + k]

        return vertices.reshape((n*n)), edges.reshape((n-1)*n*2*2), triangles.reshape(((n-1)**2)*2*3)


class CircularGrid(Grid):
    def gen_vertices(self, n: int, size: float) -> Tuple[np.ndarray, ...]:
        cell_s = size / n
        r = size / 2
        s = r/ n
        # TODO: Use the height to create a water cube
        vertices = np.zeros((n, n), [('position', np.float32, 3)])
        triangles = np.zeros((n - 1, n - 1, 2, 3), dtype=np.uint32)
        edges = np.zeros((n - 1, n, 2, 2), dtype=np.uint32)
        for i in range(n):
            for k in range(n):
                x = (i - (n - 1) / 2) * cell_s
                z = (k - (n - 1) / 2) * cell_s
                d = np.sqrt(x**2 + z**2)
                if d > r:
                    continue
                vertices[i, k]['position'] = [x, 0, z]
                if i != n - 1 and k != n - 1:
                    triangles[i, k, 0, :] = [i * n + k, i * n + k + 1, (i + 1) * n + k]
                    triangles[i, k, 1, :] = [i * n + k + 1, (i + 1) * n + k, (i + 1) * n + k + 1]

                if i != n - 1 and k != n - 1:
                    edges[i, k, 0, :] = [i * n + k, i * n + k + 1]
                    edges[i, k, 1, :] = [i * n + k, (i + 1) * n + k]

                elif k == n - 1 and i != n - 1:
                    edges[i, k, 0, :] = [k * n + i, k * n + i + 1]
                    edges[i, k, 1, :] = [i * n + k, (i + 1) * n + k]

        return vertices.reshape((n*n)), edges.reshape((n-1)*n*2*2), triangles.reshape(((n-1)**2)*2*3)


class Plane:
    def __init__(self, y: float, color: GLColor):
        self.plane = gloo.Program(PLANE_VERTEX_SHADER, PLANE_FRAGMENT_SHADER)

        # TODO: Use the height to create a water cube
        self.vertices = np.zeros(4, dtype=[('position', np.float32, 3)])
        self.vertices['position'] = [
            [-1, y, -1], [-1, y, 1],
            [1,  y, -1], [1,  y, 1]
        ]

        self.vertices = self.vertices.view(gloo.VertexBuffer)
        self.plane['color'] = color
        self.plane['zfar'] = ZFAR
        self.plane.bind(self.vertices)

    def change(self, trans=None, rot=None) -> None:
        if trans is not None:
            self.plane['translation'] = trans
        if rot is not None:
            self.plane['rotation'] = rot

    def update(self, dt: float):
        self.plane['t'] += dt*1000
        pass

    def draw(self):
        # Filled
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        # self.plane['color'] = gl_color(CELESTE)
        self.plane.draw(gl.GL_TRIANGLE_STRIP)


PLANE_COLOR = gl_color(GREY)
PLANE_Y = -15

GRID_SIZE = 50
GRID_N = 800
WCUBE_HEIGHT = 10

WAVE_AMP = 1.5
WAVE_T = 4e3
WAVE_WL = 3.5
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
        self.window.event(self.on_key_press)
        self.window.event(self.on_key_release)
        self.window.event(self.on_mouse_drag)

        self.translation = np.eye(4, dtype=np.float32)
        self.rotation = np.eye(4, dtype=np.float32)

        self.grid = SquaredGrid(GRID_SIZE, GRID_N, [
            Wave3d(1, 4e3, 3.5, WAVE_PHASE),
            Wave3d(1.5, 1e3, 8, WAVE_PHASE),
            Wave3d(8, 5e3, 20, WAVE_PHASE),
            Wave3d(1, .5e3, 2, WAVE_PHASE),
        ])
        # self.plane = Plane(PLANE_Y, PLANE_COLOR)
        self.plane = Plane(PLANE_Y, PLANE_COLOR)
        self.reset()

    def reset(self):
        translation = np.eye(4, dtype=np.float32)
        rotation = np.eye(4, dtype=np.float32)
        model = np.eye(4, dtype=np.float32)
        glm.rotate(rotation, 45, 0, 1, 0)
        glm.rotate(rotation, 30, 1, 0, 0)
        glm.translate(translation, 0, 0, -GRID_SIZE*1.5)
        self.translation = translation
        self.rotation = rotation
        self.grid.change(model=model, trans=translation, rot=rotation)
        self.plane.change(trans=translation, rot=rotation)

    def translatex(self, dist):
        glm.translate(self.translation, dist, 0, 0)
        self.grid.change(trans=self.translation)
        self.plane.change(trans=self.translation)

    def translatey(self, dist):
        glm.translate(self.translation, 0, dist, 0)
        self.grid.change(trans=self.translation)
        self.plane.change(trans=self.translation)

    def translatez(self, dist):
        glm.translate(self.translation, 0, 0, dist)
        self.grid.change(trans=self.translation)
        self.plane.change(trans=self.translation)

    def rotatex(self, angle):
        glm.rotate(self.rotation, angle, 1, 0, 0)
        self.grid.change(rot=self.rotation)
        self.plane.change(rot=self.rotation)

    def rotatey(self, angle):
        glm.rotate(self.rotation, angle, 0, 1 , 0)
        self.grid.change(rot=self.rotation)
        self.plane.change(rot=self.rotation)


    def on_key_press(self, symbol, modifiers):
        if 0 < symbol < 0x110000:
            if chr(symbol).lower() == 'r':
                self.reset()
        if modifiers & (key.LCTRL | key.RCTRL):
            if symbol == key.RIGHT:
                self.rotatey(-4)
            elif symbol == key.LEFT:
                self.rotatey(4)
            elif symbol == key.UP:
                self.rotatex(4)
            elif symbol == key.DOWN:
                self.rotatex(-4)
        else:
            if symbol == key.RIGHT:
                self.translatex(-2)
            elif symbol == key.LEFT:
                self.translatex(2)
            elif symbol == key.UP:
                self.translatey(-2)
            elif symbol == key.DOWN:
                self.translatey(2)

    def on_mouse_drag(self, _x, _y, dx, dy, _button):
        if abs(dy) > 2:
            self.rotatex(np.arctan(dy)*1.3)
        if abs(dx) > 2:
            self.rotatey(np.arctan(dx)*1.3)

    def on_key_release(self, _symbol, _modifiers):
        pass

    def on_init(self):
        # self.reset()
        gl.glEnable(gl.GL_DEPTH_TEST)

    def on_resize(self, width, height):
        ratio = width / height
        proj = glm.perspective(FOV, ratio, ZNEAR, ZFAR)
        self.grid.grid['projection'] = proj
        self.plane.plane['projection'] = proj
        # self.grid.grid['projection'] = glm.ortho(-2, 2, -2, 2, .1, 100)

    def start(self) -> None:
        app.run(framerate=FPS)

    def update(self, dt: float):
        self.grid.update(dt)

    def on_draw(self, dt: float):
        self.update(dt)
        self.window.clear()
        self.grid.draw()
        self.plane.draw()


if __name__ == '__main__':
    Simulation3D().start()
