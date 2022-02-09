"""
Script to
Written by Pablo Duran (https://github.com/pabloduran016)
"""
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Type, Dict, Any, Tuple

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


def load_shader(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()
#
# def load_wave_vertex_shader(waves: List['Wave3d']):
#     ind = WAVE_VERTEX_SHADER.find('wave_func')
#     enter = WAVE_VERTEX_SHADER[ind:].find('{') + ind + 1
#     close = WAVE_VERTEX_SHADER[enter:].find('}') + enter
#
#     string = " + \n".join([f"({w.amp} * cos(d * {2*pi/w.wl} - {2*pi/w.per} * t + {w.phase}))" for w in waves])
#
#     body = f"""
#     float d = sqrt(pos.x*pos.x + pos.z*pos.z);
#     float y = ({string});
#     return vec3(pos.x, y, pos.z);
# """
#
#     return WAVE_VERTEX_SHADER[:enter] + body + WAVE_VERTEX_SHADER[close:]

@dataclass
class Wave3d:
    amp: float
    per: float
    wl: float
    phase: float


Vec3Type = Tuple[float, float, float]
SCENE_CONFIG_PATH = 'waves3d/scene.config'
@dataclass
class Constants:
    N_ANGLES: int = 30

    WAVE_VERTEX_SHADER_PATH: str = 'waves3d/wave.vert'
    WAVE_FRAGMENT_SHADER_PATH: str = 'waves3d/wave.frag'
    PLANE_VERTEX_SHADER_PATH: str = 'waves3d/plane.vert'
    PLANE_FRAGMENT_SHADER_PATH: str = 'waves3d/plane.frag'

    PLANE_RESOLUTION: int = 10

    GRID_N: int = 150
    GRID_SIZE: float = 100.
    CUBE_HEIGHT: float = 30.
    GRID_POS: Vec3Type = 0., 0., 0.
    FOV: float = 60.  # field of view
    ZNEAR: float = .1
    ZFAR: float = 1000.


@dataclass
class Settings:
    PLANE_Y: int = -15
    PLANE_COLOR: GLColor = gl_color(GREY)
    PLANE_EDGE_COLOR: GLColor = gl_color(GREEN)
    DRAW_PLANE_TRIANGLES: bool = True
    DRAW_PLANE_EDGES: bool = False

    GRID_COLOR: GLColor = gl_color(CELESTE)
    GRID_POINTS_COLOR: GLColor = gl_color(BLACK)
    GRID_EDGE_COLOR: GLColor = gl_color(BLACK)
    DRAW_GRID_TRIANGLES: bool = True
    DRAW_GRID_POINTS: bool = False
    DRAW_GRID_EDGES: bool = False

    DELTA_TIME: float = 1000.


CONSTANTS = Constants()
INITIAL_SETTINGS = Settings()


@dataclass
class Config:
    waves: List[Wave3d]
    setts: Settings
    consts: Constants


def load_config(path: str) -> Config:
    defs: Dict[str, float] = {}
    config = Config([], INITIAL_SETTINGS, CONSTANTS)
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            if line.startswith('//'):
                continue
            if line.find('=') > 0:
                name, value = line.split('=')
                if name.startswith('$'):
                    defs[name] = float(value)
                elif hasattr(Settings, name):
                    typ = Settings.__annotations__[name]
                    new: Any
                    if typ == float:
                        new = float(value)
                    elif typ == int:
                        new = int(value)
                    elif typ == str:
                        new = value
                    elif typ == GLColor:
                        vs = [v.strip() for v in value.split(',')]
                        assert len(vs) == 4
                        r, g, b, a = vs
                        new = (float(r), float(g), float(b), float(a))
                    elif typ == Vec3Type:
                        vs = [v.strip() for v in value.split(',')]
                        assert len(vs) == 3
                        x, y, z = vs
                        new = (float(x), float(y), float(z))
                    elif typ == bool:
                        if value in {'True', 'False'}:
                            new = value == 'True'
                        else:
                            raise TypeError(f'Expected `True` or `False` for boolean constant')
                    else:
                        raise TypeError(f'Invalid type for constant: {typ}')
                    setattr(config.setts, name, new)
                else:
                    raise TypeError(f'Unknown constant {name}')
                continue
            values = [(float(x) if x not in defs else defs[x]) for x in line.split()]
            # FORMAT = AMP PERIOD WAVELENGTH PHASE
            config.waves.append(Wave3d(*values))
    return config


class Sprite(ABC):
    @abstractmethod
    def change(self, trans=None, rot=None, model=None, proj=None, config: Config = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def draw(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self, dt: float) -> None:
        raise NotImplementedError


WAVES_CAP=16


class Grid(Sprite):
    def __init__(self, config: Config):
        self.config = config

        if len(self.config.waves) > WAVES_CAP:
            print(f'Number of waves (len(waves)) is bigger than waves capacity ({WAVES_CAP})', file=sys.stderr)
            exit(1)

        self.config = config
        self.grid = gloo.Program(load_shader(self.config.consts.WAVE_VERTEX_SHADER_PATH), load_shader(self.config.consts.WAVE_FRAGMENT_SHADER_PATH))

        n_waves = len(self.config.waves)
        self.grid['size'] = self.config.consts.GRID_SIZE
        self.grid['n_waves'] = n_waves

        self.grid['amps'] = np.zeros(WAVES_CAP, dtype=np.float32)
        self.grid['amps'][:n_waves] = np.array([w.amp for w in self.config.waves])

        self.grid['ks'] = np.zeros(WAVES_CAP, dtype=np.float32)
        self.grid['ks'][:n_waves] = np.array([(2*pi/w.wl) for w in self.config.waves])

        self.grid['a_freqs'] = np.zeros(WAVES_CAP, dtype=np.float32)
        self.grid['a_freqs'][:n_waves] = np.array([2*pi/w.per for w in self.config.waves])

        self.grid['phases'] = np.zeros(WAVES_CAP, dtype=np.float32)
        self.grid['phases'][:n_waves] = np.array([w.phase for w in self.config.waves])

        vertices, edges, triangles, height = self.gen_vertices(self.config.consts.GRID_N, self.config.consts.GRID_SIZE, 0)
        vertices['position'] += self.config.consts.GRID_POS
        self.grid['height'] = height

        self.vertices = vertices.view(gloo.VertexBuffer)
        self.triangles = triangles.view(gloo.IndexBuffer)
        self.edges = edges.view(gloo.IndexBuffer)
        self.grid.bind(self.vertices)

    def gen_vertices(self, n: int, cell_s: float, height: float) -> Tuple:
        raise NotImplementedError

    def change(self, trans=None, rot=None, model=None, proj=None, config: Config = None) -> None:
        if trans is not None:
            self.grid['translation'] = trans
        if config is not None:
            self.config = config
            if len(config.waves) > WAVES_CAP:
                print(f'Number of waves (len(waves)) is bigger than waves capacity ({WAVES_CAP})', file=sys.stderr)
                exit(1)
            n_waves = len(config.waves)
            self.grid['n_waves'] = n_waves
            self.grid['size'] = self.config.consts.GRID_SIZE

            self.grid['amps'] = np.zeros(WAVES_CAP, dtype=np.float32)
            self.grid['amps'][:n_waves] = np.array([w.amp for w in config.waves])

            self.grid['ks'] = np.zeros(WAVES_CAP, dtype=np.float32)
            self.grid['ks'][:n_waves] = np.array([(2 * pi / w.wl) for w in config.waves])

            self.grid['a_freqs'] = np.zeros(WAVES_CAP, dtype=np.float32)
            self.grid['a_freqs'][:n_waves] = np.array([2 * pi / w.per for w in config.waves])

            self.grid['phases'] = np.zeros(WAVES_CAP, dtype=np.float32)
            self.grid['phases'][:n_waves] = np.array([w.phase for w in config.waves])
        if model is not None:
            self.grid['model'] = model
        if rot is not None:
            self.grid['rotation'] = rot
        if proj is not None:
            self.grid['projection'] = proj

    def update(self, dt: float):
        self.grid['t'] += dt*self.config.setts.DELTA_TIME
        pass

    def draw(self):
        # Filled
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        self.grid['color'] = self.config.setts.GRID_COLOR
        self.grid.draw(gl.GL_TRIANGLES, self.triangles)

        # Outlined
        if self.config.setts.DRAW_GRID_EDGES:
            gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
            gl.glEnable(gl.GL_BLEND)
            gl.glDepthMask(gl.GL_FALSE)
            self.grid['color'] = self.config.setts.GRID_EDGE_COLOR
            self.grid.draw(gl.GL_LINES, self.edges)
            gl.glDepthMask(gl.GL_TRUE)

        # Points
        if self.config.setts.DRAW_GRID_POINTS:
            gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
            gl.glEnable(gl.GL_BLEND)
            gl.glDepthMask(gl.GL_FALSE)
            self.grid['color'] = self.config.setts.GRID_POINTS_COLOR
            self.grid.draw(gl.GL_POINTS)
            gl.glDepthMask(gl.GL_TRUE)


class SquaredGrid(Grid):
    def gen_vertices(self, n: int, size: float, height: float) -> Tuple:
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

        return vertices.reshape((n*n)), edges.reshape((n-1)*n*2*2), triangles.reshape(((n-1)**2)*2*3), height


class CircularGrid(Grid):
    def gen_vertices(self, n: int, size: float, height) -> Tuple:
        r = size / 2
        n = n + 1 if n % 2 == 0 else n
        d_angle = 360 / (self.config.consts.N_ANGLES)
        vertices = np.zeros(self.config.consts.N_ANGLES * n, [('position', np.float32, 3)])
        edges = np.zeros((self.config.consts.N_ANGLES, n, 2, 2), np.uint32)
        triangles = np.zeros((self.config.consts.N_ANGLES, (n - 1), 2, 3), np.uint32)
        for a in range(self.config.consts.N_ANGLES):
            for p in range(n):
                angle = a * d_angle/2
                point = (p - (n//2)) / ((n - 1) // 2)
                x = point * np.cos(angle * pi / 180)
                z = point * np.sin(angle * pi / 180)
                vertices[a*n + p]['position'] = [r*x, 0, r*z]
                if p < n - 1:
                    if a < self.config.consts.N_ANGLES - 1:
                        edges[a, p, 0] = (a*n + p, a*n + p + 1)
                        edges[a, p, 1] = (a*n + p, (a+1)*n + p)
                        if p == (n - 1)//2 - 1: # center triangles
                            triangles[a, p, 0] = a*n + p, a*n + p + 1, (a + 1)*n + p
                        elif p == (n - 1)//2: # center triangles
                            triangles[a, p, 0] = a*n + p, a*n + p + 1, (a + 1)*n + p + 1
                        elif point > 0:
                            triangles[a, p, 0] = a*n + p, a*n + p + 1, (a + 1)*n + p + 1
                            triangles[a, p, 1] = (a + 1)*n + p, a*n + p, (a + 1)*n + p + 1
                        elif point < 0:
                            triangles[a, p, 0] = a*n + p, a*n + p + 1, (a + 1)*n + p + 1
                            triangles[a, p, 1] = (a + 1)*n + p, a*n + p, (a + 1)*n + p + 1
                    else:
                        edges[a, p, 0] = (a*n + p, a*n + p + 1)
                        edges[a, p, 1] = (0 + (n - 1 - p), a*n + p)
                        if p == (n - 1)//2 - 1: # center triangles
                            triangles[a, p, 0] = a*n + p, a*n + p + 1, 0*n + (n - 1 - p)
                        elif p == (n - 1)//2: # center triangles
                            triangles[a, p, 0] = a*n + p, a*n + n - p, 0*n + n - p - 2
                        elif point > 0:
                            triangles[a, p, 0] = a*n + p, a*n + p + 1, 0*n + n - p - 1
                            triangles[a, p, 1] = a*n + p + 1, 0*n + n - p - 2, 0*n + n - p - 1
                        elif point < 0:
                            triangles[a, p, 0] = a*n + p, a*n + p + 1, 0*n + n - p - 1
                            triangles[a, p, 1] = a*n + p + 1, 0*n + n - p - 2, 0*n + n - p - 1
                else:
                    if a < self.config.consts.N_ANGLES - 1:
                        edges[a, p, 0] = (a*n + p, (a + 1)*n + p)
                    else:
                        edges[a, p, 1] = (0 + (n - 1 - p), a*n + p)


        # triangles[np.reshape([(triangles[i, j, k] == [0, 0, 0]).all() for i in range(triangles.shape[0]) for j in range(triangles.shape[1]) for k in range(triangles.shape[2])], triangles.shape[:-1]), :]
        edges = edges.reshape(self.config.consts.N_ANGLES*(n)*2*2)
        triangles = triangles.reshape(self.config.consts.N_ANGLES*(n -1)*2*3)
        return vertices.reshape(self.config.consts.N_ANGLES*n), edges, triangles, height


def cube(grid: Type[Grid], height: float):
    assert grid == CircularGrid, 'Other types of grids are not implemeted, yet'
    class CubeGrid(grid):
        def gen_vertices(self, n: int, size: float, _height: float) -> Tuple:
            n = n if n % 2 == 1 else n + 1
            vertices, edges, triangles, _ = super().gen_vertices(n, size, height)
            n_vert = len(vertices)
            vertices_extruded = np.zeros(2*n_vert, dtype=[('position', np.float32, 3)])
            vertices_extruded['position'][:n_vert] = vertices['position']
            vertices_extruded['position'][n_vert:] = vertices['position'] + [0, height, 0]
            new_edges = np.array([i*n + j + k for i in range(self.config.consts.N_ANGLES) for j in [0, n - 1] for k in [0, n_vert]], np.uint32)
            edges_extruded = np.concatenate((edges, edges + n_vert, new_edges))
            new_triangles = []
            for i in range(self.config.consts.N_ANGLES):
                if i < self.config.consts.N_ANGLES - 1:
                    new_triangles.append((i*n, i*n + n_vert, (i + 1)*n + n_vert))
                    new_triangles.append((i*n, (i + 1)*n, (i + 1)*n + n_vert))
                    new_triangles.append((i*n + n - 1, i*n + n - 1 + n_vert, (i + 1)*n + n - 1 + n_vert))
                    new_triangles.append((i*n + n - 1, (i + 1)*n + n - 1, (i + 1)*n + n - 1 + n_vert))
                else:
                    new_triangles.append((i*n, i*n + n_vert, n - 1 + n_vert))
                    new_triangles.append((i*n, n - 1, n - 1 + n_vert))
                    new_triangles.append((i*n + n - 1, i*n + n - 1 + n_vert, n_vert))
                    new_triangles.append((i*n + n - 1, 0, n_vert))
            triangles_extruded = np.concatenate((triangles, triangles + n_vert, np.array(new_triangles, dtype=np.uint32).flatten()))
            return vertices_extruded, edges_extruded, triangles_extruded, height

    return CubeGrid


class Plane(Sprite):
    def __init__(self, config: Config):
        self.config = config
        self.plane = gloo.Program(load_shader(self.config.consts.PLANE_VERTEX_SHADER_PATH), load_shader(self.config.consts.PLANE_FRAGMENT_SHADER_PATH))
        # TODO: Use the height to create a water cube
        res = self.config.consts.PLANE_RESOLUTION
        self.vertices = np.zeros((res + 1)**2, dtype=[('position', np.float32, 3)])
        # self.vertices['position'] = [
        #     [-1, y, -1], [-1, y, 1],
        #     [1,  y, -1], [1,  y, 1]
        # ]
        self.vertices['position'] = [
            [2*i/res - 1, self.config.setts.PLANE_Y, 2*j/res - 1]
            for i in range(res + 1) for j in range(res + 1)
        ]

        self.indices = np.array([
            ([i + (res+1)*j, i + (res+1)*(j + 1), i + 1 + (res+1)*j],
            [i + 1 + (res+1)*(j + 1), i + (res+1)*(j + 1), i + 1 + (res+1)*j])
            for i in range(res) for j in range(res)
        ], np.uint32).flatten()


        self.vertices = self.vertices.view(gloo.VertexBuffer)
        self.indices = self.indices.view(gloo.IndexBuffer)
        self.plane.bind(self.vertices)
        self.plane['scale'] = self.config.consts.ZFAR / res

    def change(self, trans=None, rot=None, model=None, proj=None, config: Config =None) -> None:
        if trans is not None:
            self.plane['translation'] = trans
        if rot is not None:
            self.plane['rotation'] = rot
        if proj is not None:
            self.plane['projection'] = proj
        if model is not None:
            pass
        if config is not None:
            self.config = config

    def update(self, dt: float):
        pass

    def draw(self):
        # Filled
        if self.config.setts.DRAW_PLANE_TRIANGLES:
            gl.glDisable(gl.GL_BLEND)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
            self.plane['color'] = self.config.setts.PLANE_COLOR
            self.plane.draw(gl.GL_TRIANGLES, self.indices)

        # Outline
        if self.config.setts.DRAW_PLANE_EDGES:
            gl.glDisable(gl.GL_BLEND)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
            self.plane['color'] = self.config.setts.PLANE_EDGE_COLOR
            self.plane.draw(gl.GL_TRIANGLES, self.indices)


MAX_CHR = 0x110000
RCTRL = -1
LCTRL = -1


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
        self.window.event(self.on_mouse_scroll)

        self.keyboard = key.KeyStateHandler()
        self.window.push_handlers(self.keyboard)

        self.translation = np.eye(4, dtype=np.float32)
        self.rotation = np.eye(4, dtype=np.float32)
        # TODO: add lighting to the scene. Calculate normals
        self.config = load_config(SCENE_CONFIG_PATH)
        self.sprites: List[Sprite] = [
            # SquaredGrid(GRID_SIZE, GRID_N, (GRID_POS[0], GRID_POS[1] + 10, GRID_POS[2]), [
            #     Wave3d(1, 4e3, 3.5, WAVE_PHASE),
            #     # Wave3d(1.5, 1e3, 8, WAVE_PHASE),
            #     # Wave3d(8, 5e3, 20, WAVE_PHASE),
            #     # Wave3d(1, .5e3, 2, WAVE_PHASE),
            # ]),
            cube(CircularGrid, self.config.consts.CUBE_HEIGHT)(self.config),
            Plane(self.config),
        ]
        self.reset()

    def reload_config(self):
        self.config = load_config(SCENE_CONFIG_PATH)
        for sprite in self.sprites:
            if isinstance(sprite, Grid):
                sprite.change(config=self.config)

    def reset(self):
        translation = np.eye(4, dtype=np.float32)
        rotation = np.eye(4, dtype=np.float32)
        model = np.eye(4, dtype=np.float32)
        glm.rotate(rotation, 45, 0, 1, 0)
        glm.rotate(rotation, 30, 1, 0, 0)
        glm.translate(translation, 0, 0, -self.config.consts.GRID_SIZE*1.5)
        self.translation = translation
        self.rotation = rotation
        for obj in self.sprites:
            obj.change(model=model, trans=translation, rot=rotation)

    def translate(self, x, y, z):
        glm.translate(self.translation, x, y, z)
        for obj in self.sprites:
            obj.change(trans=self.translation)

    def rotate(self, angle, x, y, z):
        glm.rotate(self.rotation, angle, x, y, z)
        for obj in self.sprites:
            obj.change(rot=self.rotation)

    def translatex(self, dist): self.translate(dist, 0, 0)

    def translatey(self, dist): self.translate(0, dist, 0)

    def translatez(self, dist): self.translate(0, 0, dist)

    def rotatex(self, angle): self.rotate(angle, 1, 0, 0)

    def rotatey(self, angle): self.rotate(angle, 0, 1, 0)

    def on_key_press(self, symbol, modifiers):
        if 0 < symbol < MAX_CHR:
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
            elif symbol == key.F5:
                print('RELOADING CONFIG')
                self.reload_config()

    def on_mouse_drag(self, _x, _y, dx, dy, _button):
        if self.keyboard[LCTRL] or self.keyboard[RCTRL]:
            self.translatex(dx / 7)
            self.translatey(-dy / 7)
        else:
            if abs(dy) > 2:
                self.rotatex(np.arctan(dy)*1.3)
            if abs(dx) > 2:
                self.rotatey(np.arctan(dx)*1.3)

    def on_key_release(self, _symbol, _modifiers):
        pass

    def on_mouse_scroll(self, _x, _y, _dx, dy):
        self.translatez(dy*5)

    def on_init(self):
        # self.reset()
        gl.glEnable(gl.GL_DEPTH_TEST)

    def on_resize(self, width, height):
        ratio = width / height
        proj = glm.perspective(self.config.consts.FOV, ratio, self.config.consts.ZNEAR, self.config.consts.ZFAR)
        for obj in self.sprites:
            obj.change(proj=proj)
        # self.grid.grid['projection'] = glm.ortho(-2, 2, -2, 2, .1, 100)

    def start(self) -> None:
        app.run(framerate=FPS)

    def update(self, dt: float):
        for obj in self.sprites:
            obj.update(dt)

    def on_draw(self, dt: float):
        self.update(dt)
        self.window.clear()
        for obj in self.sprites:
            obj.draw()


if __name__ == '__main__':
    Simulation3D().start()
