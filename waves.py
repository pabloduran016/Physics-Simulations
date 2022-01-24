"""
Script to
Written by Pablo Duran (https://github.com/pabloduran016)
"""
from abc import ABC, abstractmethod
from enum import auto, Enum

import pygame as pg
from typing import TypeVar, Generic, Optional, Sequence, List
from colors import *
import numpy as np
from numpy import pi


# Constants
WIDTH, HEIGHT = SIZE = 800, 800
FPS = 60
TITLE = ''
GRAVITY = 0, 1

STRING_POINTS = 200
STRING_Y = HEIGHT / 4

A = [50 , 50 ,]
T = [1000, 1000,]
L = [200 , -200,]
P = [0   , 0   ,]

PERIOD_MIN = 500
PERIOD_MAX = 2000
WL_MIN = 100
WL_MAX = 600

POINT_R = 5

_Vec_T = TypeVar('_Vec_T')

class Vec(pg.Vector2, Generic[_Vec_T]):
    color: Optional[ColorType] = None
    r: int = POINT_R

    def set_color(self, val: ColorType):
        self.color = val

    def set_y(self, val: float):
        self.y = val


_Points_T = TypeVar('_Points_T')

class Points(Generic[_Points_T]):
    def __init__(self, ps: Sequence[Vec[float]]):
        self._ps = ps

    def __add__(self, val: Vec):
        if type(val) != Vec:
            raise TypeError(f'Can only add Vectors to points')
        return [val + p for p in self._ps]

    def __iter__(self):
        return self._ps.__iter__()

    def __getitem__(self, *args, **kwargs):
        return self._ps.__getitem__(*args, **kwargs)

    def _y(self, val: list):
        assert len(val) == len(self._ps)
        for x, p in zip(val, self._ps):
            p.set_y(x)
    y = property(lambda self: [p.y for p in self.ps], _y)


ScreenType  = pg.Surface


class PS(Enum):
    PERIOD = auto()
    WL = auto()

HUMAN_PS = {PS.PERIOD: 'PERIOD', PS.WL: 'WAVE LENGTH'}


class Wave:
    def __init__(self, amp: float, per: float, l: float, phase: float):
        """
        Parameters:
            per: Period of the  wave
            l: Wave length
            phase: Phase
        """
        self.amp = amp
        self.a_freq = 2*pi/per
        self.k = 2*pi/l
        self.phase = phase

    def _wave_func(self, x: float, t: float) -> float:
        a = self.amp*np.cos(self.a_freq*t - self.k*x + self.phase)
        # b = self.amp*np.cos(self.a_freq*t - self.k*x + self.phase)
        return a

    def set_param(self, p: PS, val: float):
        if p == PS.PERIOD:
            self.a_freq = 2*pi/val
        elif p == PS.WL:
            self.k = 2 * pi / val
        else:
            raise Exception(f'Unreachable, {p}')


class String:
    def __init__(self, n_points: int, y: float, waves: List[Wave]):
        self.waves = waves
        self.pos: Vec[float] = Vec(0, y)
        self.points: Points[Vec] = Points([Vec(x*WIDTH/n_points, self._apply(x*WIDTH/n_points, 0)) for x in range(n_points+1)])

    t = 0

    def _apply(self, x: float, t: float) -> float:
        return sum(w._wave_func(x, t) for w in self.waves)

    def update(self, dt: float) -> None:
        self.t += dt
        self.points.y = [self._apply(e.x, self.t) for e in self.points]

    def draw(self, scr: pg.Surface) -> None:
        pg.draw.lines(scr, BLACK, False, self.points + self.pos, 1)
        for point in self.points:
            if point.color is None: continue
            pg.draw.circle(scr, point.color, point + self.pos, point.r, 0)


class Slider:
    def __init__(self, x: float, y: float, width: int, height: int, min: float, max: float, val: float,
                 color_s: ColorType, color_h: ColorType):
        self.s_rect = pg.Rect((x, y), (width, height))  # Slider rectangle
        self.max = max
        self.min = min
        w = h = 2*height
        self.h_rect = pg.Rect((self.s_rect.left + self._clamp(val)*self.s_rect.width/max - w / 2, y - height/2), (w, h))  # Handler rectangle
        self.color_s = color_s
        self.color_h = color_h

    def _clamp(self, val: float) -> float:
        return max(min(self.max, val), self.min)

    def draw(self, scr: ScreenType):
        slider = pg.Surface(self.s_rect.size, pg.SRCALPHA)
        pg.draw.rect(slider, self.color_s, slider.get_rect(), border_radius=self.s_rect.height//2)
        scr.blit(slider, self.s_rect.topleft)

        handler = pg.Surface(self.h_rect.size, pg.SRCALPHA)
        pg.draw.rect(handler, self.color_h, handler.get_rect(), border_radius=self.h_rect.height//2)
        scr.blit(handler, self.h_rect.topleft)

    def get_val_by_pixels(self, x: float) -> float:
        return self._clamp(self.min + (self.max - self.min) * (x - self.s_rect.left) / self.s_rect.w)

    def update_rect(self, val: float):
        if val < self.min or val > self.max:
            raise ValueError(f'Value nopt between bounds ({self.min}, {self.max}) got {val}')
        self.h_rect.x = self.s_rect.left + self.s_rect.w * (val - self.min) / (self.max - self.min) - self.h_rect.w / 2


def is_in(point: Tuple[float, float], rect: pg.Rect) -> bool:
    x, y = point
    return rect.left < x < rect.right and rect.top < y < rect.bottom


class BaseSimulation(ABC):
    running: bool = True

    def __init__(self):
        pg.init()
        pg.display.set_caption(TITLE)
        self.screen = pg.display.set_mode(SIZE)
        self.clock = pg.time.Clock()

    def start(self) -> None:  # method to start the simulation
        self.run()

    @abstractmethod
    def update(self, dt: float) -> None:  # method to run one step further of the simulation
        raise NotImplementedError

    @abstractmethod
    def draw(self) -> None:  # method to draw the current state of the simulation
        raise NotImplementedError

    def events(self):
        events = pg.event.get()
        for event in events:
            if event.type == pg.QUIT:
                self.running = False

    def run(self) -> None:  # method to show the simulation
        while self.running:
            dt = self.clock.tick(FPS)
            self.events()
            self.update(dt)
            self.draw()
            pg.display.flip()

# Class to carry out the simulation
class Simulation2D(BaseSimulation):
    clicked: Optional[Tuple[PS, int]] = None
    prev_mouse: Tuple[float, float] = (0, 0)

    def __init__(self):
        super().__init__()
        self.string = String(STRING_POINTS, STRING_Y, [
           Wave(A[0], T[0], L[0], P[0]),
           Wave(A[1], T[1], L[1], P[1]),
        ])
        for i, point in enumerate(self.string.points):
            point.set_color(cmap_color(i/100))

        padr, padl, padt, padb = 150, 60, 0, 60
        height = 10
        width = WIDTH - padl - padr
        # color_s = (50, 150, 250, 150)
        color_h = (250, 100, 100, 255)
        self.sliders = {
            PS.PERIOD: [
                Slider(padr, HEIGHT - (height + padb)/2 - (i + 1) * (padb + height),
                       width, height, PERIOD_MIN, PERIOD_MAX, T[i], cmap_color(2*i), color_h)
                for i, _ in enumerate(self.string.waves)
            ],
            PS.WL: [
                Slider(padr, HEIGHT - (i + 1) * (padb + height),
                       width, height, WL_MIN, WL_MAX, L[i], cmap_color(2*i), color_h)
                for i, _ in enumerate(self.string.waves)
            ],
        }
        self.font = pg.font.SysFont('arial', 14, 'bold')

    def events(self) -> None:  # method to check events
        super().events()
        for event in events:
            if event.type == pg.MOUSEBUTTONDOWN:
                self.prev_mouse = pg.mouse.get_pos()
                self.clicked = None
                for p in self.sliders.keys():
                    for i, sl in enumerate(self.sliders[p]):
                        if is_in(pg.mouse.get_pos(), sl.h_rect):
                            self.clicked = p, i
                            break
                        if is_in(pg.mouse.get_pos(), sl.s_rect):
                            mouse = pg.mouse.get_pos()
                            val = self.sliders[p][i].get_val_by_pixels(mouse[0])
                            self.sliders[p][i].update_rect(val)
                            for w in self.string.waves:
                                w.set_param(p, val)
                            break
            elif event.type == pg.MOUSEBUTTONUP:
               self.clicked = None

        if pg.mouse.get_pressed()[0] and self.clicked is not None:
            mouse = pg.mouse.get_pos()
            p, i = self.clicked
            val = self.sliders[p][i].get_val_by_pixels(mouse[0])
            self.sliders[p][i].update_rect(val)
            self.string.waves[i].set_param(p, val)

    def draw(self) -> None:  # method to draw the current state of the simulation
        self.screen.fill(WHITE)
        self.string.draw(self.screen)
        for p in self.sliders.keys():
            for i, slider in enumerate(self.sliders[p]):
                slider.draw(self.screen)
                font_s = self.font.render(HUMAN_PS[p] + f' ({i})', True, BLACK, None)
                self.screen.blit(font_s, (slider.s_rect.left - font_s.get_rect().w - 5, slider.s_rect.centery - font_s.get_rect().h/2))


if __name__ == '__main__':
    s = Simulation2D()
    # start simulation
    s.start()
