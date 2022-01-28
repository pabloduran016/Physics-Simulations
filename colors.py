from typing import Tuple
from random import randint
import numpy as np


WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)
GREY = (148, 148, 148, 255)
YELLOW = (255, 255, 0, 255)
GREEN = (0, 255, 0, 255)
RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)
CELESTE = (81, 209, 246, 255)
PURPLE = (100, 25, 200, 255)
PgColor = Tuple[int, int, int, int]
GLColor = Tuple[float, float, float, float]

def random_color() -> PgColor:
    return randint(0, 255), randint(0, 255), randint(0, 255), 255

def cmap_color(i: float) -> PgColor:
    return (
        round(abs(255*np.sin(np.pi/8 + i))),
        round(25+abs(180*np.cos(i))),
        round(abs(20*np.cos(i))),
        255
    )

def gl_color(color: PgColor) -> GLColor:
    return color[0]/255, color[1]/255, color[2]/255, color[3]/255
