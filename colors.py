from typing import Tuple
from random import randint
import numpy as np


WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)
YELLOW = (255, 255, 0, 255)
GREEN = (0, 255, 0, 255)
RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)
PURPLE = (100, 25, 200, 255)
ColorType = Tuple[int, int, int, int]

def random_color() -> ColorType:
    return randint(0, 255), randint(0, 255), randint(0, 255), 255

def cmap_color(i: float) -> ColorType:
    return (
        round(abs(255*np.sin(np.pi/8 + i))),
        round(25+abs(180*np.cos(i))),
        round(abs(20*np.cos(i))),
        255
    )
