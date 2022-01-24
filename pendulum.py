import pygame as pg
from math import *
import random

vec = pg.Vector2


GRAVITY = 0.25
WIDTH = 800
HEIGHT = 800
TITLE = 'Langrnagian Pendulum'
FPS = 120
WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)
YELLOW = (255, 255, 0, 255)
GREEN = (0, 255, 0, 255)
RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)
PIVOT = (400, 0)
PIVOT_R = 10
LENGTH = 200
BOB_R = 10
LINE_W = 4
N_BALLS = [2, 2]
L_COLOR = [BLACK, BLUE]
THRESHOLD = 10
P_DESINTEGRATION = 0
PARTICLE_WIDTH = 5
FRICTION = 1
SELECTION_THREASHOLD = 10
STARTING_ANGLES = [[1, 2], [1, 2]]


def change(c):
    *arr, a = [x / 255 for x in c]
    return [round((x * a + 1 * (1 - a)) * 255) for x in arr]


class Pendulum:
    def __init__(self, game, n, i, t):
        self.id = i
        self.game = game
        self.balls = []
        self.length = LENGTH
        self.particles = []
        self.n_balls = n
        self.writing_balls = n
        self.line_color = L_COLOR[self.id]
        self.starting_angle = t
        self.pivot = Pivot(self.game, self)
        self.balls.append(self.pivot)
        for i in range(self.n_balls):
            self.balls.append(Ball(self.game, self.balls[i], self.length, self, i, self.starting_angle[i]))
        # self.balls[len(self.balls) - 1].id = 'lastball'

    def update(self):
        i = 0
        for ball in self.balls:
            ball.update()
            if ball.id == self.n_balls - 1:
                i += 1
                self.particles.append((ball.position, ball.color))
        self.writing_balls = i

    def draw(self):
        index = 0
        for particle in self.particles:
            pos, color = particle
            r, gr, b, a = color
            newcolor = (r, gr, b, a - P_DESINTEGRATION)
            self.particles[index] = (pos, newcolor)
            if newcolor[3] < 0:
                self.particles.pop(index)
            else:
                cr = change(newcolor)
                if index > self.writing_balls:
                    pg.draw.line(self.game.screen, cr, pos, self.particles[index-self.writing_balls][0], PARTICLE_WIDTH)
            index += 1
        for ball in reversed(self.balls):
            ball.draw()


class Pivot:
    def __init__(self, game, p):
        self.pendulum = p
        self.id = 'pivot'
        self.position = vec(PIVOT)
        self.game = game
        self.theta = pi / 4
        self.thetavel = 0
        self.thetaacc = 0
        self.length = 0

    def update(self):
        pass

    def draw(self):
        pg.draw.circle(self.game.screen, RED, self.position, PIVOT_R)


class Ball:
    def __init__(self, game, a, x, p, i, t=None):
        self.pendulum = p
        self.id = i
        if t is None:
            self.theta = random.randint(0, 300)/100
        else:
            self.theta = t
        self.position = a.position + vec(x*sin(self.theta), x*cos(self.theta))
        self.game = game
        self.thetavel = 0
        self.thetaacc = 0
        self.pivot = a
        self.length = x  # (self.position - self.pivot.position).length()
        self.color = (random.randint(10, 255), random.randint(10, 255), random.randint(10, 255), 255)

    def update(self):
        self.thetaacc = -(self.game.gravity / self.length) * sin(self.theta)
        self.thetavel += self.thetaacc
        self.theta += self.thetavel
        self.thetavel *= FRICTION
        self.position = vec(self.length * sin(self.theta), self.length * cos(self.theta)) + self.pivot.position

    def draw(self):
        pg.draw.line(self.game.screen, self.pendulum.line_color, self.position, self.pivot.position, LINE_W)
        pg.draw.circle(self.game.screen, self.color, self.position, BOB_R)


class Game:
    def __init__(self):
        pg.init()
        pg.mixer.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        self.gravity = GRAVITY
        self.running = True
        self.playing = True
        self.selected_ball = None
        self.wait = False
        self.mousepressed = None
        self.pendulums = []
        self.selected_pendulum = None

    def new(self):
        # start a new game
        i = 0
        if N_BALLS is not int:
            for n in N_BALLS:
                pendulum = Pendulum(self, n, i, STARTING_ANGLES[i])
                self.pendulums.append(pendulum)
                i += 1
        else:
            pendulum = Pendulum(self, N_BALLS, i, STARTING_ANGLES)
            self.pendulums.append(pendulum)
        self.run()

    def run(self):
        # Game Loop
        self.playing = True
        while self.playing:
            self.clock.tick(FPS)
            self.events()
            if not self.wait:
                self.update()
            self.draw()
            pg.display.flip()
            pg.display.update()

    def update(self):
        # Game Loop - Update
        for pendulum in self.pendulums:
            pendulum.update()
        pass

    def events(self):
        # Game Loop - events
        click = pg.mouse.get_pressed(3)
        for event in pg.event.get():
            # check for closing window
            if event.type == pg.QUIT:
                if self.playing:
                    self.playing = False
                self.running = False
            if event.type == pg.MOUSEBUTTONDOWN:
                self.mousepressed = pg.mouse.get_pressed(3)
                if self.mousepressed[0] and self.wait:
                    self.selected_ball = None
                    self.selected_pendulum = None
                    mouse = vec(pg.mouse.get_pos())
                    points = []
                    indexes = []
                    ids = []
                    index = 0
                    for pendulum in self.pendulums:
                        for ball in pendulum.balls:
                            indexes.append(index)
                            points.append(ball.position)
                            ids.append(pendulum.id)
                            index += 1
                    d = [(point - mouse).length() for point in points]
                    if min(d) < SELECTION_THREASHOLD + BOB_R:
                        selected_ball_index = indexes[d.index(min(d))]
                        self.selected_pendulum = self.pendulums[ids[d.index(min(d))]]
                        self.selected_ball = self.selected_pendulum.balls[selected_ball_index]
                    else:
                        self.wait = False
                elif self.mousepressed[0] and not self.wait:
                    self.wait = True

            '''if event.type == pg.KEYDOWN:
                if event.key == pg.K_p:
                    self.wait = True
                if event.key == pg.K_r:
                    self.wait = False'''
        if click[0] and self.wait and self.selected_ball is not None:
            self.selected_ball.position = vec(pg.mouse.get_pos())
            if self.selected_ball.id != 'pivot':
                self.selected_ball.theta = ((self.selected_ball.position -
                                             self.selected_ball.pivot.position).angle_to(vec(0, 1))) * pi / 180
                self.selected_ball.length = (self.selected_ball.position -
                                             self.selected_ball.pivot.position).length()
                self.selected_ball.thetaacc = 0
                self.selected_ball.thetavel = 0
            for ball in self.selected_pendulum.balls:
                if ball.id != 'pivot' and ball != self.selected_ball:
                    length = vec(ball.length * sin(ball.theta), ball.length * cos(ball.theta))
                    ball.position = length + ball.pivot.position
            for pendulum in self.pendulums:
                pendulum.particles = []

    def draw(self):
        # Game Loop - draw
        self.screen.fill(WHITE)
        for pendulum in reversed(self.pendulums):
            pendulum.draw()


g = Game()
while g.running:
    g.new()

pg.quit()
