import pygame as pg
from math import *
import random
from spring import spring

vec = pg.Vector2


N = 1
GRAVITY = 0.2
WIDTH = 800
HEIGHT = 800
TITLE = 'Langrnagian Sring Pendulum'
FPS = 120
WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)
YELLOW = (255, 255, 0, 255)
GREEN = (0, 255, 0, 255)
RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)
LIGHTBLUE = (100, 100, 255, 255)
PIVOT_R = 10
PIVOT = (400, 200)
BOB_R = 20
BOB_M = 5
INITIALS = [(100, 2, 20, 30)]
SPRING_W = 5
SPRING_K = 0.005
a = .5
SPRING_DOTS = [(0, 1), (a/2, 0.5), (-a, 0.5), (a, 0.5), (-a, 0.5), (a, 0.5), (-a, 0.5), (a, 0.5), (-a, 0.5),
               (a, 0.5), (-a/2, 0.5), (0, 1)]
SPRING_COLOR = BLACK
SPRING_NODES = 20
SPRING_A = 50
P_DESINTEGRATION = 1
PARTICLE_WIDTH = 3
SELECTION_THREASHOLD = 10
FRICTION = 0.1
N_BALLS = 2


def change(c):
    *arr, a = [x / 255 for x in c]
    return [round((x * a + 1 * (1 - a)) * 255) for x in arr]


class Ball:
    def __init__(self, s, t, ln, min_ln, i, game):
        self.game = game
        self.id = i
        self.theta = t
        self.thetavel = 0
        self.thetaacc = 0
        self.s = s * self.game.n
        self.svel = 0 * self.game.n
        self.sacc = 0 * self.game.n
        self.pos = self.s * vec(sin(self.theta), cos(self.theta)) * self.game.n + self.game.balls[self.id - 1].pos
        self.r = BOB_R * self.game.n
        self.m = BOB_M * self.game.n
        self.color = (random.randint(10, 255), random.randint(10, 255), random.randint(10, 255), 255)
        self.pivot = self.game.balls[self.id-1]
        self.spring = []
        self.spring_l = ln * self.game.n
        self.min_spring_l = min_ln * self.game.n
        ind = 0
        for dot in SPRING_DOTS:
            ar, b = dot
            self.spring.append(vec(ar, b) * (self.spring_l / 7) * self.game.n)
            ind += 1
        self.spring_w = SPRING_W * self.game.n
        self.spring_k = SPRING_K * self.game.n
        self.spring_nodes = SPRING_NODES
        self.spring_a = SPRING_A * self.game.n
        self.spring_color = SPRING_COLOR
        self.particles = []
        self.p_width = PARTICLE_WIDTH * self.game.n

    def update(self):
        self.thetaacc = (- self.game.gravity * sin(self.theta) - 2 * self.svel * self.thetavel) / self.s
        self.sacc = self.s * (self.thetavel ** 2) + self.game.gravity * cos(self.theta) + (self.spring_k / self.m) \
            * (self.spring_l - self.s)
        self.thetavel += self.thetaacc
        self.theta += self.thetavel
        self.svel += self.sacc
        if abs(self.s + self.svel) < self.min_spring_l:
            self.svel = 0
        self.s += self.svel
        self.pos = self.s * vec(sin(self.theta), cos(self.theta)) + self.game.balls[self.id - 1].pos
        self.particles.append((self.pos, self.color, self.p_width))  # (abs(int(self.p_width * self.svel))) + 1)

    def draw(self):
        x, y = spring((self.pivot.pos.x, self.pivot.pos.y), (self.pos.x, self.pos.y), self.spring_nodes,
                      round(self.spring_a))
        points = list(zip(x, y))
        pg.draw.lines(self.game.screen, self.spring_color, False, points, round(self.spring_w))
        '''new_spring = []
        h = (self.s - self.spring_l) / (len(self.spring))
        for d in self.spring:
            # change the spring according to the ball
            br = vec(d.x, d.y + h)
            new_spring.append(br.rotate_rad(-self.theta))
        world_spring = [self.pivot.pos]
        index = 1
        for d in new_spring:
            # change the sprign coordinates to world coordinates:
            ar = world_spring[index - 1] + d
            world_spring.append(ar)
            index += 1
        pg.draw.lines(self.game.screen, self.spring_color, False, world_spring, self.spring_w)'''
        pg.draw.circle(self.game.screen, self.color, self.pos, self.r)

    def draw_particles(self):
        index = 0
        for particle in self.particles:
            pos, color, width = particle
            r, gr, bl, al = color
            newcolor = (r, gr, bl, al - P_DESINTEGRATION)
            self.particles[index] = (pos, newcolor, width)
            if newcolor[3] < 0:
                self.particles.pop(index)
            else:
                cr = change(newcolor)
                if index > 1:
                    pg.draw.line(self.game.screen, cr, pos, self.particles[index - 1][0], round(width))
            index += 1


class Pivot:
    def __init__(self, i, game):
        self.id = i
        self.game = game
        self.pos = vec(PIVOT)
        self.r = PIVOT_R * self.game.n

    def update(self):
        pass

    def draw(self):
        pg.draw.circle(self.game.screen, RED, self.pos, self.r)

    def draw_particles(self):
        pass


class Game:
    def __init__(self):
        pg.init()
        pg.mixer.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        self.n = N
        self.gravity = GRAVITY * self.n
        self.running = True
        self.playing = True
        self.friction = FRICTION
        self.balls = []
        self.bobr = BOB_R * self.n
        self.n_balls = len(INITIALS)
        self.mousepressed = None
        self.selection = False
        self.selected_index = None
        self.wait = False

    def new(self):
        # start a new game
        self.balls.append(Pivot(0, self))
        index = 0
        for b in range(self.n_balls):
            self.balls.append(Ball(INITIALS[index][0] * self.n, INITIALS[index][1] * self.n,
                                   INITIALS[index][2] * self.n, INITIALS[index][3] * self.n, b+1, self))
            index += 1

        for ball in reversed(self.balls):
            if ball.id != self.n_balls and ball.id != 0:
                ball.m += self.balls[ball.id + 1].m
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
        for ball in self.balls:
            ball.update()

    def events(self):
        # Game Loop - events
        click = pg.mouse.get_pressed(3)
        for event in pg.event.get():
            # check for closing window
            if event.type == pg.QUIT:
                if self.playing:
                    self.playing = False
                self.running = False
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_UP:
                    self.n = 1.1
                    self.zoom()
                if event.key == pg.K_DOWN:
                    self.n = 1/1.1
                    if not round(self.balls[0].r * self.n) < 2:
                        self.zoom()
            if event.type == pg.MOUSEBUTTONDOWN:
                self.mousepressed = pg.mouse.get_pressed(3)
                if self.mousepressed[0] and self.wait:
                    self.selection = None
                    mouse = vec(pg.mouse.get_pos())
                    points = []
                    indexes = []
                    index = 0
                    for ball in self.balls:
                        indexes.append(index)
                        points.append(ball.pos)
                        index += 1
                    d = [(point - mouse).length() for point in points]
                    if min(d) < SELECTION_THREASHOLD + BOB_R:
                        self.selected_index = indexes[d.index(min(d))]
                        self.selection = self.balls[self.selected_index]
                    else:
                        self.wait = False
                elif self.mousepressed[0] and not self.wait:
                    self.wait = True
        if click[0] and self.wait and self.selection:
            self.selection.pos = vec(pg.mouse.get_pos())
            if self.selection.id != 0:
                self.selection.theta = ((self.selection.pos - self.selection.pivot.pos).angle_to(vec(0, 1)))\
                                    * pi / 180
                self.selection.s = (self.selection.pos - self.selection.pivot.pos).length()
                self.selection.thetaacc = 0
                self.selection.thetavel = 0
            for ball in reversed(self.balls):
                if ball.id != 0 and ball != self.selection:
                    length = vec(ball.s * sin(ball.theta), ball.s * cos(ball.theta))
                    ball.pos = length + ball.pivot.pos
            for ball in self.balls:
                ball.particles = []

    def draw(self):
        # Game Loop - draw
        self.screen.fill(WHITE)
        # draw particles
        for ball in reversed(self.balls):
            ball.draw_particles()
        # draw the balls and springs
        for ball in reversed(self.balls):
            ball.draw()

    def zoom(self):
        self.gravity *= self.n
        self.bobr *= self.n
        for ball in self.balls:
            ball.r *= self.n
            if ball.id != 0:
                ball.s *= self.n
                ball.svel *= self.n
                ball.sacc *= self.n
                ball.pos = ball.pivot.pos + vec(ball.s*sin(ball.theta), ball.s*cos(ball.theta))
                ball.m *= self.n
                ball.spring_l *= self.n
                ball.min_spring_l *= self.n
                ind = 0
                s = []
                for dot in ball.spring:
                    s.append(dot * self.n)
                    ind += 1
                ball.spring = s
                ball.spring_w *= self.n
                ball.spring_k *= self.n
                ball.spring_a *= self.n
                ball.p_width *= self.n
                index = 0
                for particle, color, width in ball.particles:
                    s = self.balls[0].pos - (self.balls[0].pos - particle) * self.n
                    ball.particles[index] = (s, color, width*self.n)
                    index += 1


g = Game()
while g.running:
    g.new()

pg.quit()
