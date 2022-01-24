import pygame as pg
from pygame import Vector2 as Vec
from colors import *
from math import *


N = 1
GRAVITY = 2

WIDTH = 600
HEIGHT = 600
TITLE = 'Langrangian Ball in Stick'
FPS = 60

FLOOR = (0, 275, 600, 50)
PIVOT = (300, 275)
CENTER_R = 10
CENTER = (300, 275)
STICK_L = 500
STICK_W = 10
INITIAL_S = 250
BALL_R = 20
INITIAL_THETA = 0.4
INITIAL_THETAVEL = 0

P_DESINTEGRATION = 2
PARTICLE_WIDTH = 5
FRICTION = 1

SELECTION_THREASHOLD = 10

BALLS = [(1, GREEN), (-1, PURPLE)]


def change(c):
    *arr, a = [x / 255 for x in c]
    return [round((x * a + 1 * (1 - a)) * 255) for x in arr]


class Ball:
    def __init__(self, game, x, c, i):
        self.id = i
        self.game = game
        self.x = x
        self.s = self.x*INITIAL_S*self.game.n
        self.pos = self.game.pivot + Vec(self.s*cos(self.game.theta), self.s*sin(self.game.theta))
        self.sacc = 0
        self.svel = 0
        self.r = BALL_R*self.game.n
        self.particles = []
        self.p_color = c
        self.particle_w = PARTICLE_WIDTH*self.game.n

    def update(self):
        self.sacc = self.game.gravity*sin(self.game.theta) + self.s*(self.game.thetavel**2)
        self.svel += self.sacc
        if self.x < 0:
            if not (-self.game.stick_l/2 <= self.s + self.svel <= 0):
                self.svel = 0
        elif self.x > 0:
            if not (0 <= self.s + self.svel <= self.game.stick_l/2):
                self.svel = 0
        self.s += self.svel
        self.svel *= FRICTION
        self.pos = self.game.pivot + self.game.stick.normalize()*self.s
        self.particles.append((self.pos, self.p_color))

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
                if index > 1:
                    pg.draw.line(self.game.screen, cr, pos, self.particles[index - 1][0],
                                 round(self.particle_w))
            index += 1
        pg.draw.circle(self.game.screen, self.p_color, self.pos, self.r)


class Game:
    def __init__(self):
        pg.init()
        pg.mixer.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        self.n = N
        self.center = Vec(CENTER)
        self.center_r = CENTER_R*self.n
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        self.playing = True
        self.running = True
        self.gravity = GRAVITY*self.n
        self.floor_a, self.floor_b, self.floor_c, self.floor_d = FLOOR
        self.floor = pg.Rect(self.floor_a, self.floor_b, self.floor_c, self.floor_d)
        self.theta = INITIAL_THETA
        self.thetavel = INITIAL_THETAVEL
        self.thetaacc = 0
        self.stick_l = STICK_L*self.n
        self.stick = Vec(self.stick_l*cos(self.theta), self.stick_l*sin(self.theta))
        self.pivot = Vec(PIVOT)
        self.balls = []
        self.mousepressed = None
        self.selection = False
        self.wait = False
        self.stick_w = STICK_W*self.n

    def new(self):
        # start a new game
        i = 0
        for a, b in BALLS:
            self.balls.append(Ball(self, a, b, i))
            i += 1
        self.run()
        pass

    def run(self):
        # Game Loop
        self.playing = True
        while self.playing:
            self.clock.tick(FPS)
            self.events()
            self.update()
            self.draw()
            pg.display.flip()
            pg.display.update()

    def update(self):
        # Game Loop - Update
        if not self.wait:
            self.thetaacc = cos(self.theta)/100
            self.thetavel += self.thetaacc
            self.theta += self.thetavel
        self.stick = Vec(((self.stick.x - self.pivot.x) * cos(self.theta)) -
                         ((self.pivot.y - self.stick.y) * sin(self.theta)) + self.pivot.x,
                         self.pivot.y - ((self.pivot.y - self.stick.y) * cos(self.theta)) +
                         ((self.stick.x - self.pivot.x) * sin(self.theta)))
        self.stick = Vec(self.stick_l*cos(self.theta), self.stick_l*sin(self.theta))
        if not self.wait:
            for ball in self.balls:
                ball.update()
            self.thetavel *= FRICTION

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
                if self.mousepressed[0]:
                    self.wait = True
                    mouse = Vec(pg.mouse.get_pos())
                    d = self.balls[0].pos - mouse
                    if d.length() < SELECTION_THREASHOLD + BALL_R:
                        self.selection = True

            if event.type == pg.MOUSEBUTTONUP:
                if self.mousepressed[0] and self.selection:
                    self.selection = False
                    self.wait = False
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_UP:
                    self.n = 2
                    self.zoom()
                if event.key == pg.K_DOWN:
                    self.n = 1/2
                    if not round(self.floor_d * self.n) < 2:
                        self.zoom()

        if click[0] and self.selection:
            mouse = Vec(pg.mouse.get_pos())
            self.theta = -(mouse - self.pivot).angle_to(Vec(1, 0)) * pi / 180
            if (mouse - self.pivot).length() > STICK_L/2:
                self.balls[0].pos = self.pivot + Vec(self.stick_l*cos(self.theta)/2, self.stick_l*sin(self.theta)/2)
            else:
                self.balls[0].pos = mouse
            self.balls[0].s = (self.balls[0].pos - self.pivot).length()
            self.balls[0].svel = 0
            self.balls[0].sacc = 0
            self.balls[0].particles = []

    def draw(self):
        # Game Loop - draw
        self.screen.fill(WHITE)
        pg.draw.rect(self.screen, BLACK, self.floor)
        pg.draw.line(self.screen, BLUE, self.pivot - self.stick/2, self.pivot + self.stick/2, round(self.stick_w))
        pg.draw.circle(self.screen, RED, self.center, self.center_r)
        for ball in self.balls:
            ball.draw()

    def zoom(self):
        self.stick_w *= self.n
        self.gravity *= self.n
        self.stick_l *= self.n
        self.center_r *= self.n
        self.stick = Vec(self.stick_l * cos(self.theta), self.stick_l * sin(self.theta))
        self.floor_d *= self.n
        self.floor = pg.Rect(self.floor_a, self.floor_b, self.floor_c, self.floor_d)
        for ball in self.balls:
            ball.s *= self.n
            ball.pos = self.pivot + Vec(ball.s*cos(self.theta), ball.s*sin(self.theta))
            ball.r *= self.n
            ball.particle_w *= self.n
            particle: Vec
            index = 0
            for particle, color in ball.particles:
                s = self.pivot + (self.pivot - particle) * self.n
                ball.particles[index] = (s, color)
                index += 1


if __name__ == '__main__':
    g = Game()
    while g.running:
        g.new()

    pg.quit()
