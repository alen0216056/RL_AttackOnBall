import pyglet
from pyglet.gl import *
import numpy as np
import math
import random
import scipy.misc
import cv2

import renderer


class Base(object):
    def __init__(self, position=(0.0, 0.0), velocity=(0.0, 0.0)):
        self.position = position
        self.velocity = velocity
        self.transform = renderer.Transform(translation=position)

    def set_position(self, position):
        self.position = position
        self.transform.set_translation(*position)

    def set_velocity(self, velocity):
        self.velocity = velocity


class Player(Base):
    def __init__(self, position, length):
        Base.__init__(self, position=position)
        self.geom = renderer.FrameSquare(length)
        self.geom.add_attr(self.transform)
        self.geom.add_attr(renderer.LineWidth(3))
        self.geom.set_color((0.7725, 0.7686, 0.0313, 1))


class Ball(Base):
    def __init__(self, position, radius, color, velocity=(0.0, 0.0)):
        Base.__init__(self, position=position, velocity=velocity)
        self.radius = radius
        self.color = color
        self.geom = renderer.FrameCircle(radius)
        self.geom.set_color(color)
        self.geom.add_attr(renderer.LineWidth(3))
        self.geom.add_attr(self.transform)


class Bonus(Base):
    def __init__(self, position, score):
        Base.__init__(self, position=position)
        self.score = score
        self.on_bottom = False


class AttackOnBall:
    def __init__(self):
        # environment
        self.width = 960
        self.height = 430
        self.left = 0
        self.right = 960
        self.bottom = 0
        self.top = 430
        self.fps = 50
        self.delta_time = 0.02
        self.gravity = -190
        self.count = 0

        # player
        self.player_length = 50
        self.player = Player((self.width/2-self.player_length/2, self.player_length/2), self.player_length)

        # balls
        self.ball_colors = [
            (0.7494, 0.1255, 0.1255, 1),
            (0.9294, 0.6745, 0.0313, 1),
            (0.0117, 0.6000, 0.8000, 1),
            (0.7490, 0.3607, 0.8156, 1),
            (0.3882, 0.7333, 0.0588, 1)
        ]
        self.ball_radius_min = 40
        self.ball_radius_max = 60
        self.ball_velocity_min = 120
        self.ball_velocity_max = 180
        self.balls = []

        # bonus
        self.bonus_range = [1, 2, 3]
        self.bonus_position_range = [i * self.width/10 for i in range(1, 10, 1)]
        self.bonus = None

        # pyglet
        self.window = pyglet.window.Window(width=self.width, height=self.height)
        self.transform = renderer.Transform()
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def reset(self):
        self.player.set_position((self.width/2-self.player_length/2, self.player_length/2))
        self.balls = []
        self.bonus = None
        self.count = 0

        glClearColor(0.9686, 0.9372, 0.9294, 1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.transform.enable()
        self.player.geom.render()
        self.transform.disable()
        self.window.flip()
        return self.get_frame_buffer()

    def step(self, action):
        self.update_player(action)
        self.update_balls()
        self.update_bonus()
        self.count = (self.count + 1) % self.fps

        # new right ball
        if self.count == 0:
            radius = self.ball_radius_min + random.uniform(0.0, 1.0) * (self.ball_radius_max - self.ball_radius_min)
            velocity_x = self.ball_velocity_min + random.uniform(0.0, 1.0) * (self.ball_velocity_max - self.ball_velocity_min)
            color = self.ball_colors[random.randint(0, 4)]
            position = (self.right + radius - 10, random.uniform(self.bottom + self.height/2, self.top))
            self.balls.append(Ball(position, radius, color, (-velocity_x, 0)))

        # new left ball
        if self.count == 0:
            radius = self.ball_radius_min + random.uniform(0.0, 1.0) * (self.ball_radius_max - self.ball_radius_min)
            velocity_x = self.ball_velocity_min + random.uniform(0.0, 1.0) * (self.ball_velocity_max - self.ball_velocity_min)
            color = self.ball_colors[random.randint(0, 4)]
            position = (self.left - radius + 10, random.uniform(self.bottom + self.height/2, self.top))
            self.balls.append(Ball(position, radius, color, (velocity_x, 0)))

        # new bonus
        if not self.bonus and random.uniform(0.0, 1.0) < 0.005:
            self.bonus = Bonus((self.bonus_position_range[random.randint(0, 8)], self.height), self.bonus_range[random.randint(0, 2)])

        state = self.render(return_rgb=True)
        reward = self.delta_time
        done = False

        # check touch bonus
        if self.bonus:
            if math.fabs(self.bonus.position[0] - self.player.position[0]) < 10 + self.player_length/2:
                if math.fabs(self.bonus.position[1] - self.player.position[1]) < 20 + self.player_length/2:
                    reward += self.bonus.score
                    self.bonus = None

        # check touch ball
        for ball in self.balls:
            if self.distance(ball.position, self.player.position) < (self.player_length/2 + ball.radius) * (self.player_length/2 + ball.radius):
                done = True
                break

        return state, reward, done

    def distance(self, a, b):
        return (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1])

    def update_player(self, action):
        new_position = self.player.position[0] + (6.4 if action == 1 else -6.4)
        if new_position < self.left + self.player_length/2:
            new_position = self.left + self.player_length/2
        elif new_position > self.right - self.player_length/2:
            new_position = self.right - self.player_length/2
        self.player.set_position((new_position, self.player_length/2))

    def update_balls(self):
        for ball in self.balls:
            # delta y
            new_velocity = ball.velocity[1] + self.delta_time * self.gravity
            if ball.velocity[1] > 0 and new_velocity < 0:
                delta_y = (ball.velocity[1] * ball.velocity[1]/-self.gravity)/2 + (new_velocity * new_velocity/self.gravity)/2
            else:
                delta_y = (ball.velocity[1] + new_velocity) * self.delta_time / 2.0

            new_position = (ball.position[0] + self.delta_time * ball.velocity[0], ball.position[1] + delta_y)

            if new_position[0] < self.left - ball.radius or new_position[0] > self.right + ball.radius:
                self.balls.remove(ball)
                continue

            if new_position[1] < self.bottom + ball.radius:
                new_velocity = -new_velocity

            ball.set_position(new_position)
            ball.set_velocity((ball.velocity[0], new_velocity))

    def update_bonus(self):
        if self.bonus and not self.bonus.on_bottom:
            new_velocity = self.bonus.velocity[1] + self.delta_time * self.gravity
            new_y = self.bonus.position[1] + (self.bonus.velocity[1] + new_velocity) * self.delta_time / 2.0

            if new_y < self.bottom + 20:
                new_y = self.bottom + 20
                self.bonus.on_bottom = True

            self.bonus.set_position((self.bonus.position[0], new_y))
            self.bonus.set_velocity((0.0, new_velocity))

    def render(self, return_rgb=False):
        glClearColor(0.9686, 0.9372, 0.9294, 1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.transform.enable()

        self.player.geom.render()

        for ball in self.balls:
            ball.geom.render()

        if self.bonus:
            tmp = pyglet.text.Label(str(self.bonus.score),
                          font_name='Times New Roman',
                          font_size=40, bold=True, color=(123, 130, 254, 255),
                          x=self.bonus.position[0], y=self.bonus.position[1],
                          anchor_x='center', anchor_y='center')
            tmp.draw()

        self.transform.disable()
        self.window.flip()

        if return_rgb:
            return self.get_frame_buffer()
        return True

    def get_frame_buffer(self):
        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        image_data = buffer.get_image_data()
        result = np.fromstring(image_data.data, dtype=np.uint8, sep="")
        result = result.reshape(buffer.height, buffer.width, 4)
        result = result[::-1, :, 0:3]
        return result


if __name__=="__main__":
    game = AttackOnBall()

    state = game.reset()

    for i in range(50):
        total_reward = 0
        total_step = 0
        while True:
            state, reward, done = game.step(random.randint(0, 1))
            total_reward += reward
            total_step += 1
            if done:
                #scipy.misc.imsave("end{}.png".format(i), state)
                state = game.reset()
                break
                #scipy.misc.imsave("start{}.png".format(i), state)

            #print(state, reward, done)
        print("ep{}: reward: {}, step: {}".format(i, total_reward, total_step))
