import pyglet
from pyglet.gl import *
import math
import numpy as np

RAD2DEG = 57.29577951308232

class Attribute(object):
    def enable(self):
        raise NotImplementedError
    def disable(self):
        pass

class Transform(Attribute):
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1.0, 1.0)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)

    def enable(self):
        glPushMatrix()
        glTranslatef(self.translation[0], self.translation[1], 0)
        glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
        glScalef(self.scale[0], self.scale[1], 1)

    def disable(self):
        glPopMatrix()

    def set_translation(self, x, y):
        self.translation = (float(x), float(y))

    def set_rotation(self, theta):
        self.rotation = float(theta)

    def set_scale(self, x, y):
        self.scale = (float(x), float(y))

class Color(Attribute):
    def __init__(self, vec4):
        self.vec4 = vec4
    def enable(self):
        glColor4f(*self.vec4)

class LineStyle(Attribute):
    def __init__(self, style):
        self.style = style

    def enable(self):
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(1, self.style)

    def disable(self):
        glDisable(GL_LINE_STIPPLE)

class LineWidth(Attribute):
    def __init__(self, stroke):
        self.stroke = stroke

    def enable(self):
        glLineWidth(self.stroke)


class Geometry(object):
    def __init__(self):
        self.color = Color((0, 0, 0, 1.0))
        self.attrs = [self.color]

    def render(self):
        for attr in self.attrs:
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()

    def render1(self):
        raise NotImplementedError

    def add_attr(self, attr):
        self.attrs.append(attr)

    def set_color(self, vec4):
        self.color.vec4 = vec4

class Point(Geometry):
    def __init__(self):
        Geometry.__init__(self)

    def render1(self):
        glBegin(GL_POINTS)
        glVertex2f(0.0, 0.0)
        glEnd()

class Line(Geometry):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0)):
        Geometry.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        glBegin(GL_LINES)
        glVertex2f(*self.start)
        glVertex2f(*self.end)
        glEnd()

    def set_linewidth(self, x):
        self.linewidth.stroke = x

class PolyLine(Geometry):
    def __init__(self, vertices, is_close):
        Geometry.__init__(self)
        self.vertices = vertices
        self.is_close = is_close
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        glBegin(GL_LINE_LOOP if self.is_close else GL_LINE_STRIP)
        for vertex in self.vertices:
            glVertex2f(vertex[0], vertex[1])
        glEnd()

    def set_linewidth(self, x):
        self.linewidth.stroke = x

class Polygon(Geometry):
    def __init__(self, vertices):
        Geometry.__init__(self)
        self.vertices = vertices

    def render1(self):
        if len(self.vertices) == 4:
            glBegin(GL_QUADS)
        elif len(self.vertices) > 4:
            glBegin(GL_POLYGON)
        else:
            glBegin(GL_TRIANGLES)
        for vertex in self.vertices:
            glVertex2f(vertex[0], vertex[1])
        glEnd()

class Circle(Geometry):
    def __init__(self, radius, steps=30, filled=True):
        Geometry.__init__(self)
        self.filled = filled
        self.radius = radius
        self.vertices = [(math.cos(2*math.pi*i/steps)*radius, math.sin(2*math.pi*i/steps)*radius) for i in range(steps)]

    def render1(self):
        if self.filled:
            if len(self.vertices) == 4:
                glBegin(GL_QUADS)
            elif len(self.vertices) > 4:
                glBegin(GL_POLYGON)
            else:
                glBegin(GL_TRIANGLES)
            for vertex in self.vertices:
                glVertex2f(vertex[0], vertex[1])
            glEnd()
        else:
            glBegin(GL_LINE_LOOP)
            for vertex in self.vertices:
                glVertex2f(vertex[0], vertex[1])
            glEnd()

class FrameCircle(Geometry):
    def __init__(self, radius, steps=30):
        Geometry.__init__(self)
        self.radius = radius
        self.vertices = [(math.cos(2*math.pi*i/steps)*radius, math.sin(2*math.pi*i/steps)*radius) for i in range(steps)]

    def render1(self):
        if len(self.vertices) == 4:
            glBegin(GL_QUADS)
        elif len(self.vertices) > 4:
            glBegin(GL_POLYGON)
        else:
            glBegin(GL_TRIANGLES)
        for vertex in self.vertices:
            glVertex2f(vertex[0], vertex[1])
        glEnd()

        frame_color = Color((0, 0, 0, 1))
        frame_color.enable()
        glBegin(GL_LINE_LOOP)
        for vertex in self.vertices:
            glVertex2f(vertex[0], vertex[1])
        glEnd()
        frame_color.disable()

class Square(Geometry):
    def __init__(self, length, filled=True):
        Geometry.__init__(self)
        self.filled = filled
        self.length = length
        self.vertices = [(length/2.0, length/2.0), (-length/2.0, length/2.0), (-length/2.0, -length/2.0), (length/2.0, -length/2.0)]

    def render1(self):
        if self.filled:
            glBegin(GL_QUADS)
            for vertex in self.vertices:
                glVertex2f(vertex[0], vertex[1])
            glEnd()
        else:
            glBegin(GL_LINE_LOOP)
            for vertex in self.vertices:
                glVertex2f(vertex[0], vertex[1])
            glEnd()

class FrameSquare(Geometry):
    def __init__(self, length):
        Geometry.__init__(self)
        self.length = length
        self.vertices = [(length/2.0, length/2.0), (-length/2.0, length/2.0), (-length/2.0, -length/2.0), (length/2.0, -length/2.0)]

    def render1(self):
        glBegin(GL_QUADS)
        for vertex in self.vertices:
            glVertex2f(vertex[0], vertex[1])
        glEnd()
        frame_color = Color((0, 0, 0, 1))
        frame_color.enable()
        glBegin(GL_LINE_LOOP)
        for vertex in self.vertices:
            glVertex2f(vertex[0], vertex[1])
        glEnd()
        frame_color.disable()


class Viewer(object):
    def __init__(self, width, height, display=None):
        self.width = width
        self.height = height
        self.window = pyglet.window.Window(width=width, height=height, display=display)
        self.window.on_close = self.window_closed_by_user
        self.is_open = True
        self.geoms = []
        self.onetime_geoms = []
        self.transform = Transform()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.is_open = False

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scale_x = self.width/(right-left)
        scale_y = self.height/(top-bottom)
        self.transform = Transform(translation=(-left*scale_x, -bottom*scale_y), scale=(scale_x, scale_y))

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def render(self, return_rgb_array=False):
        glClearColor(1, 1, 1, 1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.transform.enable()
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()
        self.transform.disable()
        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.fromstring(image_data.data, dtype=np.uint8, sep="")
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.window.flip()
        self.onetime_geoms = []
        return arr if return_rgb_array else self.is_open

    def __del__(self):
        self.close()


if __name__=="__main__":

    v = Viewer(800, 600)
    ball = Circle(10)
    t = Transform(translation=(400, 300))
    ball.add_attr(t)

    t2 = Transform(translation=(0, 0))
    square = Square(100)
    square.add_attr(t2)
    v.add_geom(ball)
    v.add_geom(square)


    for i in range(60):
        t.set_translation(400+i, 300+i)
        t2.set_translation(0+i, 0+i)
        v.render()
