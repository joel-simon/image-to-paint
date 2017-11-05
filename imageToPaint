#!/usr/bin/env python

from __future__ import print_function
import os
from PIL import Image, ImageFilter, ImageOps, ImageChops
import random, pygame, sys, math, colorsys
import argparse
pygame.init()

#################################CONSTANTS######################################
sobelkernalx = [[-1,0,1],
                [-2,0,2],
                [-1,0,1]]

sobelkernaly = [[-1,-2,-1],
                [0,0,0],
                [1,2,1]]

####################################CLASSES#####################################
# Gassian blur in PIL is hardcoded to 2.. ???
# Redid it without this problem.
class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"
    def __init__(self, radius):
        self.radius = radius

    def filter(self, image):
        return image.gaussian_blur(self.radius)

#Class to hold a stroke. Consits of a color, brush radius and move list
class Stroke():
    def __init__(self, color, radius):
        self.color = color
        self.radius = radius
        self.move_list = []

    def addPoint(self, point):
        assert(len(point) == 2)
        self.move_list.append(point)

    # def printStrokes(self):
    #     f.write(str(self.color))
    #     f.write(str(self.move_list))
    #     f.write("\n")

#Image class containing pil image object, image pixel access object and methods.
class MyImage():
    def __init__(self, image):
        self.image = image
        self.array = image.load()
        (self.width, self.height) = image.size

    def save(self, name):
        self.image.save(name)

    def getPixel(self,x,y):
        assert(x < self.image.size[0])
        assert(y < self.image.size[1])
        assert((x>=0) and (y>=0))
        return self.array[x,y]

    def setPixel(self,x,y, v):
        self.array[x,y] = (int(v),)

    def getSize(self):
        return self.image.size

    def getMode(self):
        return self.image.mode

# OK, this one is slightly tricky. We want to GET a 2d vector array of the
# Sobel filtered luminance of a given image.
# steps- image -> luminace image -> Sobel fiter with a kernal of 3
# also,  we are expecting a gaussian blurred image.
class Gradient():
    def __init__(self, im):
        lumImage = MyImage(Image.new('L',im.getSize(), (0)))
        self.makeLuminenceOf(lumImage, im)
        assert(lumImage.getSize() == im.getSize())

        self.size = im.getSize()
        self.array = self.sobelFilter(lumImage)

        assert(len(self.array) == im.getSize()[1])
        assert(len(self.array[0]) == im.getSize()[0])

    #take a pixel array and return a float array
    def makeLuminenceOf(self, lumImage, rgbImage):
        for y in range(lumImage.getSize()[1]):
            for x in range(lumImage.getSize()[0]):
                p = rgbImage.array[x,y]
                lumImage.setPixel(x,y, luminosity(p))

    #take a regular 2d list of floats and return a 2d list of
    #(dx,dy) float tuples
    def sobelFilter(self, image):
        assert (image.getMode() == 'L')
        result = []
        row = []
        for Y in range(0,image.getSize()[1]):
            for X in range(0,image.getSize()[0]):
                if (Y==0 or Y >= image.getSize()[1]-1 or
                    X==0 or X >= image.getSize()[0]-1):
                    v_x = 0
                    v_y = 0
                else:
                    for y in range(-1, 2):
                        for x in range(-1, 2):
                            scalar_x = sobelkernalx[y+1][x+1]
                            v_x += image.getPixel(X+x, Y+y) * scalar_x

                            scalar_y = sobelkernaly[y+1][x+1]
                            v_y += image.getPixel(X+x, Y+y) * scalar_y
                if (v_x ==0): v_x =1
                if (v_y ==0): v_y =1
                row += [(v_x, v_y)]
                v_x = 0
                v_y = 0
            result += [row]
            row = []

        return result

    def getMag(self, x, y):
        self.REQUIRES(x,y)
        vector = self.array[y][x]
        sum = (abs(vector[0])+ abs(vector[1]))
        return sum

    def getUnitVector(self, x , y ):
        self.REQUIRES(x,y)
        vector = self.array[y][x]
        mag = (vector[0]**2 + vector[1]**2)**.5
        return(vector[0]/mag, vector[1]/mag)

    def getDirection(self, x,y):
        self.REQUIRES(x,y)
        (dx, dy) = self.getUnitVector(x,y)
        return math.atan2(dy, dx)

    def getWidth(self):
        return self.size[0]

    def getHeight(self):
        return self.size[1]

    def REQUIRES(self, x,y):
        assert(y >=0 and x >=0)
        assert(x < self.getWidth())
        assert(y < self.getHeight())

################################################################################

def luminosity(p):
    return .3*p[0] + .59*p[1] + .11*p[2]

def rgb_to_XYZ(rgb):
    var_R = ( rgb[0]/ 255.0 )        #R from 0 to 255
    var_G = ( rgb[1] / 255.0 )        #G from 0 to 255
    var_B = ( rgb[2] / 255.0 )        #B from 0 to 255
    if ( var_R > 0.04045 ):
        var_R = ( ( var_R + 0.055 ) / 1.055 ) ** 2.4
    else:
        var_R = var_R / 12.92
    if ( var_G > 0.04045 ):
        var_G = ( ( var_G + 0.055 ) / 1.055 ) ** 2.4
    else:
        var_G = var_G / 12.92
    if ( var_B > 0.04045 ):
        var_B = ( ( var_B + 0.055 ) / 1.055 ) ** 2.4
    else:
        var_B = var_B / 12.92
    var_R = var_R * 100
    var_G = var_G * 100
    var_B = var_B * 100
    X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
    Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
    Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505

    return X, Y, Z

def XYZ_to_Lab(XYZ):
    var_X = XYZ[0] / 95.047
    var_Y = XYZ[1] /100.000
    var_Z = XYZ[2] /108.883

    if (var_X > 0.008856 ):
        var_X = var_X ** ( 1.0/3 )
    else:
        var_X = ( 7.787 * var_X ) + ( 16.0 / 116 )
    if (var_Y > 0.008856 ):
        var_Y = var_Y ** ( 1.0/3 )
    else:
        var_Y = ( 7.787 * var_Y ) + ( 16.0 / 116 )
    if (var_Z > 0.008856 ):
        var_Z = var_Z ** ( 1.0/3 )
    else:
        var_Z = ( 7.787 * var_Z ) + ( 16.0 / 116 )

    L = ( 116 * var_Y ) - 16
    a = 500 * ( var_X - var_Y )
    b = 200 * ( var_Y - var_Z )

    return L, a, b

def rgb_to_lab(rgb):
    return XYZ_to_Lab(rgb_to_XYZ(rgb))

# def rgb_to_YUV(a):
#     R = a[0]
#     G = a[1]
#     B = a[2]
#     Y =  (0.257 * R) + (0.504 * G) + (0.098 * B) + 16
#     V =  (0.439 * R) - (0.368 * G) - (0.071 * B) + 128
#     U = -(0.148 * R) - (0.291 * G) + (0.439 * B) + 128

#     return Y, U, V

def difference(p1, p2):
    """ Take two LAB colors and return float.
    """
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)**.5

def colorDistance(a, b):
    """ Take two colors and return float.
    """
    a_p = rgb_to_lab(a)
    b_p = rgb_to_lab(b)

    return difference(z_p, b_p)

def closestColor(hsv_colors, s):
    return min(hsv_colors, key=lambda c: colorDistance(c, s))

def areaError(x_0, y_0, array, grid):
    s = 0
    for x in range(x_0-grid//2, x_0+grid//2):
        for y in range(y_0-grid//2, y_0+grid//2):
            s += array[y][x]
    return s/(grid**2)

def largestDif(x_0,y_0, array, grid):
    dmax = 0
    X, Y = 0, 0

    for x in range(x_0 - grid//2, x_0 + grid//2):
        for y in range(y_0 - grid//2, y_0 + grid//2):
            if array[y][x] > dmax:
                dmax = array[y][x]
                (X, Y) = (x, y)

    return (X,Y)


def diffArray(arrayA, arrayB, w, h):
    diff_array = []
    diff_row = []
    for y in range(h):
        for x in range(w):
            diff_row += [difference(arrayA[x,y], arrayB[x,y])]
        diff_array += [diff_row]
        diff_row = []
    return diff_array

#stroke , stroke -> int
# def strokeLum(s):
#     return
#stroke.color = colorsys.hsv_to_rgb(best_fit[0],best_fit[1], best_fit[2])

#stroke.color = (stroke.color[0] * 255,stroke.color[1] * 255,stroke.color[2] * 255)
################################################################################

def start_stroke(x_0, y_0, r, canvas, blurImage, gradient, config):
    stroke_color = blurImage.getPixel(x_0, y_0)
    stroke = Stroke(stroke_color, r)

    f_c = config['f_c']

    stroke.addPoint((x_0, y_0))
    x, y = x_0, y_0
    lastDx, lastDy = 0, 0

    for i in range(config['max_len']):
        if (config['f_l']*difference(blurImage.getPixel(x,y), canvas.getPixel(x, y)) < \
            difference(blurImage.getPixel(x, y), stroke_color)):
            return stroke

        if ((i > config['min_len']) and gradient.getMag(x, y) < 3):
            return stroke

        gx, gy = gradient.getUnitVector(x,y)
        dx, dy = -gy, gx

        if lastDx * dx + lastDy * dy < 0:
            dx, dy = -dx, -dy

        dx, dy = ((f_c * dx) + (1-f_c)*lastDx, (f_c * dy)+ (1-f_c)*lastDx)
        mag = max(1,(dx**2 + dy**2)**.5)
        dx, dy = dx / mag, dy / mag

        x, y = (int(x+r*dx), int(y+r*dy))
        lastDx, lastDy = dx, dy

        # Stop if stoke has gone out of bounds.
        if x < 0 or y < 0 or x >= gradient.getWidth() or y >= gradient.getHeight():
            return stroke

        stroke.addPoint((x, y))

    return stroke

def _to_RGB_tuple(i):
    return ((i>>16)/255.0, ((i>> 8)& 255)/255.0, (i & 255)/255.0)

def to_RGB_tuple(i):
    return ((i>>16), ((i>> 8)& 255), (i & 255))

# def generate_hsv():
#     global hsv
#     for i in colors:
#         t = to_RGB_tuple(i)
#         hsv +=[t]#[colorsys.rgb_to_hsv(t[0],t[1],t[2])]

def paintLayer(canvas, image, r, config):
    #the canvas will start blank on the first itereation, after that it
    #contains the previous layers.
    #Reference image is a copy of the original blurred to the r kernal.

    strokes = []

    blurImage = MyImage(image.image.copy())
    blurImage.image = blurImage.image.filter(MyGaussianBlur(4*r))

    dif_array = diffArray(canvas.array, blurImage.array, *canvas.getSize())
    gradient = Gradient(blurImage)

    grid = r * config['f_grid']

    for x in range(grid+1, canvas.getSize()[0] - grid, grid):

        for y in range(grid+1, canvas.getSize()[1] - grid, grid):

            error = areaError(x, y, dif_array, grid)

            if error > config['t']:
                x_s, y_s = largestDif(x, y, dif_array, r)
                stroke = start_stroke(x_s, y_s, r, canvas, blurImage, gradient, config)
                strokes.append(stroke)


    if config['lightdark']:
        strokes = sorted(strokes, key=lambda s: luminosity(s.color), reverse=True)
    else:
        random.shuffle(strokes)

    return strokes

def place_strokes(strokes, window, gcode_out, config, width, height):
    """ Take the strokes and write them to the pygame canvas and gcode file.
    """
    if gcode_out:
        f = open(gcode_out, 'w+')
        f.write('(Generated G-code by www.github.com/Sloth6/image-to-paint)\n')
        f.write('(********************************************************)\r')
        f.write(config['gc_tool']+'\rF'+str(config['gc_feed'])+'.\r')

    xScale = config['gc_width'] / width
    yScale = config['gc_height'] / height

    for stroke in strokes:
        if len(stroke.move_list) < config['min_len']:
            continue

        if gcode_out:
            f.write('G1 Z.200\n')
            f.write('G1 X'+ str(stroke.move_list[0][0]*xScale)[:6]+
                    ' Y'+ str(stroke.move_list[0][1]*yScale)[:6]+'\r')
            f.write('G1 Z.00\n')

            for p in range(1, len(stroke.move_list)):
                f.write('G1 X'+ str(stroke.move_list[p][0]*xScale)[:6]+
                        ' Y'+ str(stroke.move_list[p][1]*yScale)[:6]+'\r')

        if config['colors'] != ['all']:
            stroke.color = closestColor(config['colors'], stroke.color)

        pygame.draw.lines(window, stroke.color, False, stroke.move_list, stroke.radius)

    if gcode_out:
        f.close()

def main(config):
    image = MyImage(Image.open(config['in']))
    width, height = image.getSize()
    canvas = MyImage(Image.new('RGB', image.getSize(), (255,255,255)))
    window = pygame.display.set_mode(image.getSize())
    window.fill((255, 255, 255))

    strokes_used = 0

    for r in sorted(config['brushes'], reverse=True):

        gcode_out = None
        if config['gcode']:
            out_dir = os.path.dirname(config['out'])
            gcode_out = os.path.join(out_dir, 'strokes_%i.txt'%r)

        strokes = paintLayer(canvas, image, r, config)
        place_strokes(strokes, window, gcode_out, config, width, height)
        strokes_used += len(strokes)

        print("Completed brush size %i - %i strokes used" % (r, len(strokes)))

    pygame.image.save(window, config['out'])

    print("Completed: image uses ", strokes_used, " brushstrokes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in", help="Path to input image.")
    parser.add_argument("out", help="Path to where to save output image.")

    # Algorithm options.
    parser.add_argument("--brushes", default=[4], nargs='+', help="Size of brushes strokes in pixels.", type=int)
    parser.add_argument("--max_len", default=60, help="Maximum stroke Length.", type=int)
    parser.add_argument("--min_len", default=5, help="Maximum stroke Length.", type=int)
    parser.add_argument("--f_grid", default=6, help="How spaced apart new strokes will be, 1 = 1*radius_stroke apart.", type=int)
    parser.add_argument("--colors", default=["all"], nargs='+', help="Use a limited color palate.")
    parser.add_argument("--lightdark", default=True, help="Paint brush strokes in increasing brightness.", type=bool)
    parser.add_argument("--t", default=10.0, help="Threshold to create a new stroke.", type=float)
    parser.add_argument("--f_l", default=0.75, help="Higher values will result in longer, less acurate brushstrokes.", type=float)
    parser.add_argument("--f_c", default=1.0, help="Curvature modifier.", type=float)

    # GCODE options.
    parser.add_argument("--gcode", dest='gcode', action='store_true')
    parser.set_defaults(gcode=False)
    parser.add_argument("--gc_feed", default=600, help="GCODE feedrate..", type=int)
    parser.add_argument("--gc_tool", default="T1", help="GCODE tool.", type=str)
    parser.add_argument("--gc_width", default=46.0, help="Width (in inches) for gcode.", type=float)
    parser.add_argument("--gc_height", default=46.0, help="Height (in inches) for gcode.", type=float)

    args = parser.parse_args()
    main(vars(args))
