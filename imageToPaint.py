from PIL import Image, ImageFilter, ImageOps, ImageChops
import random, pygame, sys, math, colorsys

#imageToPaint.py
#Joel S. Simon
#Dec 24, 2012

###################################OPTIONS######################################
brushes = [4] #resolutions of the brush strokes used in decreasing order
max_stroke_length = 50
min_stroke_length = 10
f_blur = 1
f_grid = 6 # how spaced apart strokes will be, 1 = 1*radius_stroke apart.
f_c = 1.0 #curvature modifier (weird results so far :( )
T =100 #Difference requires to start a stroke
f_l = 3.0 #Higher values will result in longer, less acurate brushstrokes.
randomStrokes = True
lightToDark = True #paint brush strokes in increasing or decreasing brightness
blackOnly = True
limitedColors = True
forcedFirstLayer = True
#berry, black, blue,Brown, Green, Turquoise, Lime Green,
#Orange, Magenta, purple, red, yellow 
#colors = [0x000000,
    #colors = [0x960090, 0x000000, 0x0040ff, 0x361500, 0x00ff00, 0x40ffed, 0x94ff33,
#0xff5900, 0xff00ff, 0x800080, 0xff0000, 0xffff00]
colors = [0xff0000, 0xAD5C2A, 0xCB9573, 0x362419]
#0x0000ff, 0x006400, 0xE8B569, 0xB9B1A3, 0xFFC690, 0xFAC592, 0x4B2F16, 0xC67E66]
hsv = []
###################################-GCODE-######################################
writeGcode = True
feedrate = '600'
tool = 'T1'
fileOut = 'strokes.txt'
canvasSize = (46.0, 46.0) #W, H in inches.

##################################GLOBALS#######################################
global strokesUsed
strokesUsed= 0

#################################CONSTANTS######################################
sobelkernalx = [[-1,0,1],[-2,0,2],[-1,0,1]]
sobelkernaly = [[-1,-2,-1],[0,0,0],[1,2,1]]

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
        self.move_list += [point]
    def printStrokes(self):
        f.write(str(self.color))
        f.write(str(self.move_list))
        f.write("\n")
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
        self.array[x,y] = v
    
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
        for y in xrange(lumImage.getSize()[1]):
            for x in xrange(lumImage.getSize()[0]):
                p = rgbImage.array[x,y]
                lumImage.setPixel(x,y, luminosity(p))
 
    #take a regular 2d list of floats and return a 2d list of
    #(dx,dy) float tuples
    def sobelFilter(self, image):
        assert (image.getMode() == 'L')
        result = []
        row = []
        for Y in xrange(0,image.getSize()[1]):
            for X in xrange(0,image.getSize()[0]):
                if (Y==0 or Y >= image.getSize()[1]-1 or
                    X==0 or X >= image.getSize()[0]-1):
                    v_x = 0
                    v_y = 0
                else:
                    for y in xrange(-1, 2):
                        for x in xrange(-1, 2):
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
        #print math.degrees(math.atan2(dy, dx))
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
def paint(sourceImage, brushes):
    canvas = MyImage(Image.new('RGB', sourceImage.getSize(), (255,255,255)))
    for r in brushes:
        referenceImage = MyImage(sourceImage.image.copy())
        referenceImage.image = referenceImage.image.filter(MyGaussianBlur(4*r))
        canvas = paintLayer(canvas, referenceImage, r)
    return canvas

def areaError(x_0, y_0, array, grid):
    sum = 0
    for x in xrange(x_0-grid/2, x_0+grid/2):
        for y in xrange(y_0-grid/2, y_0+grid/2):
            sum += array[y][x]
    return sum/(grid**2)

def largestDif(x_0,y_0, array, grid):
    max = 0
    (X,Y) = (0,0)
    for x in xrange(x_0 - grid/2, x_0 + grid/2):
        for y in xrange(y_0 - grid/2, y_0 + grid/2):
            if array[y][x] > max:
                max = array[y][x]
                (X, Y) = (x, y)
            
    return (X,Y)

# rgb tuple -> float
def luminosity(p):
    return .3*p[0] + .59*p[1] + .11*p[2]

# pixel , pixel -> float
def difference(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)**.5

def diffArray(arrayA, arrayB, (w, h)):
    diff_array = []
    diff_row = []
    for y in xrange(h):
        for x in xrange(w):
            diff_row += [difference(arrayA[x,y], arrayB[x,y])]
        diff_array += [diff_row]
        diff_row = []
    return diff_array

#stroke , stroke -> int
def strokeCmp(a, b):
    if(luminosity(b.color)>luminosity(a.color)):
        return -1^lightToDark
    return 1^lightToDark

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
    return(X,Y,Z)

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
    return(L,a,b)

def rgb_to_lab(rgb):
    return XYZ_to_Lab(rgb_to_XYZ(rgb))

def rgb_to_YUV(a):
    R = a[0]
    G = a[1]
    B = a[2]
    Y =  (0.257 * R) + (0.504 * G) + (0.098 * B) + 16
    V =  (0.439 * R) - (0.368 * G) - (0.071 * B) + 128
    U = -(0.148 * R) - (0.291 * G) + (0.439 * B) + 128
    return (Y,U,V)

def colorDistance(a, b):
    a_p = rgb_to_lab(a)
    b_p = rgb_to_lab(b)
    return difference(a_p,b_p)
#((((512+rmean)*r*r)>>8) + 4*g*g + (((767-rmean)*b*b)>>8))**(.5)

def closestColor(s):
    assert(len(hsv)>2)
    best_fit = 0
    best_value = 0x13337
    for i in hsv:
        diff = colorDistance(i,s)#difference(i, s)
        
        if(diff < best_value):
            best_fit = i
            best_value = diff
    #print best_value, best_fit
    return best_fit
#stroke.color = colorsys.hsv_to_rgb(best_fit[0],best_fit[1], best_fit[2])
#stroke.color = (stroke.color[0] * 255,stroke.color[1] * 255,stroke.color[2] * 255)

def paintLayer (canvas, referenceImage, r):
    #the canvas will start blank on the first itereation, after that it
    #contains the previous layers.
    #Reference image is a copy of the original blurred to the r kernal. 
    global strokesUsed
    assert(canvas.getSize() == referenceImage.getSize())
    strokes = []
    dif_array = diffArray(canvas.array, referenceImage.array, canvas.getSize())
    gradient = Gradient(referenceImage)
    #referenceImage.image.save('lum+blur.png')
    grid = r*f_grid
    for x in xrange(grid+1,canvas.getSize()[0]-grid, grid):
        for y in xrange(grid+1,canvas.getSize()[1]-grid, grid):
            error = areaError(x, y, dif_array, grid)
            if (error > T):
                (x_s,y_s) = largestDif(x,y,dif_array,r)
                stroke = makeStroke(x_s, y_s, r, canvas, referenceImage, gradient)
                strokes += [stroke]
    
    if(writeGcode):
        f = open(fileOut, 'w')
        f.write('(Generated G-code by Joel Simon)\n')
        f.write('(********************************************************)\r')
        f.write(tool+'\rF'+feedrate+'.\r')
        scalar = 1.0
    
    if(randomStrokes):
        strokes = random.sample(strokes, len(strokes))
    else:
        strokes = sorted(strokes, cmp = strokeCmp)
    xScale = canvasSize[0]/canvas.getSize()[0]
    yScale = canvasSize[1]/canvas.getSize()[1]
    for stroke in strokes:
        if (len(stroke.move_list) <= min_stroke_length):
            pass #pygame.draw.circle(ccanvas, stroke.color, stroke.move_list[0], r)
        else:
            if(writeGcode):
                f.write('G1 Z.200\n')
                f.write('G1 X'+ str(stroke.move_list[0][0]*xScale)[:6]+
                        ' Y'+ str(stroke.move_list[0][1]*yScale)[:6]+'\r')
                f.write('G1 Z.00\n')
                for p in xrange(1, len(stroke.move_list)):
                    f.write('G1 X'+ str(stroke.move_list[p][0]*xScale)[:6]+
                            ' Y'+ str(stroke.move_list[p][1]*yScale)[:6]+'\r')
            if(blackOnly):
                pygame.draw.lines(ccanvas, (0,0,0), False, stroke.move_list, r)
            elif(limitedColors):
                
                stroke.color = closestColor(stroke.color)
                pygame.draw.lines(ccanvas, stroke.color, False, stroke.move_list, r)
            else:
                pygame.draw.lines(ccanvas, stroke.color, False, stroke.move_list, r)

            strokesUsed += 1
    if(writeGcode):
        f.close()
    print "Completed ", r, " layer."
    return canvas


def makeStroke(x_0, y_0, r, canvas, referenceImage, gradient):
    
    
    
    stroke_color = referenceImage.getPixel(x_0, y_0)
    stroke = Stroke(stroke_color, r)
    if(x_0 < 0 or y_0 < 0):
        return stroke
    if(x_0 > gradient.getWidth() or y_0 > gradient.getHeight()):
        return stroke
    stroke.addPoint((x_0, y_0))
    (x,y) = (x_0, y_0)
    (lastDx,lastDy) = (0,0)
    
    for i in xrange(0, max_stroke_length): #(i > min_stroke_length) and
        if ((True) and
        
        f_l*.25*difference(referenceImage.getPixel(x,y), canvas.getPixel(x,y)) <
        difference(referenceImage.getPixel(x,y), stroke_color)):
            return stroke
        if ((i > min_stroke_length) and gradient.getMag(x,y) < 3):
            return stroke
        (gx,gy) = gradient.getUnitVector(x,y)
        (dx,dy) = (-gy, gx)
        
        if (lastDx * dx + lastDy * dy < 0):
            (dx,dy) = (-dx, -dy)
        
        (dx, dy) = ((f_c * dx) + (1-f_c)*lastDx, (f_c * dy)+ (1-f_c)*lastDx)
        mag = max(1,(dx**2 + dy**2)**.5)
        (dx, dy) = (dx/mag, dy/mag)
        
        (x,y) = (int(x+r*dx), int(y+r*dy))
        (lastDx,lastDy) = (dx,dy)
        
        if((x<=0) or (y<=0) or
           (x >= gradient.getWidth()) or (y >= gradient.getHeight())):
            return stroke
        stroke.addPoint((x,y))
    return stroke

def _to_RGB_tuple(i):
    return ((i>>16)/255.0, ((i>> 8)& 255)/255.0, (i & 255)/255.0)

def to_RGB_tuple(i):
    return ((i>>16), ((i>> 8)& 255), (i & 255))

def generate_hsv():
    global hsv
    for i in colors:
        t = to_RGB_tuple(i)
        hsv +=[t]#[colorsys.rgb_to_hsv(t[0],t[1],t[2])]

def main(argv):
    generate_hsv()
    print hsv
    global ccanvas
    pygame.init()
    path = argv[1]
    #path2 = argv[2]
    picture = MyImage(Image.open(path))
    #picture2 = MyImage(Image.open(path2))
    ccanvas = pygame.display.set_mode(picture.getSize())
    ccanvas.fill((255,255,255))
    #ccanvas.fill((0,0,0))
    paint(picture,brushes)
    #paint(picture2, brushes)
    picture 
    
    pygame.image.save(ccanvas, ("output.png"))
    #print "Completed: image uses ", strokesUsed, " brushstrokes."
    pygame.quit()
    sys.exit()

    #pygame.display.update()
    #print "Completed: image uses ", strokesUsed, " brushstrokes."
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                
                pygame.quit()
                sys.exit()
    
#main(['out.png', brushes])
if __name__ == "__main__":
    main(sys.argv)