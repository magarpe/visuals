import turtle, random, math
import numpy as np
import matplotlib.pyplot as plt

colors = ["red", "green", "blue", "orange", "purple", "pink", "yellow"]
screenSize = turtle.screensize()
screen = turtle.Screen()

numSensors = 12
robotSize = 25
tsize = 21  # turtle size
linesize = 2
turtleSpeed = 1
autoHeading = False
visualiseMode = True
showDust = False

# xMax = (screenSize[0] - robotSize)
# xMin = -(screenSize[0] + robotSize)
# yMax = (screenSize[1] - robotSize)
# yMin = -(screenSize[1] + robotSize)
xMax = 200
xMin = -50
yMax = 200
yMin = -50


weightMin = 0
weightMax = 1

numRobots = 1
steps = 1
keepRobots = 5
# of which there are:
keepExtra = 2
mutationChance = 1

lines = []

sl = 100  # sensor length

if visualiseMode and showDust:
    dustTurtle = turtle.Turtle(visible=False)
    dustTurtle.pensize(1)
    dustTurtle.color("black")
    dustTurtle.speed(0)
    dustTurtle.penup()

dust = []
for d in range(500):
    dustparticle = random.uniform(xMin, xMax), random.uniform(yMin, yMax)
    if visualiseMode and showDust:
        dustTurtle.goto(dustparticle)
        dustTurtle.dot()
    dust.append(dustparticle)


# ############################################
#
#  R O B O T
#
# ############################################

class Robot(turtle.Turtle):
    shapes = []
    weightsr = {}
    weightsl = {}
    score = None
    dust = 0
    xy = {}
    xyi = 0

    def __repr__(self):
        return '{}: {}'.format(self.__class__.__name__,
                               self.score)

    def __lt__(self, other):
        return self.score < other.score

    def __init__(self, x, y, teta, col):
        global robotSize, tsize, linesize
        self.xy[0] = np.array([x, y, teta])
        for w in range(numSensors):
            self.weightsr[w] = random.uniform(weightMin, weightMax)     # chooses random weights
            self.weightsl[w] = random.uniform(weightMin, weightMax)
        turtle.Turtle.__init__(self, visible=False)
        if visualiseMode:
            self.shapesize(outline=linesize)
            self.shape(self.createTurtleShape(col))
            self.speed(turtleSpeed)
            self.color(col)
            self.turtlesize(robotSize / tsize)
            self.pensize(robotSize)
            self.penup()
            self.goto(x, y)
            self.pendown()
            self.pencolor("grey")
            self.setheading(teta)
            self.showturtle()
        else:
            self.penup()
            self.speed(0)
            self.goto(x, y)
            self.setheading(random.randint(0, 360))

    def createTurtleShape(self, color):
        global screen
        if not str(color) in self.shapes:
            cir = ((10, 0), (9.51, 3.09), (8.09, 5.88), (5.88, 8.09),
                   (3.09, 9.51), (0, 10), (-3.09, 9.51), (-5.88, 8.09),
                   (-8.09, 5.88), (-9.51, 3.09), (-10, 0), (-9.51, -3.09),
                   (-8.09, -5.88), (-5.88, -8.09), (-3.09, -9.51), (-0.0, -10.0),
                   (3.09, -9.51), (5.88, -8.09), (8.09, -5.88), (9.51, -3.09))
            line = ((0, 0), (0, 10))
            s = turtle.Shape("compound")
            s.addcomponent(cir, color, "black")
            s.addcomponent(line, "black", "black")
            screen.register_shape(str(color), s)
            self.shapes.append(str(color))
        return str(color)

    def overwriteRobot(self, x, y, col, wr, wl):
        self.weightsr = wr
        self.weightsl = wl
        self.hideturtle()
        self.penup()
        if visualiseMode:
            self.clear()
            self.shape(self.createTurtleShape(col))
            self.setheading(random.randint(0, 360))
            self.goto(x, y)
            self.pendown()
            self.showturtle()
        else:
            self.goto(x, y)

    def moveToTarget(self, target):
        global lines
        lines.append((self.pos(), target))
        # degrees = math.degrees(math.atan2(target[1] - self.pos()[1],
        #                                   target[0] - self.pos()[0]))
        # heading = self.heading()
        # if autoHeading:
        #     self.setheading(degrees)
        # else:
        #     if degrees < 0:
        #         degrees = 360 + degrees
        #     if ((heading - degrees + 360) % 360) > 180:
        #         if heading > 180 > degrees:
        #             self.left(360 - (heading - degrees))
        #         else:
        #             self.left(degrees - heading)
        #     else:
        #         if heading < 180 < (degrees):
        #             self.right(360 - (degrees - heading))
        #         else:
        #             self.right(heading - degrees)

        self.goto(target[0], target[1])
        self.setheading(target[2])
        # self.forward(self.distance(target))

    def move(self, inputs):
        output1 = 0                 # NN (inputs are sensors, outputs are the engines)
        output2 = 0
        for i in range(len(inputs)):                # calculate the output based on inputs and weights of the individual
            output1 += inputs[i] * self.weightsr[i]
            output2 += inputs[i] * self.weightsl[i]

        print (self.xyi)
        self.xyi += self.xyi
        print (self.xyi)
        self.xy[self.xyi+1] = self.kinematics(output1, output2)  # calculates new point (kinematics), adds to positions
        xytry = (self.xy[1][0][0], self.xy[1][1][0], self.xy[1][2][0])
        self.moveToTarget(xytry)               # visualisation
        # move based on output 1 and 2

    def evaluate(self):
        # add fitness function
        self.score = random.random()
        return self.score

    def sensors(self, wall):
        pi = self.xy[self.xyi]
        sensx = np.array(
            [1, np.cos(np.pi / 6), np.cos(np.pi / 3), 0, -np.cos(np.pi / 3), -np.cos(np.pi / 6), -1, -np.cos(np.pi / 6),
             -np.cos(np.pi / 3), 0, np.cos(np.pi / 3), np.cos(np.pi / 6), 1])
        sensy = np.array(
            [0, np.cos(np.pi / 3), np.cos(np.pi / 6), 1, np.cos(np.pi / 6), np.cos(np.pi / 3), 0, -np.cos(np.pi / 3),
             -np.cos(np.pi / 6), -1, -np.cos(np.pi / 6), -np.cos(np.pi / 3), 0])

        sensor = np.ones(12) * 100

        for x in range(0, numSensors):
            for i in range(0, wall.shape[1] - 1):
                a1 = np.array([pi[0], pi[1]])
                a2 = np.array([sensx[x], sensy[x]]) * sl + pi[:2]

                b1 = np.array([wall[0, i], wall[1, i]])
                b2 = np.array([wall[0, i + 1], wall[1, i + 1]])

                da = a2 - a1
                db = b2 - b1
                dp = a1 - b1

                dap = np.array([-da[1], da[0]])
                denom = np.dot(dap, db)
                num = np.dot(dap, dp)

                if ((abs(da[0]) == abs(db[0])) and (abs(da[1]) == abs(db[1]))) == 0:  # not paralels
                    x3 = ((num / denom.astype(float)) * db + b1)[0]
                    y3 = ((num / denom.astype(float)) * db + b1)[1]  # intersection of the lines

                    if ((a1[0] >= x3 >= a2[0]) | (a2[0] >= x3 >= a1[0])) and \
                            ((a1[1] >= y3 >= a2[1]) | (a2[1] >= y3 >= a1[1])) and \
                            ((b1[0] >= x3 >= b2[0]) | (b2[0] >= x3 >= b1[0])) and \
                            ((b1[1] >= y3 >= b2[1]) | (b2[1] >= y3 >= b1[1])):
                        inter = np.array([x3, y3])
                        distance = np.sqrt((inter[0] - pi[0]) ** 2 + (inter[1] - pi[1]) ** 2)
                        if distance < sensor[x]:
                            sensor[x] = distance
                            # sensx = inter[0]/sl - pi[0]
                            # sensy = inter[1]/sl - pi[1]

                # plt.plot([pi[0], pi[0] + sensx[x] * sl], [pi[1], pi[1] + sensy[x] * sl], 'b')

        return(sensor)

    def kinematics(self, ml, mr):
        pi = self.xy[self.xyi]
        if mr == ml:  # no rotation
            vc = (mr + ml) / 2
            po = pi + np.array([vc * d, vc * d, 0])

        else:  # rotation equations
            r = 0.5 * (ml + mr) / (mr - ml)  # radius of rotation
            w = (mr - ml) / robotSize  # angular velocity

            icc = np.array([pi[0] - r * np.sin(pi[2]), pi[1] + r * np.cos(pi[2])])  # instantaneous center of curvature

            rotmax = np.array([  # rotation matrix
                [np.cos(w * d), -np.sin(w * d), 0],
                [np.sin(w * d), np.cos(w * d), 0],
                [0, 0, 1]
            ])

            secmax = np.array([  # second matrix (multiplies rotation one)
                [pi[0] - icc[0]],
                [pi[1] - icc[1]],
                [pi[2]]
            ])

            po = np.dot(rotmax, secmax) + np.array([  # output position [x, y, teta]
                [icc[0]],
                [icc[1]],
                [w * d]
            ])

        po[2] = po[2] / (2 * np.pi) * 360  # to change from radians to degrees

        return po


def reproduceAndOverwrite(parent1, parent2, child):
    global colors
    if random.randint(0, 1) == 0:
        childW1 = parent1.weights1
        childW2 = parent2.weights2
    else:
        childW1 = parent1.weights2
        childW2 = parent2.weights1
    child.overwriteRobot(random.randint(xMin, xMax), random.randint(yMin, yMax), random.choice(colors), childW1,
                         childW2)
    return child


# start evolution


# initialisation

wall = np.array([               # nodes of the wall
    [120, 120, 20, 20, 120],
    [20, 120, 120, 20, 20]
])

boulder = turtle.Turtle(visible=False)          # drawing the walls with an invisible turtle
boulder.color("black", "red")
boulder.penup()
boulder.speed(0)
boulder.goto(wall[0, 0], wall[1, 0])
boulder.pendown()
boulder.begin_fill()
for i in range(1, wall.shape[1] - 1):
    boulder.goto(wall[0, i], wall[1, i])
boulder.end_fill()

robot = Robot(0, 0, 0, "green")                 # creates 1 robot position [x, y, teta (angle), color)

robot.move(robot.sensors(wall))                 # move(function) according to the sensors(function)

turtle.done()
