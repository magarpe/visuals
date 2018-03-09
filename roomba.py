import turtle, random, math
import numpy as np

colors = ["red", "green", "blue", "orange", "purple", "pink", "yellow"]
screenSize = turtle.screensize()
screen = turtle.Screen()

numSensors = 12
robotSize = 25
tsize = 21  # turtle size
linesize = 2
turtleSpeed = 10
autoHeading = False
visualiseMode = True
showDust = False
numberofpopulation=10

# I need to change those two parameter according to calculation
numberofbumping=2
newarea=5

population= [[0 for x in range(numSensors)] for y in range(numberofpopulation)]
reproduction=[[0 for x in range(numSensors)]for y in range(numberofpopulation)]
index=[0 for x in range(3)]
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
    weights1 = {}
    weights2 = {}
    output1 = 0
    output2 = 0
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
            self.weights1[w] = random.uniform(weightMin, weightMax)  # chooses random weights
            self.weights2[w] = random.uniform(weightMin, weightMax)
        self.weights1[12] = random.uniform(weightMin, weightMax) * 10  # recursive with the motors
        self.weights2[12] = random.uniform(weightMin, weightMax) * 10
        for w in range(13, 15):
            self.weights1[w] = random.uniform(weightMin, weightMax) * 10  # for an average of points (x,y) covered
            self.weights2[w] = random.uniform(weightMin, weightMax) * 10
        for w in range(15, 25):
            self.weights1[w] = random.uniform(weightMin, weightMax) * 10  # last 5 points (xy xy xy xy xy)
            self.weights2[w] = random.uniform(weightMin, weightMax) * 10

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
        self.weights1 = wr
        self.weights2 = wl
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
        self.goto(target[0], target[1])
        self.setheading(target[2])

    def move(self, inputs):  # NN (inputs are sensors, outputs are the engines)

        genetic_algorithm()
        w1=random.randint(0, 9)
        w2= random.randint(0, 9)
        for i in range(0,numSensors):
            self.weights1[i]=population[w1][i]
            self.weights2[i]=population[w2][i]

        for i in range(len(inputs)):  # calculate the output based on inputs and weights of the individual
             self.output1 += inputs[i] * self.weights1[i]  # [0] till [11]
             self.output2 += inputs[i] * self.weights2[i]

        self.output1 *= self.weights1[12]  # [12] repulsiveness
        self.output2 *= self.weights2[12]

        if self.xyi > 1:  # calculates the average of the points visited
            xav = 0
            yav = 0
            for j in range(1, self.xyi):
                xav += self.xy[j][0][0]
                yav += self.xy[j][1][0]
            xav /= self.xyi
            yav /= self.xyi  # [13] and [14] based on average points visited
            self.output1 += xav * self.weights1[len(inputs) + 1] + yav * self.weights1[len(inputs) + 2]
            self.output2 += xav * self.weights2[len(inputs) + 1] + yav * self.weights2[len(inputs) + 2]

        if self.xyi > 8:  # [15] to [24] of the points visited
            for j in range(0, 5):
                self.output1 += self.xy[self.xyi - j][0][0] * self.weights1[len(inputs) + 3 + j * 2] + \
                                self.xy[self.xyi - j][1][0] * self.weights1[len(inputs) + 4 + j * 2]
                self.output2 += self.xy[self.xyi - j][0][0] * self.weights1[len(inputs) + 3 + j * 2] + \
                                self.xy[self.xyi - j][1][0] * self.weights1[len(inputs) + 4 + j * 2]

        self.xy[self.xyi + 1] = self.kinematics()  # calculates new point (kinematics), adds to positions
        self.xyi += 1
        xytry = (self.xy[self.xyi][0][0], self.xy[self.xyi][1][0], self.xy[self.xyi][2][0])
        self.moveToTarget(xytry)  # visualisation
        # move based on output 1 and 2

    def evaluate(self):
        # add fitness function
        self.score = random.random()
        return self.score

    def sensors(self, wall):
        if self.xyi == 0:  # sets the initial position
            pi = self.xy[self.xyi]
        else:
            pi = np.array([self.xy[self.xyi][0][0], self.xy[self.xyi][1][0], self.xy[self.xyi][2][0]])

        sensx = np.array(  # 12 sensors, circle radius1
            [1, np.cos(np.pi / 6), np.cos(np.pi / 3), 0, -np.cos(np.pi / 3), -np.cos(np.pi / 6), -1, -np.cos(np.pi / 6),
             -np.cos(np.pi / 3), 0, np.cos(np.pi / 3), np.cos(np.pi / 6), 1])
        sensy = np.array(
            [0, np.cos(np.pi / 3), np.cos(np.pi / 6), 1, np.cos(np.pi / 6), np.cos(np.pi / 3), 0, -np.cos(np.pi / 3),
             -np.cos(np.pi / 6), -1, -np.cos(np.pi / 6), -np.cos(np.pi / 3), 0])

        sensor = np.ones(12) * sl  # sets the length and the position of the sensors

        for x in range(0, numSensors):  # for each sensor
            for j in range(0, 2):       # 2 walls
                for i in range(0, 4):  # for each wall
                    a1 = np.array([pi[0], pi[1]])  # look for intersection, check the intersection is in the lines
                    a2 = np.array([sensx[x], sensy[x]]) * sl + pi[:2]

                    b1 = np.array([wall[j, 0, i], wall[j, 1, i]])
                    b2 = np.array([wall[j, 0, i + 1], wall[j, 1, i + 1]])

                    da = a2 - a1
                    db = b2 - b1
                    dp = a1 - b1

                    dap = np.array([-da[1], da[0]])
                    denom = np.dot(dap, db)
                    num = np.dot(dap, dp)

                    if ((abs(db[0]) == abs(da[0])) and (abs(da[1]) == abs(db[1]))) == 0:  # not parallels
                        x3 = ((num / denom.astype(float)) * db + b1)[0]
                        y3 = ((num / denom.astype(float)) * db + b1)[1]  # x3, y3 :intersection of the lines

                        if ((a1[0] >= x3 >= a2[0]) | (a2[0] >= x3 >= a1[0])) and \
                                ((a1[1] >= y3 >= a2[1]) | (a2[1] >= y3 >= a1[1])) and \
                                ((b1[0] >= x3 >= b2[0]) | (b2[0] >= x3 >= b1[0])) and \
                                ((b1[1] >= y3 >= b2[1]) | (b2[1] >= y3 >= b1[1])):  # they are in the lines
                            inter = np.array([x3, y3])
                            distance = np.sqrt((inter[0] - pi[0]) ** 2 + (inter[1] - pi[1]) ** 2)
                            if distance < sensor[x]:
                                sensor[x] = distance  # update if the current value is the smaller

        return (sensor)  # return array of 12 numbers (the output of each sensor)

    def kinematics(self):
        ml = self.output1
        mr = self.output2
        if self.xyi == 0:  # sets the initial position
            pi = self.xy[self.xyi]
        else:
            pi = np.array([self.xy[self.xyi][0][0], self.xy[self.xyi][1][0], self.xy[self.xyi][2][0]])

        if mr == ml:  # no rotation equations
            vc = (mr + ml) / 2
            po = pi + np.array([vc * d, vc * d, 0])

        else:  # rotation equations
            delta = 10
            r = 0.5 * (ml + mr) / (mr - ml)  # radius of rotation
            w = (mr - ml) / robotSize  # angular velocity

            icc = np.array([pi[0] - r * np.sin(pi[2]), pi[1] + r * np.cos(pi[2])])  # instantaneous center of curvature

            rotmax = np.array([  # rotation matrix
                [np.cos(w * delta), -np.sin(w * delta), 0],
                [np.sin(w * delta), np.cos(w * delta), 0],
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
                [w * delta]
            ])

        return po


def genetic_algorithm():

    initial_population()

    for generation in range(0,2):
        rank_population=[[0 for x in range(numSensors)] for y in range(numberofpopulation)]
        rank_population_fit=[0 for x in range(numberofpopulation)]
        for one in range(0,numberofpopulation):
            fit=fitness(population[one])
            #rank_population_fit contains the fitness and rank_population have same data
            rank_population_fit[one]=fit
            for j in range(numSensors):
                rank_population[one][j]=population[one][j]
        for i in range(10):
            selection(rank_population_fit,rank_population)
            parents1 = random.randint(0, 9)
            parents2 = random.randint(0, 9)
            reproduction[parents1]=mutation(reproduction[parents1])
            reproduction[parents2]=mutation(reproduction[parents2])

        for i in range(numberofpopulation):
            for j in range(numSensors):
                population[i][j] = reproduction[i][j]

    #here i need to upload my population table data to NN
    #i can choose or random select


def initial_population():
    for i in range(numberofpopulation):
        for j in range(numSensors):
            population[i][j] =random.uniform(weightMin, weightMax)


def my_max(array):
    a=max(array)
    for i in range(0,len(array)):
        if(a==array[i]):
            a1=i
            break
    b=max(array)
    for i in range(0,len(array)):
        if(b==array[i]):
            b1=i
            break
    c=max(array)
    for i in range(0,len(array)):
        if(c==array[i]):
            c1=i
            break

    return a1,b1,c1


# function for selection and reprodution

def selection(rank,rank_population):

    index[0],index[1],index[2]=my_max(rank)

    index[0], index[1], index[2] = my_max(rank)

    for j in range(0, numberofpopulation):
        if (j < 4):
            for i in range(0, numSensors):
                max1 = index[0]
                reproduction[j][i] = rank_population[max1][i]
        elif (j > 3 & j < 7):
            for i in range(0, numSensors):
                reproduction[j][i] = rank_population[index[1]][i]
        elif (j > 6):
            for i in range(0, numSensors):
                reproduction[j][i] = rank_population[index[2]][i]


    print('Before crossover Reproduction:')
    for i in range(10):
        for j in range(12):
            print(reproduction[i][j], end='     ')
        print()

    parents1=random.randint(0,9)
    parents2=random.randint(0,9)
    reproduction[parents1]=crossover(reproduction[parents1],reproduction[parents2])

    print('After crossover Reproduction:')
    for i in range(10):
        for j in range(12):
            print(reproduction[i][j], end='     ')
        print()


def crossover(array1,array2):
    if(random.randint(0,1)==0):
        choromosome1=array1[:6]
        choromosome2=array2[6:]
        totalchoromosome=choromosome1+choromosome2
    else:
        choromosome2 = array1[:6]
        choromosome1 = array2[6:]
        totalchoromosome = choromosome1 + choromosome2
    return totalchoromosome


def mutation(array):
    array_out=[0 for x in range(numSensors)]
    for i in range(0,numSensors):
        if random.randint(0,20)==1:
            array_out[i] = array_out[i]+random.randint(0,20)
        else:
            array_out[i] =array[i]
    return array_out


def fitness(array):
    fit=1
    fit=1+newarea*10-numberofbumping
    if fit<0: fit=1
    return fit


# initialisation

wall = np.array([
    np.array([  # nodes of the wall
        [220, 220, 120, 120, 220],
        [120, 220, 220, 120, 120]
    ]),
    np.array([  # nodes of the wall
        [0, 0, 300, 300, 0],
        [0, 300, 300, 0, 0]
    ])
    ])

wall += -110

for j in range (0, 2):
    boulder = turtle.Turtle(visible=False)  # drawing the walls with an invisible turtle
    boulder.color("black", "red")
    boulder.penup()
    boulder.speed(0)
    boulder.goto(wall[j, 0, 0], wall[j, 1, 0])
    boulder.pendown()
    # boulder.begin_fill()
    for i in range(1, 5):
        boulder.goto(wall[j, 0, i], wall[j, 1, i])
    # boulder.end_fill()

robot = Robot(0, 0, 0, "green")  # creates 1 robot position [x, y, teta (angle), color)

# movement of the robot

for mov in range(0, 10):
    robot.move(robot.sensors(wall))  # move(function) according to the sensors(function)

turtle.done()

