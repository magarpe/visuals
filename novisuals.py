import numpy as np
import random
from numpy.linalg import norm
import pickle

niterations = 10
numRobots = 10
step = 500
robotSize = 21
numSensors = 12
sl = 100  # sensor length

bestFitness = np.zeros(niterations)

weightMin = 950
weightMax = 1000


# ############################################
#
#  R O B O T
#
# ############################################

class Robot:
    generation = 0
    weights1 = {}
    weights2 = {}
    output1 = 0
    output2 = 0
    fit = 0
    rank = 0
    xy = {}
    xyi = 0
    collision = 0
    coverage = 0

    def __init__(self, x, y, teta):
        global robotSize, tsize, linesize
        self.xy[0] = np.array([x, y, teta])
        for w in range(numSensors):
            self.weights1[w] = random.uniform(weightMin, weightMax)  # chooses random weights
            self.weights2[w] = random.uniform(weightMin, weightMax)
        self.weights1[12] = random.uniform(0, 1)  # recursive with the motors
        self.weights2[12] = random.uniform(0, 1)
        for w in range(13, 15):
            self.weights1[w] = random.uniform(weightMin, weightMax) * 10  # for an average of points (x,y) covered
            self.weights2[w] = random.uniform(weightMin, weightMax) * 10
        for w in range(15, 25):
            self.weights1[w] = random.uniform(weightMin, weightMax) * 10  # last 5 points (xy xy xy xy xy)
            self.weights2[w] = random.uniform(weightMin, weightMax) * 10

    def move(self, inputs):  # NN (inputs are sensors, outputs are the engines)
        # genetic_algorithm()
        # w1 = random.randint(0, 9)
        # w2 = random.randint(0, 9)
        # for i in range(0, numSensors):
        #     self.weights1[i] = population[w1][i]
        #     self.weights2[i] = population[w2][i]

        for i in range(len(inputs)):  # calculate the output based on inputs and weights of the individual
            self.output1 += inputs[i] * self.weights1[i]  # [0] till [11]
            self.output2 += inputs[i] * self.weights2[i]

        self.output1 *= self.weights1[12]  # [12] recursive
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

        self.xy[self.xyi + 1] = self.kinematics  # calculates new point (kinematics), adds to positions

        self.xyi += 1
        # xytry = (self.xy[self.xyi][0][0], self.xy[self.xyi][1][0], self.xy[self.xyi][2][0])
        # self.moveToTarget(xytry)  # visualisation
        # move based on output 1 and 2

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
            for j in range(0, 2):  # 2 walls
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

                    if ((abs(db[0]) == abs(da[0])) and (abs(da[1]) == abs(db[1]))) == 0 and (
                            denom != 0):  # not parallels
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
        return sensor  # return array of 12 numbers (the output of each sensor)

    @property
    def kinematics(self):
        delta = 10000000
        ml = self.output1
        mr = self.output2
        if self.xyi == 0:  # sets the initial position
            pi = self.xy[self.xyi]
        else:
            pi = np.array([self.xy[self.xyi][0][0], self.xy[self.xyi][1][0], self.xy[self.xyi][2][0]])

        pi[2] = pi[2] / 360 * 2 * np.pi

        if mr == ml:  # no rotation equations
            # vc = (mr + ml) / 2
            # po = pi + np.array([vc * delta, vc * delta, 0])
            mr = 1.1
            ml = 1

        # else:  # rotation equations
        r = 0.5 * (ml + mr) / (mr - ml)  # radius of rotation
        w = (mr - ml) / robotSize  # angular velocity

        icc = np.array([pi[0] - r * np.sin(pi[2]), pi[1] + r * np.cos(pi[2])])  # instantaneous center of curvature

        rotmax = np.array([  # rotation matrix
            [np.cos(w * delta), -np.sin(w * delta), 0],
            [np.sin(w * delta), np.cos(w * delta), 0],
            [0, 0, 1]])
        secmax = np.array([  # second matrix (multiplies rotation one)
            [pi[0] - icc[0]],
            [pi[1] - icc[1]],
            [pi[2]]])
        po = np.dot(rotmax, secmax) + np.array([  # output position [x, y, teta]
            [icc[0]],
            [icc[1]],
            [w * delta]])
        po[2] = po[2] * 360 / 2 / np.pi

        # check if point is ok:     # COLLISION#############################

        # po = np.array([[175], [175], [-90]])       #to test the collision
        d_min = 1000000
        inter = False
        p3 = np.array([po[0][0], po[1][0]])

        for j in range(0, 2):  # 2 walls
            for i in range(0, 4):  # for each wall
                p1 = np.array([wall[j, 0, i], wall[j, 1, i]])  # too close
                p2 = np.array([wall[j, 0, i + 1], wall[j, 1, i + 1]])

                dist = np.abs(np.cross(p2 - p1, p3 - p1) / norm(p2 - p1))  # distance to walls
                k = ((p2[1] - p1[1]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[1] - p1[1])) / \
                    ((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)
                pcol = np.array([  # point of collision
                    p3[0] - k * (p2[1] - p1[1]),
                    p3[1] + k * (p2[0] - p1[0])
                ])
                if (p1[0] <= pcol[0] <= p2[0] or p2[0] <= pcol[0] <= p1[0]) and \
                        (p1[1] <= pcol[1] <= p2[1] or p2[1] <= pcol[1] <= p1[1]) and \
                        (dist < d_min):  # collision in wall
                    d_min = dist  # distance from the position to the walls
                    col = pcol  # collision point

                # too far
                da = p2 - p1
                db = p3 - np.array([pi[0], pi[1]])
                dp = p1 - np.array([pi[0], pi[1]])

                dap = np.array([-da[1], da[0]])
                denom = np.dot(dap, db)
                num = np.dot(dap, dp)

                if ((abs(db[0]) == abs(da[0])) and (abs(da[1]) == abs(db[1]))) == 0 and (denom != 0):  # not parallels
                    x3 = ((num / denom.astype(float)) * db + np.array([pi[0], pi[1]]))[0]
                    y3 = ((num / denom.astype(float)) * db + np.array([pi[0], pi[1]]))[
                        1]  # x3, y3 :intersection of the lines

                    if ((p1[0] >= x3 >= p2[0]) | (p2[0] >= x3 >= p1[0])) and \
                            ((p1[1] >= y3 >= p2[1]) | (p2[1] >= y3 >= p1[1])) and \
                            ((np.array([pi[0], pi[1]])[0] >= x3 >= p3[0]) |
                             (p3[0] >= x3 >= np.array([pi[0], pi[1]])[0])) and \
                            ((np.array([pi[0], pi[1]])[1] >= y3 >= p3[1]) |
                             (p3[1] >= y3 >= np.array([pi[0], pi[1]])[1])):  # they are in the lines
                        col = np.array([x3, y3])
                        dist = np.abs(np.sqrt((col[0] - pi[0]) ** 2 + (col[1] - pi[1]) ** 2))
                        inter = True
                        if dist < d_min:
                            d_min = dist

            # print("in", col, d_min)

        if d_min < robotSize / 1.8 and not inter:  # if collision
            self.collision += 1
            pcol = col - p3
            pcol = pcol / np.abs(np.sqrt((col[0] - p3[0]) ** 2 + (col[1] - p3[1]) ** 2))
            po[0] = (col - robotSize / 1.8 * pcol)[0]
            po[1] = (col - robotSize / 1.8 * pcol)[1]

        if inter:
            self.collision += 5
            # print("collision far", col)
            po[0] = pi[0]
            po[1] = pi[1]
        return po

    def write(self):
        filename = str(self.generation) + "-" + str(self.rank)

        filehandler = open(filename, 'wb')
        pickle.dump(self, filehandler)

    def read(self):
        with open('robot', 'rb') as fp:     # not funccional
            file = pickle.load(fp)
            # print(file)

    def fitness(self):
        if self.fit == 0:
            self.fit = self.coverage*(1 - self.collision)         #add coverage
        return self.fit

    def givecoverage(self, coverage):
        self.coverage = coverage
        self.fitness()

    def setrank(self, rank):
        self.rank = rank

    def getrank(self):
        return self.rank

    def getweights(self):
        weights = np.zeros(50)
        for i in range(0, 25):
            weights[i] = self.weights1[i]
            weights[i+25] = self.weights2[i]
        return weights

    def setweights(self, weights, gen):
        self.generation += gen
        self.weights1 = weights[0:25]
        self.weights2 = weights[25:50]

    def getxy(self):
        return self.xy

# ############################################
#
#  G A    F U N C T I O N S
#
# ############################################

def selection(bots):
    first = True
    ordered = bots
    bestbot = 0
    bestsofar = 0
    for rank in range(0, numRobots):
        for nbots in range(0, numRobots):
            if bots[nbots].getrank() == 0:
                if first:
                    bestsofar = bots[nbots].fitness()
                    bestbot = nbots
                    first = False
                if bots[nbots].fitness() > bestsofar:
                    bestsofar = bots[nbots].fitness()
                    bestbot = nbots
        if range == 0:
            bestFitness[bots[bestbot].generation()] = bestsofar         # stores the best fitness of each iteration
            print("\nBest fitness of this generation stored\n")
        bots[bestbot].setrank(rank+1)
        first = True
        ordered[rank] = bots[bestbot]

    print("\nRanks of this generation computed\n")

    for i in range(0, 4):       # files with the 4 best of each generation
        ordered[i].write()


    # probability of selection based on rank
    selprob = [0.00074074, 0.00148148, 0.00296296, 0.00518519, 0.00814815, 0.01185185, 0.0162963, 0.02148148,
               0.02740741, 0.03407407, 0.04148148, 0.04962963, 0.05851852, 0.06814815, 0.07851852, 0.08962963,
               0.10148148, 0.11407407, 0.12740741, 0.14148148]

    # selected = np.zeros(20)

    selected = np.random.choice(ordered, 20, selprob)
    print("\nparents chosen based on their rank\n")
    return selected


def crossmut(array1, array2):
    cut = random.randrange(5, 20, 1)            # crossover
    choromosome = np.array([np.zeros(50), np.zeros(50)])
    rand = np.zeros(50)
    for j in range(0, 2):
        for i in range(0, 25):
            choromosome[j][i] = array1[i]
            choromosome[j][i + 25] = array2[i]

    for j in range(0, 2):           # mutation
        ran1 = np.zeros(25)
        ran2 = np.zeros(25)
        for w in range(numSensors):
            ran1[w] = random.uniform(weightMin, weightMax)  # chooses random weights
            ran2[w] = random.uniform(weightMin, weightMax)

        ran1[12] = random.uniform(0, 1)  # recursive with the motors
        ran2[12] = random.uniform(0, 1)
        for w in range(13, 15):
            ran1[w] = random.uniform(weightMin, weightMax) * 10  # for an average of points (x,y) covered
            ran2[w] = random.uniform(weightMin, weightMax) * 10
        for w in range(15, 25):
            ran1[w] = random.uniform(weightMin, weightMax) * 10  # last 5 points (xy xy xy xy xy)
            ran2[w] = random.uniform(weightMin, weightMax) * 10

        for i in range(0, 25):
            rand[i] = ran1[i]
            rand[i + 25] = ran2[i]

        for i in range(0, 50):
            choromosome[j][i] = np.random.choice([choromosome[j][i], rand[i]], 1, [0.9, 0.1])   # 10%change of mutation

        return choromosome


# ############################################
#
#  P E R C E N T   C O V E R E D
#
# ############################################

# def coverage(points, wall):
#     # coverage(robots[s].getxy()) in line 410
#     # inputs are the points covered 1 robot, and the walls
#
#     # make a flexible grid according to the (wall), don't just take numbers, take the values of the variable
#     # the from the grid make a matrix of 0s
#     # for each point check which squares of the grid it has traversed, change it to 1 in the matrix
#     # after all points have been checked, make a percentage of the number of 1 / the total space that can be covered
#
#     # percov sould be a number between 1 and 0
#     return perccov

# ############################################
#
#  M A I N
#
# ############################################

# initialisation

wall = np.array([
    np.array([  # nodes of the wall
        [50, 100, 100, 50, 50],
        [50, 50, 100, 100, 50]
    ]),
    np.array([  # nodes of the wall
        [0, 0, 150, 150, 0],
        [0, 150, 150, 0, 0]
    ])
])      # obstacles

robots = [Robot(25, 25, 0)
    for r in range(numRobots)]

print("\n-----------------------------------CLEANING ROBOT EVOLUTION------------------------------------\n",
      numRobots, "robots initialized\n\n")

for gen in range(0, niterations):
    print("\n\nevaluating generation: ", gen, " of ", niterations, " ...")
    for s in range(0, numRobots):
        print("\n moving robot: ", s, " of ", numRobots ," ...")
        for mov in range(0, step):
            robots[s].move(robots[s].sensors(wall))

        # covered = coverage(robots[s].getxy(), wall)
        # robots[s].givecoverage(covered)

    print("evaluation complete.\n\n starting selection...")
    selec = selection(robots)
    print("Selection completed")

    robots = [Robot(25, 25, 0)
              for r in range(numRobots)]
    print("\nNew offspring initialised, ready to get the dna from their parents\n")

    print("\nStarting reproduction...\n")
    for rep in range(0, numRobots, 2):      # reproduction
        dna = crossmut(selec[rep].getweights(), selec[rep].getweights())
        # print(
        #     # "\n\nfather1\n", selec[rep].getweights(), "\nfather2\n", selec[rep].getweights(),
        #       "\nsons\n", dna[1])

        robots[rep].setweights(dna[0], gen)
        robots[rep+1].setweights(dna[1], gen)
    print("\nreproduction complete.\nNew generation is ready\n\n")


print("\n\n\n------------------------------FINISHED-------------------------------------------\n\n\n "
      "Best fitnesses: ", bestFitness)