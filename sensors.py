import numpy as np
import matplotlib.pyplot as plt

length = 21
d = 1  # delta (increment of t)
sl = 100  # sensor length
pi = np.array([0, 0, 0])

# walls
wall = np.array([
    [120, 120, 20, 20, 120],
    [20, 120, 120, 20, 20]
])

fig = plt.figure()

plt.plot(wall[0], wall[1])



# limit of our grig
plt.plot(150, 150)

# title of the figure
plt.title('Our Robot')


# to make a circle with center x1,y1 and radius 0.75
circle = plt.Circle((pi[0], pi[1]), radius=length/2, facecolor='salmon', edgecolor='black')
plt.gca().add_patch(circle)

sensx = np.array([1, np.cos(np.pi/6), np.cos(np.pi/3), 0, -np.cos(np.pi/3), -np.cos(np.pi/6), -1, -np.cos(np.pi/6),
                 -np.cos(np.pi/3), 0, np.cos(np.pi/3), np.cos(np.pi/6), 1])
sensy = np.array([0, np.cos(np.pi/3), np.cos(np.pi/6), 1, np.cos(np.pi/6), np.cos(np.pi/3), 0, -np.cos(np.pi/3),
                 -np.cos(np.pi/6), -1, -np.cos(np.pi/6), -np.cos(np.pi/3), 0])

sensor = np.ones(12)*100

for x in range(0, 12):
    for i in range(0, wall.shape[1]-1):
        a1 = np.array([pi[0], pi[1]])
        a2 = np.array([sensx[x], sensy[x]])*sl + pi[:2]

        b1 = np.array([wall[0, i], wall[1, i]])
        b2 = np.array([wall[0, i+1], wall[1, i+1]])

        da = a2 - a1
        db = b2 - b1
        dp = a1 - b1

        dap = np.array([-da[1], da[0]])
        denom = np.dot(dap, db)
        num = np.dot(dap, dp)

        if ((abs(da[0]) == abs(db[0])) and (abs(da[1]) == abs(db[1]))) == 0:  #not paralels
            x3 = ((num / denom.astype(float)) * db + b1)[0]
            y3 = ((num / denom.astype(float)) * db + b1)[1]   # intersection of the lines

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

        plt.plot([pi[0], pi[0]+sensx[x]*sl], [pi[1], pi[1]+sensy[x]*sl], 'b')

print(sensor)

# to display the grid according to scale
plt.grid(True)
plt.axis('scaled')


plt.show()
