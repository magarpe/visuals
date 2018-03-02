# import files
import matplotlib.pyplot as plt
import matplotlib.patches as patches

x1=0.0
y1=0.0
x2=0.0
y2=0.75
width = 8
height = 7
#The Figure is the overall window or page that everything is drawn on
fig=plt.figure()
#Adding a subplot to display Rectangle
ax1 = fig.add_subplot(111, aspect='equal')
ax1.add_patch(patches.Rectangle((2, 2), width, height))
#limit of our grig
plt.plot(12, 12)
#title of the figure
plt.title('Our Robot')
#to make a circle with center x1,y1 and radius 0.75
circle = plt.Circle((x1,y1), radius=0.75, facecolor='salmon', edgecolor='black')
plt.gca().add_patch(circle)
# to make the line inside circle (to express forward direction)
for j in range(0,75):
    plt.plot(x1,j/100,'b|')
#to display the grid according to scale
plt.grid(True)
plt.axis('scaled')
plt.show()
