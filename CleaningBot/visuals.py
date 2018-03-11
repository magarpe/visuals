import turtle, random

colors  = ["red","green","blue","orange","purple","pink","yellow"]
screenSize = turtle.screensize()
screen = turtle.Screen()

robotTurtle = None
tsize = 21
shapelinesize = 2
turtleSpeed = 8

def showPath(robot, robotSize):
    global robotTurtle, colors, turtleSpeed, tsize, shapelinesize
    xy = robot['xy']
    x,y,h = xy[0]

    #call turtle window to front
    rootwindow = screen.getcanvas().winfo_toplevel()
    rootwindow.call('wm', 'attributes', '.', '-topmost', '1')
    rootwindow.call('wm', 'attributes', '.', '-topmost', '0')
    
    if robotTurtle is None:
        robotTurtle = turtle.Turtle(visible = False)
        
    robotTurtle.shapesize(outline=shapelinesize)
    robotTurtle.shape(createTurtleShape(random.choice(colors)))

    robotTurtle.speed(turtleSpeed)
    robotTurtle.turtlesize(robotSize/tsize)
    robotTurtle.pensize(robotSize)
    robotTurtle.penup()
    robotTurtle.goto((x,y))
    robotTurtle.setheading(h)
    robotTurtle.pencolor("grey")
    robotTurtle.pendown()
    robotTurtle.showturtle()
    for p in range(1,len(xy)):
        x,y,h = xy[p]
        robotTurtle.setheading(h)
        robotTurtle.goto(x,y)
    done()
    turtle.done()

def setup(wall):
    builder = turtle.Turtle(visible=False)  # drawing the walls with an invisible turtle
    builder.color("black", "red")
    builder.penup()
    builder.speed(0)
    builder.begin_fill()
    for z in range(len(wall)):
        for w in range(len(wall[z][0])):
            if w == 1:
                builder.pendown()
            builder.goto(wall[z][0][w], wall[z][1][w])
            if w == len(wall[z][0])-1:
                builder.penup()
    builder.end_fill()


def createTurtleShape(color):
        global screen
        cir = ((10, 0), (9.51, 3.09), (8.09, 5.88),(5.88, 8.09),
               (3.09, 9.51), (0, 10), (-3.09, 9.51), (-5.88, 8.09),
               (-8.09, 5.88),(-9.51, 3.09), (-10, 0), (-9.51, -3.09),
               (-8.09, -5.88), (-5.88, -8.09), (-3.09, -9.51), (-0.0, -10.0),
               (3.09, -9.51), (5.88, -8.09), (8.09, -5.88), (9.51, -3.09))
        line = ((0,0),(0,10))
        s = turtle.Shape("compound")
        s.addcomponent(cir, color, "black")
        s.addcomponent(line, "black", "black")
        screen.register_shape(str(color), s)
        return str(color)
def done():
    writer = turtle.Turtle(visible = False)
    writer.penup()
    writer.goto(-100,100)
    writer.write("Done", font=("Arial", 16, "normal"))
