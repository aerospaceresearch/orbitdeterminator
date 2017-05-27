from vpython import *
from vpython.graph import *
from color import *

scene = display(width=600, height=600, center=(0, 5, 0))

Sun = sphere(pos=vector(0,0,0), radius=100, color=color.orange)
earth = sphere(pos=vector(-200,0,0), radius=10,
               make_trail=True)

earthv = vector(0, 0, 6)

gd = gdiplay(x=800, y=0, width=600, height=600,
             ##foreground=color.blaco, background=color.white,
            ## xmax=3000, xmin=0, ymax=500, ymin=0)
f1 = gcurve(color=color.red)
t = 0

for i in range(1000):
    rate(100)
    earth.pos = earth.pos + earthv
    dist = (earth.x ** 2 + earth.y ** 2 + earth.z ** 2) ** 0.5
    RadialVector = (earth.pos - Sun.pos) / dist
    Fgrav = -10000 * RadialVector / dist ** 2
    earthv = earthv + Fgrav
    earth.pos += earthv
    f1.plot(pos=(t, mag(earhv)))
    t += 1

    if dist <= Sun.radius: break
