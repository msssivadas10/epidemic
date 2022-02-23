#!/usr/bin/python3

from _epidemic.core import Disease, Mask, Community, getGlobalClock, INFECTED

d = Disease('x', 0.2, 10)
m = Mask(0.5)
c = Community(50, 5., 100., )
c.startInfection(d)


def f(t):
    c.evolve()

clock = getGlobalClock()
clock.addTask(f)

import matplotlib.pyplot as plt
plt.style.use('ggplot')

fig, ax = plt.subplots(1,1,figsize=[6,6])

for i in range(100):
    ax.cla()
    for person in c.people:
        x, y = person.p
        col = 'red' if person.state is INFECTED else 'black'
        ax.plot(x, y, 'o', ms = 4, color = col)
        ax.add_patch(plt.Circle([x,y], person.ri, color = 'black', fill = 0,))

        if i == 50:
            person.wearMask(m)
    ax.set_aspect('equal')
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    plt.pause(0.01)
    clock.tick()

plt.show()