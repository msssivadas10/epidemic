import math, random
from turtle import color
import matplotlib.pyplot as plt
plt.style.use('ggplot')

r_inf0 = 1.5
r_inf = r_inf0
p_inf = 0.5
tres = 10
dt = 1 / tres
colours = ['green', 'red', 'yellow']

class person:
    def __init__(self) -> None:
        self.status = 0 # infection status

        # random position in 100 side box
        self.x = random.uniform(0., 100.)
        self.y = random.uniform(0., 100.)

        # random velocity
        self.vx = random.uniform(-10., 10.)
        self.vy = random.uniform(-10., 10.)

        self.trec = 0

    def infect(self):
        self.status = 1
        self.trec = 5

    def recover(self):
        self.status = 2

    def update(self):
        self.x = (self.x + self.vx * dt) % 100.
        self.y = (self.y + self.vy * dt) % 100.

        if self.status == 1:
            self.trec -= dt
            if self.trec <= 0.:
                self.recover()

    def infect_nearest(self, others):
        if self.status != 1:
            return
        for p in others:
            if p is self or p.status != 0:
                continue
            if (self.x - p.x)**2 + (self.y - p.y)**2 < r_inf**2:
                if random.random() < p_inf:
                    p.infect()

    def draw(self, ax):
        c = colours[self.status]
        ax.plot(self.x, self.y, 'o', ms = 4, color = c, markeredgecolor = 'k')

        if self.status == 1:
            ax.add_patch(
                plt.Circle((self.x, self.y), r_inf, fill = False, color = c)
            )

        ax.set_xlim([0., 100])
        ax.set_ylim([0., 100])

n = 250
people = [person() for i in range(n)]

n_exp, n_inf, n_rec = [n, ], [0, ], [0, ]

random.choice(people).infect()

# fig, (ax, ax2) = plt.subplots(2, 1, figsize = [6, 8], gridspec_kw = {"height_ratios":[1, 0.25]})

# ax2.set_xlabel('t')
# ax2.set_ylabel('count')

# ax2.plot([], [], '-', color = colours[0], label = "exp.")
# ax2.plot([], [], '-', color = colours[1], label = "inf.")
# ax2.plot([], [], '-', color = colours[2], label = "rec.")
# ax2.legend()

t = [0, ]
change_v = False
for ti in range(100):
    t.append(ti)

    for _ in range(tres):
        # ax.cla()
        # ax.text(0., 0., 'Day {}'.format(t[-1]))

        for p in people:
            # p.draw(ax)
            p.update()

        # ax2.plot(t, n_exp, '-', color = colours[0], label = "exp.")
        # ax2.plot(t, n_inf, '-', color = colours[1], label = "inf.")
        # ax2.plot(t, n_rec, '-', color = colours[2], label = "rec.")


        for p in people:
            p.infect_nearest(people)

        plt.pause(0.1)

    _n_exp, _n_inf, _n_rec = 0, 0, 0
    for p in people:
        if p.status == 0:
            _n_exp += 1
        elif p.status == 1:
            _n_inf += 1
        else:
            _n_rec += 1
    n_exp.append(_n_exp)
    n_inf.append(_n_inf)
    n_rec.append(_n_rec)

    



plt.figure()
# plt.plot(t, n_exp, '-', color = colours[0])
plt.plot(t, n_inf, '-', color = colours[1])
# plt.plot(t, n_rec, '-', color = colours[2])
plt.show()
