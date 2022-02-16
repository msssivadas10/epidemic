import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

class Grid:

    def __init__(self, data, boxsize, subdiv = 10) -> None:
        # self.data = data
        self.boxsize = boxsize
        self.subdiv = subdiv
        self.cellsize = boxsize / subdiv

        cell_idx = (data // self.cellsize).astype(int)
        self.cells = []
        for i in range(self.subdiv):
            for j in range(self.subdiv):
                in_cell = data[(cell_idx[:,0] == i) & (cell_idx[:,1] == j)]
                cell = {'points': in_cell, }
                self.cells.append(cell)

    def intersect_ball(self, cell, center, radius):
        cell_x, cell_y = cell[0] * self.cellsize, cell[1] * self.cellsize
        circ_x, circ_y = center

        near_x = max(cell_x, min(circ_x, cell_x + self.cellsize))
        near_y = max(cell_y, min(circ_y, cell_y + self.cellsize))
        dx, dy = circ_x - near_x, circ_y - near_y
        return dx * dx + dy * dy < radius * radius
    
    def inside_ball(self, cell, center, radius):
        cell_x, cell_y = cell[0] * self.cellsize, cell[1] * self.cellsize
        circ_x, circ_y = center

        far_dx = max(abs(cell_x - circ_x), abs(cell_x + self.cellsize - circ_x))
        far_dy = max(abs(cell_y - circ_y), abs(cell_y + self.cellsize - circ_y))
        return far_dx * far_dx + far_dy * far_dy < radius * radius

    def query_ball(self, point, radius):
        point  = np.asarray(point)
        qrange = (-int(-radius // self.cellsize))
        qi, qj = (point // self.cellsize).astype(int)

        qpoints = None
        for i in range(max(0, qi - qrange), min(qi + qrange + 1, self.subdiv)):
            for j in range(max(0, qj - qrange), min(qj + qrange + 1, self.subdiv)):
                if not self.intersect_ball([i,j], point, radius):
                    continue
                _qpoints = self.cells[i * self.subdiv + j]['points']
                _dist2   = np.sum((_qpoints - point)**2, axis = -1)
                _qpoints = _qpoints[_dist2 <= radius * radius]
                if qpoints is None:
                    qpoints = _qpoints
                    continue
                qpoints = np.vstack([qpoints, _qpoints])
        return qpoints

    def query_shell(self, point, rbins):
        point  = np.asarray(point)

        r_in, r_out = min(rbins), max(rbins)
        qrange = (-int(-r_out // self.cellsize))
        qi, qj = (point // self.cellsize).astype(int)

        qpoints = [None for _ in range(len(rbins)-1)]
        for i in range(max(0, qi - qrange), min(qi + qrange + 1, self.subdiv)):
            for j in range(max(0, qj - qrange), min(qj + qrange + 1, self.subdiv)):
                if not self.intersect_ball([i,j], point, r_out):
                    continue
                if self.inside_ball([i,j], point, r_in):
                    continue
                points = self.cells[i * self.subdiv + j]['points']
                _dist2   = np.sum((points - point)**2, axis = -1)
                for k in range(len(rbins)-1):
                    _qpoints = points[(_dist2 <= rbins[k+1] * rbins[k+1]) & (_dist2 > rbins[k] * rbins[k])]
                    if qpoints[k] is None:
                        qpoints[k] = _qpoints
                    else:
                        qpoints[k] = np.vstack([qpoints[k], _qpoints])
        return qpoints

    def draw(self, ax):
        for i in range(1, self.subdiv):
            x, y = i * self.cellsize, i * self.cellsize
            ax.plot([x, x], [0., self.boxsize], lw = 0.5, color = 'black')
            ax.plot([0., self.boxsize], [y, y], lw = 0.5, color = 'black')
             
        ax.add_patch(plt.Rectangle((0., 0.), self.boxsize, self.boxsize, 
            color = 'black', fill = False, lw = 1.5))
        
        for cell in self.cells:
            ax.plot(cell['points'][:,0], cell['points'][:,1], 'o', color = 'tab:blue', ms = 1.5)


if __name__ == "__main__":
    data = np.random.uniform(0., 100., (10000, 2))

    qpoint = np.array([35., 25.]) #np.random.uniform(0., 100., (2, ))
    qrad   = 40. #np.random.uniform(1., 50., )
    qrad2  = 10.
    qcirc  = plt.Circle(qpoint, qrad, color = 'red', fill = False, lw = 1., )
    qcirc2 = plt.Circle(qpoint, qrad2, color = 'red', fill = False, lw = 1., )
    qrbins = [qrad2, 15., 20., 25., 30., 35., qrad]


    grid = Grid(data, 100., 10)
    qpts = grid.query_shell(qpoint, qrbins)
    # print(qpts)

    fig, ax = plt.subplots(figsize = (7,7))

    grid.draw(ax)

    ax.plot(qpoint[0], qpoint[1], 'o', color = 'red', ms = 2)
    # ax.add_patch(qcirc)
    # ax.add_patch(qcirc2)
    for r in qrbins:
        ax.add_patch(plt.Circle(qpoint, r, color = 'red', fill = False, lw = 1., ))


    cmap = ['green', 'yellow', ]
    
    for i, _qpts in enumerate(qpts):
        if _qpts is not None:
            ax.plot(_qpts[:,0], _qpts[:,1], 'o', color = cmap[i%2], ms = 1.5)

    ax.set_xlim([-5., 105,])
    ax.set_ylim([-5., 105,])
    
    # plt.axis('equal')
    plt.grid(0)
    plt.show()
