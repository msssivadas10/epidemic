from epidemic import EpidemicSimulation, Job
from epidemic import Infected, Recovered
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def draw(self: EpidemicSimulation, axs: list):
    """ function to draw the results. """
    # clear all axes
    for ax in axs:
        ax.cla()
    
    # draw the configuration:
    for p in self.community.people:
        axs[0].plot(
                    p.pos.x, 
                    p.pos.y, 
                    'o', 
                    ms = 5, 
                    color = 'red' if p.status is Infected else ('gray' if p.status is Recovered else 'black')
                )
        axs[0].add_patch(
                        plt.Circle(
                                        (p.pos.x, p.pos.y), 
                                        p.reff(), 
                                        color = 'black', 
                                        lw = 0.5, 
                                        fill = 0
                                    )
                    )

    axs[0].set_xlim(self.region.xrange())
    axs[0].set_ylim(self.region.yrange())
    axs[0].set_aspect('equal')
    axs[0].set_xticklabels([])
    axs[0].set_yticklabels([])

    stats   = self.getCurrentStats()
    s, i, r = stats['nexp'], stats['ninf'], stats['nrec']
    axs[0].set_title(f"t = {self.time():.3f} days, S = {s}, I = {i}, R = {r}")

    # draw stats:
    time = self.history.get('t')
    sir  = self.history.get('nexp'), self.history.get('ninf'), self.history.get('nrec')
    for stat, name, colour in zip(sir, ['exp', 'inf', 'rec'], ['green', 'red', 'gray']):
        axs[1].plot(time, stat, '-', color = colour, label = name)
    axs[1].set_xlabel('t (days)')
    axs[1].set_ylabel('SIR')
    axs[1].legend(
                    bbox_to_anchor = (0., 1.02, 1., .102), 
                    loc = 'lower left', 
                    ncol = 3, 
                    mode = "expand", 
                    borderaxespad = 0.
                )

    # jobs status:
    # vred = self.history.get('reduce_v')
    # axs[2].plot(time, vred, '-', label = "reduce_v")
    # axs[2].set_xlabel('t (days)')
    # axs[2].set_ylabel('value')
    # axs[2].set_ylim([-0.1, 1.1])
    # axs[2].legend(
    #                 bbox_to_anchor = (0., 1.02, 1., .102), 
    #                 loc = 'lower left', 
    #                 ncol = 3, 
    #                 mode = "expand", 
    #                 borderaxespad = 0.
    #             )

    plt.pause(0.005)
    return

def createfig():
    """ create the screen. """
    fig = plt.figure(figsize = [12, 6])

    ax  = plt.subplot(1, 2, 1) # to show configuration
    
    ax2 = plt.subplot(2, 2, 2) # to show stats

    ax3 = ... # plt.subplot(2, 2, 4, sharex = ax2) # to show signals

    plt.subplots_adjust(left = 0.05, right = 0.95, wspace = 0.1, hspace = 0.4)

    return fig, [ax, ax2, ax3]

# def f(self: Job):
#     return

def main():
    # initialise the community:
    reg = EpidemicSimulation.newRegion(100.)                   # a square region with size 100 unit
    com = EpidemicSimulation.newCommunity(100, reg, rinf = 3.) # initialise a community of 100 people

    # initialise disease:
    dis = EpidemicSimulation.newDisease(0.2, 10) # a disease with transmission probability 0.2 and 10 day recovery

    # job-1: reduce movements under some conditions
    # job1 = Job("job1", job = f, active = lambda t: True, track_history = True) 

    jobs = []

    # create the simulation onject:
    sim = EpidemicSimulation(tres = 10, jobs = jobs)
    sim.setup(community = com, disease = dis, infect = 1, draw = draw) # set-up the simulation

    # simulation main loop:
    fig, axs = createfig()

    # run the simulation upto t = 50
    # sim.run(until = lambda self: self.time() <= 50., args = [axs]) 

    plt.show()

if __name__ == "__main__":
    main()