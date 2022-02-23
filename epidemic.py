from math import sqrt, sin, cos, tau
from random import gauss, uniform, choice
from itertools import product

def clip(x: float, to: list) -> float:
    """ Clip a number `x` in the range `to = [a, b]`. """
    return max(to[0], min(x, to[1]))

class Vector2D:
    """ A 2D vector. """

    def __init__(self, x: float, y: float) -> None:
        self.x, self.y = x, y

    @classmethod
    def fromAngle(cls, t: float) -> object:
        """ Unit vector from angle. """
        return cls(cos(t), sin(t))

    def __repr__(self) -> str:
        return f"Vector2D({self.x}, {self.y})"

    def __add__(self, o: object) -> object:
        if isinstance(o, Vector2D):
            return Vector2D(self.x + o.x, self.y + o.y)
        elif isinstance(o, (int, float)):
            return Vector2D(self.x + o, self.y + o)
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, o: object) -> object:
        if isinstance(o, Vector2D):
            return Vector2D(self.x - o.x, self.y - o.y)
        elif isinstance(o, (int, float)):
            return Vector2D(self.x - o, self.y - o)
        return NotImplemented

    def __rsub__(self, o: object) -> object:
        if isinstance(o, (int, float)):
            return Vector2D(o - self.x, o - self.y)
        return NotImplemented

    def __mul__(self, o: object) -> object:
        if isinstance(o, (int, float)):
            return Vector2D(self.x * o, self.y * o)
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, o: object) -> object:
        if isinstance(o, (int, float)):
            return Vector2D(self.x / o, self.y / o)
        return NotImplemented

    def __floordiv__(self, o: object) -> object:
        if isinstance(o, (int, float)):
            return Vector2D(self.x // o, self.y // o) 
        return NotImplemented

    def __mod__(self, o: object) -> object:
        if isinstance(o, (int, float)):
            return Vector2D(self.x % o, self.y % o)
        return NotImplemented

    def dot(self, o: object) -> float:
        """ Vector dot product. """
        if not isinstance(o, Vector2D):
            raise TypeError("o must be a 'Vector2D'")
        return self.x * o.x + self.y * o.y

    def dist2(self, o: object = ... ) -> float:
        """ Squared length/distance from another vector. """
        if o is ... :
            return self.x**2 + self.y**2
        if not isinstance(o, Vector2D):
            raise TypeError("o must be a 'Vector2D'")
        return (self.x - o.x)**2 + (self.y - o.y)**2
    
    def dist(self, o: object) -> float:
        """ Length/distance from another vector. """
        return sqrt(self.dist2(o)) 

    def clip(self, xrange: list, yrange: list) -> object:
        """ Clip the vector to given range. """
        return Vector2D(clip(self.x, xrange), clip(self.y, yrange))       

class Clock:
    """ A clock to keep time in days. """

    def __init__(self, start: float = 0., steps: int = 1) -> None:
        self.t, self.dt = start, 1. / steps
        self.steps      = steps

        self.updateWorld = None

    def __repr__(self) -> str:
        return f"Clock({self.t})"
    
    def time(self) -> float:
        """ Get the exact time. """
        return self.t

    def days(self) -> int:
        """ Get the number of days passed. """
        return int(self.t)

    def reset(self, t: float = 0) -> None:
        """ Reset the clock. """
        self.t = t
        return

    def run(self) -> None:
        """ Run the clock. """
        if self.updateWorld:
            self.updateWorld() 
        self.t += self.dt    
        return   

    def setUpdateFunction(self, f: object) -> None:
        """ Set a function to update the world. """
        if callable(f):
            self.updateWorld = f
            return
        raise TypeError("f should be a callable")

class Region:
    """ A rectangular region. """

    def __init__(self, xw: float, yw: float = ..., x0: float = 0., y0: float = 0., name: str = '') -> None:
        yw = xw if yw is ... else yw
        if xw <= 0. or yw <= 0.:
            raise ValueError("x and y widths must be positive")
        
        self.x0, self.y0 = x0, y0           # lower-left position
        self.xw, self.yw = xw, yw           # widths
        self.x1, self.y1 = x0 + xw, y0 + yw # upper-rigth position

        self.name = name # an optional name used to identify this region

    def __repr__(self) -> str:
        name = ''
        if self.name:
            name = f", name = '{self.name}'"
        return f"Region(x0 = {self.x0}, y0 = {self.y0}, xw = {self.xw}, yw = {self.yw}{name})"

    def randomPoint(self, ) -> Vector2D:
        """ Get a random point in this region. """
        return Vector2D(uniform(self.x0, self.x1), uniform(self.y0, self.y1))

    def xrange(self, ) -> tuple:
        """ Get the x range or the region. """
        return self.x0, self.x1

    def yrange(self, ) -> tuple:
        """ Get the y range or the region. """
        return self.y0, self.y1
    
    def size(self, ) -> tuple:
        """ Size of the region. """
        return self.xw, self.yw

    def inside(self, point: Vector2D) -> bool:
        """ Check if a point is inside or outside the region. """
        if not isinstance(point, Vector2D):
            raise TypeError("point must be a 'Vector2D'")
        return self.x0 <= point.x <= self.x1 and self.y0 <= point.y <= self.y1

class Grid:
    """ A grid structure to store some objects. """

    def __init__(self, objects: list, region: Region, posgetter: object = lambda __o: __o) -> None:
        self.objects = objects
        self.region  = region

        self._grid   = None  
        self._csize  = None
        self.xsubdiv = -1
        self.ysubdiv = -1
        
        if not callable(posgetter):
            raise TypeError("posgetter must be a callable")
        self.positionOf = posgetter # function to get position as Vector2D

    def configure(self, cellsize: float) -> None:
        """ Prepare the grid configuration. """
        if cellsize <= 0.:
            raise TypeError("cellsize must be positive")
        self._grid   = {}
        self._csize  = cellsize
        self.xsubdiv = -int(-self.region.xw // cellsize)
        self.ysubdiv = -int(-self.region.yw // cellsize)

        for i, o in enumerate(self.objects):
            pos  = self.positionOf(o)
            cell = (pos.x - self.region.x0) // cellsize, (pos.y - self.region.y0) // cellsize
            if cell not in self._grid.keys():
                self._grid[cell] = []
            self._grid[cell].append(i)
        return

    def intersect(self, cell: tuple, center: Vector2D, radius: float) -> bool:
        """ Check if a cell intersect with a circle. """
        delta = self._csize
        if delta is None:
            raise RuntimeError("grid is not configured")
        
        x, y  = cell[0] * delta, cell[1] * delta
        near  = center.clip(xrange = [x, x + delta], yrange = [y, y + delta])
        return (center.x - near.x)**2 + (center.y - near.y)**2 < radius**2

    def neighboursOf(self, o: object, radius: float = ... ) -> list:
        """ Neighbours of a given object within a radius. """
        delta = self._csize
        if delta is None:
            raise RuntimeError("grid is not configured")

        pos  = self.positionOf(o)
        i, j = int(pos.x // delta), int(pos.y // delta)

        step   = 1 if radius is ... else -int(-radius // delta)
        radius = delta if radius is ... else radius

        neighbours = []
        radius2    = radius**2

        # set of all cells to look:
        cells2look = product(
                                range(
                                        max(0, i-step), 
                                        min(i+step+1, self.xsubdiv)
                                     ), 
                                range(
                                        max(0, j-step), 
                                        min(j+step+1, self.ysubdiv)
                                     )
                            )

        for cell in cells2look:
            if cell not in self._grid.keys():
                continue
            if not self.intersect(cell, pos, radius):
                continue
            for k in self._grid[cell]:
                o2 = self.objects[k]
                if o is o2:
                    continue
                if pos.dist2(self.positionOf(o2)) > radius2:
                    continue
                neighbours.append(o2)
        return neighbours

class Disease:
    """ An infectious disease. """

    def __init__(self, p: float, trec: float, pspread: float = 0., name: str = '') -> None:
        self.p, self.trec = p, trec

        self.pspread = pspread
        self.name    = name

    def __repr__(self) -> str:
        return f"<Disease '{self.name}': p = {self.p}, trec = {self.trec}>"

    @property
    def ptrans(self) -> float:
        """ Probability of transmission. """
        if self.pspread:
            return gauss(self.p, self.pspread) 
        return self.p

class Status:
    """ State of something. """

    def __init__(self, value: int, displayValue: str = None, **attrs) -> None:
        if not isinstance(value, int):
            raise TypeError("value must be 'int'")
        self.value   = value
        self.display = displayValue
        self.attrs   = attrs

    def __repr__(self) -> str:
        return f"Status('{self.display if self.display else self.value}')"

    def getattr(self, name: str) -> object:
        """ Get an attribute value. """
        if name in self.attrs.keys():
            return self.attrs[name]
        raise AttributeError(f"no attribute called '{name}'")
            
Exposed   = Status(0, "exp", )
Infected  = Status(1, "inf", )
Recovered = Status(2, "rec", )

class Mask:
    """ A mask to hide from infection. """

    def __init__(self, rred: float, name: str = '') -> None:
        self.rred = rred # radius reduction factor
        self.name = name

    def __repr__(self) -> str:
        name = f" '{self.name}' " if self.name else ''
        return f"<Mask{name}: rred = {self.rred}>"

class Vaccine:
    """ A vaccine. """

    def __init__(self, pred: float, name: str = '') -> None:
        if pred < 0. or pred > 1.:
            raise ValueError("pred must be in range [0, 1]")
        self.pred = pred
        self.name = name

    def __repr__(self) -> str:
        name = f" '{self.name}':" if self.name else ""
        return f"<Vaccine{name} pred = {self.pred}>"

class Person:
    """ A person living in a community. """

    def __init__(self, pos: Vector2D, vel: Vector2D, clock: Clock = None, bc: object = None, rinf: float = 5.) -> None:
        self.status       = Exposed # infection status
        self.clock        = None    # clock to keep time (also update the object)
        self.bc           = None    # function to apply any boundary conditions
        self.rinf         = rinf    # randius of influence
        self.disease      = None    # disease this person has
        self.t2rec        = 0.0     # time left to recover (if infected)
        self.ptrans       = 0.0     # probability to transmit disease
        self._comm        = None    # community this person belongs to  
        self.vfact        = 1.0     # velocity reduction factor
        self.mask         = None    # mask this person wearing
        self.vaccine      = None    # vaccine took
        self.atQuarentine = False   # is in quarentine

        if not isinstance(pos, Vector2D):
            raise TypeError("pos mut be a 'Vector2D'")
        if not isinstance(vel, Vector2D):
            raise TypeError("vel mut be a 'Vector2D'")
        self.pos, self.vel = pos, vel

        if clock:
            self.setClock(clock)
        if bc:
            self.setBoundaryCondition(bc)
        
    def __repr__(self) -> str:
        return f"Person(pos = {self.pos}, vel = {self.vel}, status = {self.status})"

    def setClock(self, clock: Clock) -> None:
        """ Set a clock. """
        if isinstance(clock, Clock):
            self.clock = clock
            return
        raise TypeError("clock must be a 'Clock'")

    def setBoundaryCondition(self, bc: object) -> None:
        """ Set boundary conditions. """
        if callable(bc):
            self.bc = bc
            return 
        raise TypeError("bc must be a callable")
    
    def applyBoundaryCondition(self, ) -> None:
        """ Appply boundary conditions. """
        self.bc(self.pos, self.vel)
        return
        
    def walk(self) -> None:
        """ Move this person in the region. """
        if not self.clock:
            raise RuntimeError("no clock is attached to the person")
        vel      = self.vel * self.vfact # effective velocity
        self.pos = self.pos + vel * self.clock.dt
        if self.bc:
            self.applyBoundaryCondition()
        self.updateInfectionStatus()
        return
    
    def infect(self, disease: Disease) -> None:
        """ Infect this person with a disease. """
        if not self.status is Exposed:
            return
        if self.atQuarentine:
            return
        if isinstance(disease, Disease):
            self.disease, self.status = disease, Infected
            self.t2rec, self.ptrans   = disease.trec, disease.ptrans
            return
        raise TypeError("disease must be a 'Disease' object")

    def recover(self) -> None:
        """ Recover from a disease. """
        if self.status is Infected:
            self.disease, self.status = None, Recovered
        if self.atQuarentine:
            self.endQuarentine()
        return

    def updateInfectionStatus(self) -> None:
        """ Update infection status. """
        if self.status is Infected:
            self.t2rec = max(self.t2rec - self.clock.dt, 0.)
            if self.t2rec <= 0.:
                self.recover()
        return

    def infectOthers(self) -> None:
        """ Transmit the infection neighbours. """
        if not self.disease or not self._comm:
            return
        if self.atQuarentine:
            return
        
        for p in self._comm.grid.neighboursOf(self, self.reff()):
            if uniform(0., 1.) > self.peff():
                continue
            p.infect(self.disease)
        return

    def reff(self) -> float:
        """ Effective transmission radius. """
        radius = self.rinf
        if self.mask:
            radius = radius * self.mask.rred # effective radius after wearing mask
        return radius

    def peff(self) -> float:
        """ Effective transmission probability. """
        ptrans = self.ptrans
        if self.vaccine:
            ptrans = ptrans * self.vaccine.pred
        return ptrans

    def rescaleVelocity(self, by: float) -> None:
        """ Re-scale the velocity. """
        self.vfact = by
        return

    def goQuarentine(self) -> None:
        """ Go to quarentine. """
        self.atQuarentine = True

    def endQuarentine(self) -> None:
        """ End quarentine. """
        self.atQuarentine = False
    
    def putMask(self, mask: Mask, p: float = 1.) -> None:
        """ Put a mask. """
        if not isinstance(mask, Mask) or mask is not None:
            raise TypeError("mask must be a 'Mask' or None")
        if uniform(0., 1.) <= p:
            self.mask = mask
        return

    def removemask(self, ) -> None:
        """ Remove any mask. """
        self.mask = None

    def vaccinate(self, vaccine: object, p: float = 1.) -> None:
        """ Vaccinate this person. """
        if not isinstance(vaccine, Vaccine):
            raise TypeError("vaccine must be a 'Vaccine' object")
        if uniform(0., 1.) <= p:
            self.vaccine = vaccine
        return

class Community:
    """ A community of people. """

    def __init__(self, size: int, region: Region, clock: Clock = None, vmax: float = 10., rinf: float = 5., rspread: float = 0., bc: object = 'periodic'):
        if not isinstance(size, int):
            raise TypeError("community size must be 'int'")
        if not isinstance(region, Region):
            raise TypeError("region must be a 'Region'")

        self.size    = size   # population size
        self.region  = region # region the community is living
        self.clock   = clock  # clock to keep time
        self.vmax    = vmax   # maximum speed
        self.disease = None   # active disease
        self.people  = []
        self.grid    = None

        self.rinf, self.rspread = rinf, rspread # mean and spread in influence radius

        if self.clock:
            self.setClock(clock)
        self.setBoundaryCondition(bc)

        self.initPeople()

    def __repr__(self) -> str:
        return f"<Community size = {self.size}>"

    def setClock(self, clock: Clock) -> None:
        """ Set a clock. """
        if not isinstance(clock, Clock):
            raise TypeError("clock must be a 'Clock'")
        self.clock = clock
        for person in self.people:
            person.setClock(clock)
        return

    def setBoundaryCondition(self, bc: object) -> None:
        """ Set boundary conditions. """
        def periodic(reg: Region, pos: Vector2D, vel: Vector2D) -> None:
            """ periodic boundary conditions. """
            pos.x = (pos.x - reg.x0) % reg.xw + reg.x0
            pos.y = (pos.y - reg.y0) % reg.yw + reg.y0
            return

        allBC = {'periodic': periodic}

        if isinstance(bc, str):
            if bc not in allBC.keys():
                raise ValueError("invalid boundary condition")
            bc = allBC[bc]

        if not callable(bc):
            raise TypeError("bc must be a callable")

        def bcfunc(pos: Vector2D, vel: Vector2D):
            """ boundary condition function. """
            bc(self.region, pos, vel)

        self.bc = bcfunc
        return

    def add(self, person: Person) -> None:
        """ Add a person to this community. """
        if not isinstance(person, Person):
            raise TypeError("person must be a 'Person' object")
        if not self.region.inside(person.pos):
            person.pos = self.region.randomPoint()
        person._comm   = self
        self.people.append(person)
        return

    def initPeople(self) -> None:
        """ Create and add people to the community. """
        def randomVelocity() -> Vector2D:
            """ a random velocity vector. """
            return Vector2D.fromAngle(uniform(0., tau)) * uniform(0., self.vmax)

        self.people = []
        for i in range(self.size):
            person = Person(
                                pos   = self.region.randomPoint(),
                                vel   = randomVelocity(),
                                clock = self.clock,
                                bc    = self.bc,
                                rinf  = gauss(self.rinf, self.rspread) if self.rspread else self.rinf,
                            )
            self.add(person)
        self.grid = Grid(self.people, self.region, lambda o: o.pos)

    def startDisease(self, disease: Disease, i0: int = 1) -> None:
        """ Start a disease by infecting a random person. """
        if isinstance(disease, Disease):
            self.disease = disease
            if i0 > self.size:
                raise ValueError("number of infections exceed community size")
            while i0:
                p = choice(self.people)
                while p.status is Infected:
                    print(p.status)
                    p = choice(self.people)
                p.infect(disease)
                i0 -= 1
            return
        raise TypeError("disease must be a 'Disease' object")

    def spreadInfection(self) -> None:
        """ Spread the disease in the community. """
        if not self.disease:
            return
        self.grid.configure(cellsize = max(map(lambda p: p.rinf, self.people)))
        for person in self.people:
            person.infectOthers()

    def evolve(self) -> None:
        """ Evolve community. """
        if not self.clock:
            raise RuntimeError("no clock is attached to the community")
        self.spreadInfection()
        for person in self.people:
            person.walk()
        return

    def getCurrentStats(self) -> list:
        """ Get the stats. """
        sir = [0, 0, 0]
        for person in self.people:
            state       = person.status.value
            sir[state] += 1
        return sir

    def enforceMask(self, mask: Mask, p: float = 1.) -> None:
        """ Enforce people to wear mask. """
        for person in self.people:
            person.putMask(mask, p)
        return

    def removeMasks(self) -> None:
        """ Remove all masks. """
        for person in self.people:
            person.removemask()
        return

    def vaccinate(self, vaccine: Vaccine, p: float = 1.) -> None:
        """ Vaccinate people. """
        for person in self.people:
            person.vaccinate(vaccine, p)
        return

# ===========================================
# Simulation objects:
# ===========================================

class History:
    """ An object to track the stats. """

    def __init__(self, vars: list) -> None:
        self.varlist = {}

        for v in vars:
            if not isinstance(v, str):
                raise TypeError("variable names should be 'str'")
            self.varlist[v] = []
    
    def __repr__(self) -> str:
        return f"Stats({', '.join(self.varlist.keys())})"

    def isTracking(self, var: str) -> bool:
        """ Check if a variable is tracking. """
        return var in self.varlist.keys()

    def push(self, **kwargs) -> None:
        """ Push stats to the que. """
        for var, value in kwargs.items():
            if not self.isTracking(var):
                raise ValueError(f"variable `{var}` is not tracking")
            self.varlist[var].append(value)
        return
    
    def get(self, var: str) -> list:
        """ Get a variable. """
        if not self.isTracking(var):
            raise ValueError(f"variable `{var}` is not tracking")
        return self.varlist[var]

class Job:
    """ A specfic job to do at a specific time. """

    def __init__(self, name: str, job: object, active: object, default_val: float = 0., args: list = [], track_history: bool = False) -> None:
        if not callable(job):
            raise TypeError("job must be a callable")
        if not callable(active):
            raise TypeError("active must be a callable")

        self.name   = name
        self.active = active
        self.job    = job
        self.args   = args
        self.value  = default_val
        self.track  = track_history
        
        self.linkedto = None # object this job is linked to

    def status(self, t: float) -> int:
        """ Tell if the job is active at this time. """
        return 1 if self.active(t) else 0

    def do(self, t: float) -> None:
        """ Do the job. """
        if self.active(t):
            self.job(self, *self.args)
        return

    def linkto(self, o: object) -> None:
        """ Link this job to an object. """
        self.linkedto = o
        return

class EpidemicSimulation:
    """ An epidemic simulation on a single community. """

    def __init__(self, tres: int = 1, jobs: list = [], track_stats: bool = True) -> None:
        self.clock = Clock(steps = tres) # clock to track time
        self.clock.setUpdateFunction(self._update)

        self.draw = None # function to draw

        self.track_stats = track_stats
        vars2track       = ['t', 'nexp', 'ninf', 'nrec', ]

        self.jobs = []
        for job in jobs:
            if not isinstance(job, Job):
                raise TypeError("jobs must be a list of 'Job' objects")
            job.linkto(self)
            self.jobs.append(job)
            if job.track:
                vars2track.append(job.name)
        
        self.history = History(vars2track)

        # simulation parameters:
        self.useQuarentine = False
        self.useMask       = False
        self.useVaccine    = False
        self.reinfect      = False 

        # simulation setup:
        self.disease   = None
        self.community = None
        self.region    = None

        # default specifications for people objects:
        self.pplspec   = {'vfact': 10., 'rinf': 5., 'rspread': 0., }

    def linkDisease(self, disease: Disease) -> None:
        """ Link a disease to the simulation. """
        if isinstance(disease, Disease):
            self.disease = disease
            return
        raise TypeError(("disease must be a 'Disease' object"))

    def linkCommunity(self, community: Community) -> None:
        """ Link a community to the simulation. """
        if isinstance(community, Community):
            self.community = community
            self.region    = community.region
            self.community.setClock(self.clock)
            return
        raise TypeError(("community must be a 'Community' object"))

    def startDisease(self, infect: int = 1) -> None:
        """ Start the epidemic. """
        if self.ready():
            self.community.startDisease(self.disease, infect)
            return
        raise RuntimeError("simulation is not ready to start - either the community or disease not set")

    def drawFunction(self, f: object) -> None:
        """ Set draw function. """
        if callable(f) or f is None:
            self.draw = f
            return
        raise TypeError("f must be a callable.")

    def setup(self, community: Community, disease: Disease, infect: int = 1, draw: object = None) -> None:
        """ Setup the simulation. """
        self.linkCommunity(community)
        self.linkDisease(disease)
        self.startDisease(infect)
        self.drawFunction(draw)
        return 

    def ready(self) -> bool:
        """ Is ready to start the simulation. """
        return (self.disease and self.community)

    def time(self) -> float:
        """ Simulation time. """
        return self.clock.time()

    def _update(self) -> None:
        """ Function to update the simulation. """
        # all the jobs:
        t = self.time()
        for job in self.jobs:
            job.do(t)

        self.community.evolve()
        return

    def run(self, until: object, show: bool = True, args: list = []) -> None:
        """ Run the simulation until the condition. """
        if not callable(until):
            raise TypeError("until must be a callable")

        while until(self):
            self.pushCurrentStats()
            if show and self.draw:
                self.draw(self, *args)
            self.clock.run()
        return

    def getCurrentStats(self) -> dict:
        """ Get the stats. """
        t = self.time()

        nexp, ninf, nrec  = self.community.getCurrentStats()
        
        vars = {'t': t, 'nexp': nexp, 'ninf': ninf, 'nrec': nrec, }

        for job in self.jobs:
            if job.track:
                vars[job.name] = job.value

        return vars

    def pushCurrentStats(self) -> None:
        """ Push the stats. """ 
        if not self.track_stats:
            return
        
        vars = self.getCurrentStats()
        self.history.push(**vars)
        return

    def enforceMask(self, mask: Mask, p: float = 1.) -> None:
        """ Enforce people in thsi simulation to wear mask. """
        return self.community.enforceMask(mask, p)

    def removeMasks(self) -> None:
        """ Remove all masks. """
        return self.community.removeMasks()

    def vaccinate(self, vaccine: Vaccine, p: float = 1.) -> None:
        """ Vaccinate people in this simulation. """
        return self.community.vaccinate(vaccine, p)

    @staticmethod
    def newRegion(xw: float, yw: float = ..., x0: float = 0., y0: float = 0., name: str = '') -> Region:
        """ Create a new `Region` object. """
        return Region(
                        xw   = xw, 
                        yw   = yw, 
                        x0   = x0, 
                        y0   = y0, 
                        name = name
                     )

    @staticmethod
    def newCommunity(size: int, region: Region, vmax: float = 10., rinf: float = 5., rspread: float = 0., bc: object = 'periodic') -> Community:
        """ Create a new `Community` object. """
        return Community(
                            size    = size,
                            region  = region,
                            clock   = None,
                            vmax    = vmax,
                            rinf    = rinf,
                            rspread = rspread,
                            bc      = bc,
                        )

    @staticmethod
    def newDisease(p: float, trec: float, pspread: float = 0., name: str = '') -> Disease:
        """ Create a new `Disease` object. """
        return Disease(
                            p       = p,
                            trec    = trec,
                            pspread = pspread,
                            name    = name,
                       )




