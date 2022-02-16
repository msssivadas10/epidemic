from math import sqrt
from random import uniform
import matplotlib.pyplot as plt

class Vector:
    """ A simple 2-vector. """

    def __init__(self, x: float, y: float) -> None:
        self.x, self.y = x, y

    def __repr__(self) -> str:
        return f"Vector({self.x}, {self.y})"

    def __add__(self, o: object) -> object:
        if isinstance(o, Vector):
            return Vector(self.x + o.x, self.y + o.y)
        elif isinstance(o, (float, int)):
            return Vector(self.x + o, self.y + o)
        return NotImplemented
    
    __radd__ = __add__

    def __sub__(self, o: object) -> object:
        if isinstance(o, Vector):
            return Vector(self.x - o.x, self.y - o.y)
        elif isinstance(o, (float, int)):
            return Vector(self.x - o, self.y - o)
        return NotImplemented

    def __rsub__(self, o: object) -> object:
        if isinstance(o, (float, int)):
            return Vector(o - self.x, o - self.y)
        return NotImplemented

    def __mul__(self, o: object) -> object:
        if isinstance(o, (float, int)):
            return Vector(self.x * o, self.y * o)
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, o: object) -> object:
        if isinstance(o, (float, int)):
            return Vector(self.x / o, self.y / o)
        return NotImplemented

    def __neg__(self, ) -> object:
        return Vector(-self.x, -self.y)

    def dist(self, o: object = ...) -> float:
        """ 
        If a Vector object is given, return the distance to a point specified 
        by that vector. If no argument is given, return the length. 
        """
        if o is ... :
            return sqrt(self.x**2 + self.y**2)
        elif isinstance(o, Vector):
            return sqrt((self.x - o.x)**2 + (self.y - o.y)**2)
        raise TypeError("object must be a vector")

class Clock:
    """ A clock object to keep time. """

    def __init__(self, start: float = 0., dt: float = 1.) -> None:
        self.time = start
        self.dt   = dt

    def __repr__(self) -> str:
        return f"Clock({self.time})"

    def update(self, ) -> None:
        """ Update time. """
        self.time += self.dt
        return 
    
    def reset(self, t: float = 0.) -> None:
        """ Reset the time to a sepecific time (default 0). """
        self.time = t
        return 

class Region:
    """ A square region in 2D plane. """

    def __init__(self, xwidth: float, ywidth: float = ..., x0: float = 0., y0: float = 0.) -> None:
        # use square region if no y-width is given
        ywidth = xwidth if ywidth is ... else ywidth

        self.x0, self.y0 = x0, y0         # origin point: lower-left corner
        self.xw, self.yw = xwidth, ywidth # region side lengths
        self.x1, self.y1 = x0 + xwidth, y0 + ywidth # upper-right corner

    def __repr__(self) -> str:
        return f"Region(x0 = {self.x0}, y0 = {self.y0}, x1 = {self.x1}, y1 = {self.y1})"

    def clip(self, pos: Vector, vel: Vector = ..., ) -> None:
        """ 
        Clip the position to within this region. If an optional velocity 
        is given, flip it. This will update the arguments. 
        """
        if not isinstance(pos, Vector):
            raise TypeError("pos must be a 'Vector'")

        def in_range(x: float, a: float, b: float) -> bool:
            return (a <= x <= b)
        
        def clip(x: float, a: float, b: float) -> float:
            return max(a, min(x, b))
        
        # clip position:
        flipx, flipy = False, False
        if not in_range(pos.x, self.x0, self.x1):
            pos.x = clip(pos.x, self.x0, self.x1)
            flipx = True
        if not in_range(pos.y, self.y0, self.y1):
            pos.y = clip(pos.y, self.y0, self.y1)
            flipy = True
        
        # flip velocity:
        if isinstance(vel, Vector):
            if flipx:
                vel.x = -vel.x
            if flipy:
                vel.y = -vel.y
        
        return

    def randomPosition(self, ) -> Vector:
        """ Get a random point in the region. """
        x = uniform(self.x0, self.x1)
        y = uniform(self.y0, self.y1)
        return Vector(x, y)

    def draw(self, ax: plt.Axes) -> None:
        """ Draw the region on the matplotlib axis. """
        ax.add_patch(
                        plt.Rectangle(
                                    (self.x0, self.y0),
                                    self.xw, self.yw,
                                    color = 'black',
                                    fill  = False,
                                    lw    = 1.5,
                                )
                    )
        return

class Disease:
    """ Class representing a disease. """

    def __init__(self, p: float, tr: float, r: float, name: str = '') -> None:
        if p < 0. or p > 1.:
            raise ValueError("probability must be in range [0, 1]")
        elif tr < 0.:
            raise ValueError("recovery time cannot be negative")
        elif r < 0.:
            raise ValueError("radius cannot be negative")
        self.ptrans = p
        self.trec   = tr
        self.radius = r
        self.name   = name

    def __repr__(self) -> str:
        name = f" '{self.name}'" if self.name else ''
        return f"<Disease{name}: porb.transmission = {self.ptrans}, recovery = {self.trec} d, radius = {self.radius}>"

class Status:
    """ Status of a person. """

    def __init__(self, status: str, colour: str = ...) -> None:
        self.setValue(status)
        self.setColour(colour)

    def __repr__(self) -> str:
        return str(self.value)

    def setValue(self, value: str) -> None:
        """ Set the status value. """
        self.value = value
        return

    def setColour(self, colour: str) -> None:
        """ Set the status indicator colour. """
        self.colour = colour
        return

Susceptible = Status('s', colour = "blue")
Infected    = Status('i', colour = "red")
Recovered   = Status('r', colour = "gray")

class Person:
    """ Class representing a random person. """

    def __init__(self, pos: Vector, vel: Vector, region: Region) -> None:
        if not isinstance(pos, Vector):
            raise TypeError("`pos` must be a 'Vector'")
        elif not isinstance(vel, Vector):
            raise TypeError("`vel` must be a 'Vector'")
        elif len(vel) != 2:
            raise TypeError("vel must have two coordinates")
        if not isinstance(region, Region):
            raise TypeError("region must be a 'Region'")
        
        self.pos        = pos         # position of this person, ...
        self.vel        = vel         # ... her/his velocity, ...
        self.status     = Susceptible # ... her/his infection status
        self.region     = region      # if given, tell the region
        self.toRecover  = 0.          # time (days) remaining for recovery 
        self.clock      = None        # to keep time (in days)
        self.myDisease  = None        # disease this person got (initialy none)

    def __repr__(self) -> str:
        return f"<Person pos = {(self.x, self.y)}, vel = {(self.vx, self.vy)}, status: '{self.status}'>"

    def setClock(self, clock: Clock) -> None:
        """ Give a clock to this person. """
        self.clock = clock
        return

    def infect(self, disease: Disease, infect_recovered: bool = False) -> None:
        """ Infect the person with a disease. """
        if self.isSusceptible or (self.isRecovered and infect_recovered):
            self.myDisease = disease
            self.status    = Infected
        return

    def recover(self, ) -> None:
        """ Recover from the disease. """
        if self.status is Infected:
            self.myDisease = None # no active disease
            self.status    = Recovered
        return

    def update(self) -> None:
        """ Update this person. """
        if not self.clock:
            raise RuntimeError("clock should be set to update")
        dt = self.clock.dt

        self.pos = self.pos + self.vel * dt  # update position
        self.region.clip(self.pos, self.vel) # limit the position to the region

        if self.status is Infected:
            self.toRecover = max(0., self.toRecover - dt)
            if self.toRecover == 0.:
                self.recover()
        return

    def draw(self, ax: plt.Axes) -> None:
        """ Draw this person on a matplotlib axis. """
        x, y = self.pos.x, self.pos.y

        ax.plot(x, y, 'o', color = self.status.colour)
        if self.status is Infected:
            # show the transmission area as a red circle around the person
            ax.add_patch(
                            plt.Circle(
                                        x, y, 
                                        self.myDisease.radius, 
                                        color = self.status.colour, 
                                        fill  = False, 
                                        lw    = 1., 
                                    )
                        )
        return

class Community:
    """ Class representing a community of random walkers. """

    def __init__(self, size: int, region: Region, clock: Clock = ..., vfact: float = 0.01, name: str = '') -> None:
        if not isinstance(region, Region):
            raise TypeError("region should be a 'Region'")
        self.region = region

        if not isinstance(size, int):
            raise TypeError("size (population) must be an 'int'")
        self.size = size

        if clock is ... :
            self.clock = Clock()
        else:
            if not isinstance(clock, Clock):
                raise TypeError("clock must be a 'Clock'")
            self.clock = clock
        
        self.vfact   = vfact
        self.name    = name
        self.disease = None # no active disease in the community

        self.makeRandomPopulation()

    def __repr__(self) -> str:
        name = f" '{self.name}'" if self.name else ''
        return f"<Population{name} size = {self.size}>"

    def makeRandomPopulation(self, ) -> None:
        """ Make a set of random people. """
        def randomVelocity() -> Vector:
            vx = uniform(-1., 1.) * self.region.xw * self.vfact
            vy = uniform(-1., 1.) * self.region.yw * self.vfact
            return Vector(vx, vy)
        
        pool = []
        for i in range(self.size):
            p = Person(self.region.randomPosition(), randomVelocity(), self.region)
            p.setClock(self.clock)
            pool.append(p)
        self.pool = pool
        return

    def draw(self, ax: plt.Axes) -> None:
        """ Draw the community on the matplotlib axis. """
        self.region.draw(ax)
        for person in self.pool:
            person.draw(ax)
        return

    



        

