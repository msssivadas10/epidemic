#!/usr/bin/python3

from math import pi, sin, cos, sqrt, atan2, ceil
from random import uniform
from collections import namedtuple
from itertools import product
from subprocess import call
from typing import Any, Callable


# ========================================
# 2D vectors and points
# ======================================== 

class VectorError(Exception):
    """ Exceptions used by 2D vector objects. """
    ...

class Vector2D:
    """ 
    A 2D vector class. This can be used to specify a 2-vector or a 
    coordinates of a point in plane. The two components are given by the 
    attributes `x` and `y`.  This supports vector math operations.
    """
    __slots__ = 'x', 'y', '_i' 

    def __init__(self, x: float, y: float = ..., fmt: str = 'cart') -> None:            
        if fmt == 'cart':
            if y is ... :
                raise VectorError("y is a required argument")
            self.x, self.y = x, y
        elif fmt == 'polar':
            # y is angle and x is radius
            y = 1.0 if y is ... else y
            self.x, self.y = y * cos(x), y * sin(x)
        else:
            raise ValueError(f"invalid value for fmt, `{fmt}`")

    @property
    def r(self) -> float:
        """ Length of the vector or distance from origin. """
        return sqrt(self.x**2 + self.y**2)

    @property
    def theta(self) -> float:
        """ Angle or direction of the vector. """
        return atan2(self.y, self.x)

    def __repr__(self) -> str:
        return f"Vector2D({self.x}, {self.y})"

    def __add__(self, o: Any) -> Any:
        if isinstance(o, Vector2D):
            return Vector2D(self.x + o.x, self.y + o.y)
        elif isinstance(o, (int, float)):
            return Vector2D(self.x + o, self.y + o)
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, o: Any) -> Any:
        if isinstance(o, Vector2D):
            return Vector2D(self.x - o.x, self.y - o.y)
        elif isinstance(o, (int, float)):
            return Vector2D(self.x - o, self.y - o)
        return NotImplemented

    def __rsub__(self, o: Any) -> Any:
        if isinstance(o, (int, float)):
            return Vector2D(o - self.x, o - self.y)
        return NotImplemented

    def __mul__(self, o: Any) -> Any:
        if isinstance(o, (int, float)):
            return Vector2D(self.x * o, self.y * o)
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, o: Any) -> Any:
        if isinstance(o, (int, float)):
            return Vector2D(self.x / o, self.y / o)
        return NotImplemented

    def __floordiv__(self, o: Any) -> Any:
        if isinstance(o, (int, float)):
            return Vector2D(self.x // o, self.y // o) 
        return NotImplemented

    def __mod__(self, o: Any) -> Any:
        if isinstance(o, (int, float)):
            return Vector2D(self.x % o, self.y % o)
        return NotImplemented

    def __neg__(self) -> Any:
        return Vector2D(-self.x, -self.y)

    def __pos__(self) -> Any:
        return Vector2D(self.x, self.y)

    def __iter__(self) -> Any:
        self._i = 0
        return self

    def __next__(self) -> float:
        _i = self._i
        self._i += 1
        if _i > 1:
            raise StopIteration()
        return self.y if _i else self.x

    def dot(self, o: Any) -> float:
        """ Dot product with another vector. """
        if not isinstance(o, Vector2D):
            raise TypeError("o must be a 'Vector2D'")
        return self.x * o.x + self.y * o.y

    def dist2(self, o: Any) -> float:
        """ Squared distance to another point. """
        if not isinstance(o, Vector2D):
            raise TypeError("o must be a 'Vector2D'")
        return (self.x - o.x)**2 + (self.y - o.y)**2
    
    def dist(self, o: Any) -> float:
        """ Distance to a point. """
        return sqrt(self.dist2(o)) 

# ========================================
# clocks 
# ======================================== 

class ClockError(Exception):
    """ Exceptions used by clock objects. """
    ...

class Clock:
    """ 
    A clock class. This clock object can be used to track time and do tasks 
    evolving with time. This can be done by linking some task objects to the 
    clock, which is executed when the clock is running.
    """
    __slots__ = 't', 'dt', 'spu', 'tasks', 

    def __init__(self, steps_per_unit: int = 1, t0: float = 0.) -> None:
        self.t     = t0
        self.spu   = steps_per_unit
        self.dt    = 1.0 / steps_per_unit
        self.tasks = []

    def __repr__(self) -> str:
        return f"Clock({self.t})"

    def addTask(self, task: Callable) -> None:
        """ 
        Add a task function to execute at every clock tick. This function 
        should accept one argument - the time. 
        """
        if not callable(task):
            raise ClockError("task must be a callable")

        if not isinstance(task, Task):
            if task.__code__.co_argcount != 1:
                raise ClockError("task must accept only one input argument (time)")
        self.tasks.append(task)

    def tick(self) -> None:
        """ Run the clock once and do any assigned task.  """
        # complete all the tasks
        for task in self.tasks:
            task(self.t)
        self.t += self.dt

    def run(self, until: float) -> None:
        """ Run the clock until a given time.  """
        if not isinstance(until, float):
            raise ClockError("until must be a 'float'")

        while self.t < until:
            self.tick()

    def reset(self, t0: float = 0.) -> None:
        """ Reset the clock time to `t0` (0 by default). """
        self.t = t0

# ========================================
# status variable objects 
# ======================================== 

class Status:
    """ 
    A status class. It is used to specify a state, with an integer value 
    associated to it. 
    """
    __slots__ = 'value', 

    def __init__(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError("value must be an integer")
        self.value = value
    
    def __repr__(self) -> str:
        return f"Status({self.value})"

# ========================================
# space structures 
# ======================================== 

class RegionError(Exception):
    """ Exceptions used by region objects. """
    ...

class Region:
    """ 
    A rectangular region class. This region may hold objects at different 
    positions. It can be used for nearest neighbour detection etc.
    """
    __slots__    = 'vert0', 'vert1', 'width', 'name', 'objects', 'grid', '_gprops', 'getPosition', 

    REGION_COUNT = 0 # count the number of live regions    

    grid_properties = namedtuple(
                                    'grid_properties', 
                                    ['cellsize', 'subdiv_x', 'subdiv_y']
                                )    

    def __init__(self, x_width: float, y_width: float = ..., x0: float = 0., y0: float = 0., name: str = ..., ) -> None:
        if y_width is ... :
            y_width = x_width
        if not min(x_width, y_width) > 0.0:
            raise Region("region area must be positive and non-zero")
        
        self.width = Vector2D(x_width, y_width) # rectangle widths
        self.vert0 = Vector2D(x0, y0)           # lower-left vertex
        self.vert1 = self.vert0 + self.width    # upper-right vertex

        if name is ... : 
            name = f"R{Region.REGION_COUNT}"
        Region.REGION_COUNT += 1

        self.name  = name # name of this region

        self.objects = [] # objects placed on this region
        self.grid    = {} # grid subdivision of the region into cells

        self.getPosition = lambda __o: __o # function to get position from custom objects

        self._gprops = None # grid properties

    def __repr__(self) -> str:
        return f"<Region '{self.name}': vert0 = {self.vert0}, vert0 = {self.vert1}>"

    def randpos(self) -> Vector2D:
        """ 
        Get a random position of this region. The sequence of points 
        generated will be uniformly distributed inside the region. 
        """
        return Vector2D(
                            x = uniform(self.vert0.x, self.vert1.x), 
                            y = uniform(self.vert0.y, self.vert1.y), 
                        )
                        
    @property
    def xrange(self) -> tuple:
        return self.vert0.x, self.vert1.x

    @property
    def yrange(self) -> tuple:
        return self.vert0.y, self.vert1.y

    def isInside(self, point: Vector2D) -> bool:
        """ Tell if a point is inside the region or not. """
        if not isinstance(point, Vector2D):
            raise TypeError("point must be a 'Vector2D'")
        in_xrange = self.vert0.x <= point.x <= self.vert1.x
        in_yrange = self.vert0.y <= point.y <= self.vert1.y
        return in_xrange and in_yrange

    def placeObject(self, o: object) -> bool:
        """ 
        Place an object in this region, if its position is inside the 
        region. If placed successfully, return true and return false 
        otherwise. 
        """
        if self.isInside(self.getPosition(o)):
            self.objects.append(o)
            return True
        return False

    def setPositionGetter(self, f: Callable) -> None:
        """ 
        Set the function to get the position information from a custom 
        object. This function must accept one argument, the object and 
        return the position as a `Vector2D` object. 
        """
        if not callable(f):
            raise RegionError("f must be a callable")
        if f.__code__.co_argcount != 1:
            raise RegionError("f must accept exactly 1 input argument")
        self.getPosition = f

    def prepareGrid(self, cellsize: float) -> None:
        """ 
        Prepare the grid by subdividing the region and placing the objects 
        in to cells. 
        """
        # grid set-up:
        if cellsize <= 0.:
            raise RegionError("cellsize must be positive")

        subdiv_x = ceil(self.width.x // cellsize)
        subdiv_y = ceil(self.width.y // cellsize)

        self._gprops = Region.grid_properties(cellsize, subdiv_x, subdiv_y)

        # grid making:
        grid = {}
        for k, ok in enumerate(self.objects):
            pos  = self.getPosition(ok)
            cell = (
                        int((pos.x - self.vert0.x) // cellsize),
                        int((pos.y - self.vert0.y) // cellsize)
                    )
            if cell not in grid.keys():
                grid[cell] = []
            grid[cell].append(k)
        self.grid = grid

    def intersect(self, cell: tuple, center: Vector2D, radius: float) -> bool:
        """ Check if a cell intersect with a circle. """
        if self._gprops is None:
            raise RegionError("grid is not prepared")
        delta = self._gprops.cellsize
        x, y  = cell[0] * delta, cell[1] * delta
        nx    = clip(center.x, [x, x + delta])
        ny    = clip(center.y, [y, y + delta])
        return (center.x - nx)**2 + (center.y - ny)**2 < radius**2
    
    def neighboursOf(self, o: object, radius: float = ... ) -> list:
        """ Neighbours of a given object within a radius. """
        if self._gprops is None:
            raise RegionError("grid is not prepared")
        delta = self._gprops.cellsize

        pos  = self.getPosition(o)

        # cell the object sit
        i = int((pos.x - self.vert0.x) // delta)
        j = int((pos.y - self.vert0.y) // delta)

        step   = 1 if radius is ... else -int(-radius // delta)
        radius = delta if radius is ... else radius

        neighbours = []
        radius2    = radius**2

        # set of all cells to look:
        cells2look = product(
                                range(
                                        max(0, i-step), 
                                        min(i+step+1, self._gprops.subdiv_x)
                                     ), 
                                range(
                                        max(0, j-step), 
                                        min(j+step+1, self._gprops.subdiv_y)
                                     )
                            )

        for cell in cells2look:
            if cell not in self.grid.keys():
                continue
            if not self.intersect(cell, pos, radius):
                continue
            for k in self.grid[cell]:
                ok = self.objects[k]
                if o is ok:
                    continue
                if pos.dist2(self.getPosition(ok)) > radius2:
                    continue
                neighbours.append(ok)
        return neighbours

# ========================================
# objects to track variable evolution  
# ======================================== 

class HistoryError(Exception):
    """ Exceptions used by history objects. """
    ...

class History:
    """ 
    Variable history class. Used to track the history/evolution of a set of 
    variables. 
    """
    __slots__ = 'field_names', 'state_vector', '_states', '_time', 

    def __init__(self, field_names: list) -> None:
        if len(field_names) != len(set(field_names)):
            raise HistoryError("multiple fields with same name")
        if '__t' in field_names:
            raise HistoryError("`__t` is a researved name")

        # a state_vector holds the values of variables at a time
        self.state_vector = namedtuple("state_vector", field_names)

        self.field_names  = field_names 
        self._states      = []          # track the states 
        self._time        = []          # track time

    def __repr__(self) -> str:
        _repr = f"History fields={self.field_names}"
        if self._time:
            _repr += f" trange=({self.trange})"
        return f"<{_repr}>"

    @property
    def trange(self) -> tuple:
        if not self._time:
            raise HistoryError("nothing in history")
        return min(self._time), max(self._time)

    def pushstate(self, __t: float, *args, **kwargs) -> None:
        """ Push a state to the history. """
        state = self.state_vector(*args, **kwargs)
        if self._time:
            if __t in self._time:
                raise HistoryError("time is already recorded")
            if __t < self._time[-1]:
                raise HistoryError("time should be monotonically increasing")
        self._time.append(__t)
        self._states.append(state)

    def getstate(self, __t: float):
        """ Get a state at a time, with neareset neighbour interpolation. """
        __tmin, __tmax = self.trange
        if __t <= __tmin:
            return self._states[0]
        for i, ti in reversed(list(enumerate(self._time))):
            if __t >= ti:
                return self._states[i]
        if __t >= __tmax:
            return self._states[-1]

# ========================================
# time dependent tasks 
# ======================================== 

class TaskError(Exception):
    """ Exceptions used by task objects. """
    ...

class Task:
    """ 
    A task class, representing a task to do under a certain condition. When 
    the condition is met, a job function is executed. This job function must 
    accept a single argument, which is the task object and can modify any 
    object that is linked to the task. Set the `stop` function, if the task 
    is needed to be stopped when still active (e.g., to execute the task 
    only once, set it after first execution!). The condition is also a 
    function of one argument, time. All tasks use the global clock to track 
    time.
    """    
    __slots__ = 'active_at', 'job', 'stopped', 'paused', 'args', 'linked_obj'

    def __init__(self, active_at: Callable, job: Callable, linked_to: object = None, args: list = []) -> None:
        if not callable(active_at):
            raise TaskError("active_at must be a callable")
        elif active_at.__code__.co_argcount != 1:
            raise TaskError("active_at must accept one argument of type 'float'")

        if not callable(job):
            raise TaskError("job must be a callable")
        elif job.__code__.co_argcount < 1:
            raise TaskError("job must accept atleaset one argument of type 'Task'")

        self.active_at  = active_at # activation condition
        self.job        = job       # actual task to do
        self.args       = args      # extra arguments to pass to job
        self.stopped    = False     # execution is stopped
        self.paused     = False     # pause execution
        self.linked_obj = linked_to # object linked with this task

    def __call__(self, t: float) -> None:
        """ Do the task, given the time. """
        if self.active_at(t) and not self.stopped:
            if self.paused:
                return
            self.job(self, *self.args)

    def stop(self) -> None:
        """ Stop the task from executing further. """
        self.stopped = True

    def pause(self) -> None:
        """ Pause the task. """
        self.paused = True

    def restart(self) -> None:
        """ Restart a paused task. """
        if self.paused:
            self.paused = False

    def getLinkedObject(self) -> object:
        """ Get the linked object. """
        return self.linked_obj

# ========================================
# object modifiers and generators
# ======================================== 

class ModifierError(Exception):
    """ Exceptions used by modifier objects. """
    ...

class Modifier:
    """
    A modifier class. Modifier objects can modify attributes of other 
    objects, when the modifier is enabled or desabled by calling the `apply` 
    and `remove` functions on that object. Both these functions are user 
    defined and specified by the `apply_action` and `remove_action`     
    arguments, which are functions of two arguments one is the mask itself 
    and the other of type `object`.
    """
    __slots__ = 'name', 'value', '_apply', '_remove', 

    def __init__(self, value: Any, apply_action: Callable, remove_action: Callable, name: str) -> None:
        if not callable(apply_action):
            raise TypeError("apply_action should be a callable")
        elif apply_action.__code__.co_argcount != 2:
            raise ModifierError("apply_action should be a single argument function")

        if not callable(remove_action):
            raise TypeError("remove_action should be a callable")
        elif remove_action.__code__.co_argcount != 2:
            raise ModifierError("remove_action should be a single argument function")
        
        if not isinstance(name, str):
            raise TypeError("name must be a 'str'")
        elif not name:
            raise ModifierError("name should not be empty")

        self.value   = value
        self._apply  = apply_action
        self._remove = remove_action
        self.name    = name

    def __repr__(self) -> str:
        return f"<Modifier '{self.name}' value={self.value}>"

    def apply(self, o: object) -> None:
        """ Apply the modifier on the object. """
        self._apply(o)

    def remove(self, o: object) -> None:
        """ 
        Remove the modifier from the object. Be careful when calling this on 
        non-modified objects, which results in unwanted results - this will 
        not check the modifier is applied on it. 
        """
        self._remove(o)

class GeneratorError(Exception):
    """ Exceptions used by generator objects. """
    ...

class Generator:
    """ 
    An object generator class. This generator object stores some parameters 
    and generate random objects using that parameters. The `generator` 
    argument must be a function taking one argument and returns a random 
    object.
    """
    __slots__ = 'params', 'generator', 

    def __init__(self, generator: Callable = ..., **params) -> None:
        self.params = params

        if not callable(generator):
            raise GeneratorError("generator must be a callable")
        elif generator.__code__.co_argcount != 1:
            raise GeneratorError("generator must accept a single argument of type 'Generator'")
        self.generator = generator

    def __getattr__(self, __name: str) -> Any:
        return self.params[__name]
    
    def __setattr__(self, __name: str, __value: float) -> None:
        try:
            super().__setattr__(__name, __value)
        except AttributeError:
            if __name not in self.params.keys():
                raise AttributeError(f"generator has no parameter '{__name}'")
            self.params[__name] = __value

    def __call__(self) -> object:
        return self.generator(self)


# ========================================
# useful functions 
# ======================================== 

def randdir(a: float = 0., b: float = 2*pi) -> Vector2D:
    """ Return a unit vector pointing in a random direction. """
    t = uniform(a, b)
    return Vector2D(t, fmt = 'polar')

def isnumeric(x: Any):
    """ Check if the value is a number. """
    return isinstance(x, (float, int))

def isprobability(x: float) -> bool:
    """ Check if a value is a valid probability. i.e., in [0, 1] range. """
    if isnumeric(x):
        return (0.0 <= x <= 1.0)
    return False

def map2(x: float, _from: list, to: list) -> float:
    """ Linearly map a value from one range to another. """
    (a1, b1), (a2, b2) = _from, to

    t = (x - a1) / (b1 - a1)
    return a2 + t * (b2 - a2)

def clip(x: float, to: list) -> float:
    """ Clip a number x to the range [a, b], specified by `to` """
    a, b = to
    return max(a, min(x, b))



