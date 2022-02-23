#!/usr/bin/python3
import warnings
from . import utils as u
from random import uniform, choice
from typing import Callable, Any

def random(b: float = 1, a: float = 0) -> float:
    """ Return a random number between 0 and 1. """
    return uniform(a, b)

class EpidemicError(Exception):
    """ Base class of exceptions used by epidemic module. """
    ...

class EpidemicWarning(Warning):
    """ Base class of warnings used by epidemic module. """
    ...

# =================================
# global clock setup
# =================================
 
# global clock object is used to make all the simulation instances use the 
# same clock. it can be accessed or its attributes can be modified by user.
# one can also set a clock before the start of simulations. 
gclock = u.Clock(10) 

def getGlobalClock() -> u.Clock:
    """ Get the global clock, if any. """
    global gclock

    if gclock:
        return gclock
    raise EpidemicError("no glabal clock is found")

def setGlobalClock(clk: u.Clock) -> None:
    """ Set the global clock. """
    global gclock

    if not isinstance(clk, u.Clock):
        raise TypeError("clk must be a 'Clock' object")
    gclock = clk


# =================================
# objects
# =================================

# a disease model is used to parametrise an infectious disease in terms of
# some attributes. 
class Disease:
    """ 
    An infectious disease class. A disease is modelled by parameters such as 
    the recovery time, infection probability and information such as whether 
    it give immunity after recovery etc. These objects only store data and 
    does not do anything else, except drawing random data. 
    """
    __slots__ = 'pinf', 'trec', 'pimt', 'name',     

    def __init__(self, name: str, p_infect: float, t_recovery: float, p_immunity: float = 1.0) -> None:
        if not u.isprobability(p_infect):
            raise ValueError("p_infect must be in range [0, 1]")
        elif not u.isprobability(p_immunity):
            raise ValueError("p_immunity must be in range [0, 1]")
        elif t_recovery < 0:
            raise ValueError("t_recovery must be positive")
        elif not isinstance(name, str):
            raise TypeError("name must be a 'str'")
        elif not name:
            raise ValueError("name should not be empty")

        self.name = name
        self.pinf = p_infect
        self.pimt = p_immunity
        self.trec = t_recovery

    def __repr__(self) -> str:
        return f"<Disease '{self.name}': p={self.pinf:.3g}, trec={self.trec:.3g}, pimt={self.pimt:.3g}>"

# objects such as mask and vaccine modify the attributes of a person. they 
# are object modifiers with an `apply` method to apply it on an object and 
# a `remove` method, which is called when the modifier is removed. 
class Mask(u.Modifier):
    """ 
    A mask class. Mask objects should reduce spreading of the infection. A 
    mask object applied to a person can modify his/her infection attributes. 
    By default, it will reduce the radius of infection by a factor, but this 
    behaviour can be modified by the `on_wearing` and `on_removing` 
    arguments. 
    """
    
    def __init__(self, value: float, on_wearing: Callable = None, on_removing: Callable = None, name: str = 'mask') -> None:
        if not on_wearing:
            on_wearing = self._default_onWearing
        if not on_removing:
            on_removing = self._default_onRemoving

        super().__init__(value, on_wearing, on_removing, name)

    def __repr__(self) -> str:
        return super().__repr__().replace('Modifier', 'Mask')

    def _default_onWearing(self, o: object) -> None:
        """ 
        Default behaviour when wearing a mask - reduse the radius 
        modification factor of the person by the value attribute (float). 
        """
        if not isinstance(o, Person):
            raise TypeError("object must be a 'Person'")
        
        o.ri_modifier = self.value

    def _default_onRemoving(self, o: object) -> None:
        """
        Default behaviour when removing a mask - reset the radius        
        modification factor of the person to 1. 
        """
        if not isinstance(o, Person):
            raise TypeError("object must be a 'Person'")
        o.ri_modifier = 1.0

    def wear(self, o: object) -> None:
        """ Wear this mask. """
        self.apply(o)
        o.mask = self

class Vaccine(u.Modifier):
    """ 
    A vaccine class. Vaccines also reduse the spreading of a disease and its 
    effects are irreversible. Vaccinated individulals have lesser or no 
    probability for getting infected. Default behaviour af a vaccine is to 
    reduce the infection probability of an un-infected person and reduce the 
    recovery time for an infected person. But, this can be modified by 
    specifying the `on_vaccination` argument.
    """

    def __init__(self, pvalue: float, tvalue: float = ..., on_vaccination: Callable = None, name: str = 'vaccine') -> None:
        if tvalue is ... :
            tvalue = pvalue
        if pvalue < 0. or tvalue < 0.:
            raise ValueError("values must be positive") 

        if not on_vaccination:
            on_vaccination = self._default_onVaccination

        value = [pvalue, tvalue] 
        super().__init__(value, on_vaccination, self._rm, name)    

    def __repr__(self) -> str:
        return super().__repr__().replace('Modifier', 'Vaccine')

    def _default_onVaccination(self, o: object):
        """ 
        Default behaviour of a vaccine - reduce probability for un-infected 
        people and reduce recovery time by infected people. 
        """
        if not isinstance(o, Person):
            raise TypeError("object must be a person")

        # vaccine will directly modify the `pi` and `tminus_rec` attributes 
        # of a person, as its effects are irreversible 
        pfactor, tfactor = self.value
        
        o.pi *= pfactor # vaccine reduce the infection probability
        if o.state is INFECTED:
            o.tminus_rec *= tfactor # vaccines reduce the recovery time
    
    def _rm(self, o: object) -> None:
        """ Vaccination cannot be reversed - raise a warning. """
        warnings.warn("vaccination is irreversible", EpidemicWarning)

    def vaccinate(self, o: object) -> None:
        """ Give the person this vaccine. """
        self.apply(o)
        o.vaccine = self

# constants to specify the infection status. there will be three states
# for infection: [1] exposed/susceptible - the person can get infection 
# from others, [2] infected - the person is has got an infection and 
# [3] recovered - the person has recovered from the disease and will not 
# get infected again. 
EXPOSED, INFECTED, RECOVERED = u.Status(0), u.Status(1), u.Status(2)

# a person object is the main object, representing a randomly moving person
# in a community/region. 
class PersonError(EpidemicError):
    """ Exceptions used by person objects. """
    ...

class Person:
    """ 
    A random person class. A random person has some properties and 
    attributes associated with him, which tell his/her behaviour in a 
    community attacked by a infectiuous disease. 
    """
    __slots__ = (
                    'p', 'v', 'state', '_ri', '_pi', 'disease', 'vaccine', 
                    'mask', 'at_quarentine', 'tminus_rec','pi_modifier', 
                    'v_modifier', 'ri_modifier', 'alive', 'community', 
                )

    def __init__(self, pos: u.Vector2D, vel: u.Vector2D, r_infect: float) -> None:
        if not isinstance(pos, u.Vector2D):
            raise TypeError("pos must be a 'Vector2D' object")
        elif not isinstance(vel, u.Vector2D):
            raise TypeError("vel must be a 'Vector2D' object")
        self.p, self.v = pos, vel

        if not u.isnumeric(r_infect):
            raise TypeError("ri must be a number")
        self._ri = r_infect # infection radius

        # other parameters:
        self.state         = EXPOSED # person is initially un-infected
        self._pi           = 0.0     # infection probability
        self.tminus_rec    = 0.0     # time left to recover (if infected)
        self.at_quarentine = False   # is at quarentine
        self.alive         = True    # is this person alive
        
        # modifier values (weight):
        self.pi_modifier = 1.0
        self.v_modifier  = 1.0
        self.ri_modifier = 1.0

        # objects
        self.disease   = None # disease this person got
        self.vaccine   = None # vaccine this person took
        self.mask      = None # mask this person is wearing
        self.community = None # community of this person belongs

    def __repr__(self) -> str:
        s = 'SIR'[self.state.value]
        return f"<Person: p={self.p}, v={self.v}, ri={self.ri}, state='{s}'>"

    @property
    def clock(self) -> u.Clock:
        return getGlobalClock()

    @property
    def pi(self) -> float:
        return self._pi * self.pi_modifier

    @property
    def ri(self) -> float:
        return self._ri * self.ri_modifier

    @property
    def veff(self) -> u.Vector2D:
        return self.v * self.v_modifier # effective velocity

    def hasVaccinated(self) -> bool:
        return self.vaccine is not None

    def wearingMask(self) -> bool:
        return self.mask is not None
    
    def die(self) -> None:
        self.alive = False

    def goQuarentine(self) -> None:
        """ 
        Start quarentine. This will effective put the person out of the 
        community and he/she will not be infected nor transmit the 
        infection. 
        """
        self.at_quarentine = True

    def endQuarentine(self) -> None:
        """ 
        End the quarentine. This will put the person back to the community 
        and make susceptible or recovered based on the model.
        """
        self.at_quarentine = False

    def wearMask(self, mask: Mask) -> None:
        """ 
        Wear a mask. This will cause some properties of this person to be 
        modified. The person can have the freedom to choose whether to wear 
        a mask or not, simulated by a probability (default is 1). 
        """
        if not isinstance(mask, Mask):
            raise TypeError("mask object must be a 'Mask'")
        mask.wear(self) 

    def removeMask(self) -> None:
        """ 
        Remove the mask. This will cause some properties of this person to 
        be modified. 
        """
        if self.mask:
            self.mask.remove(self)
        self.mask = None

    def takeVaccine(self, vaccine: Vaccine) -> None:
        """
        Take a shot of vaccine. This will cause some properties of this 
        person to be modified. The person can have the freedom to choose to 
        get vaccinated or not, simulated by a probability (default is 1). 
        """
        if not isinstance(vaccine, Vaccine):
            raise TypeError("vaccine object must be a 'Vaccine'")
        vaccine.vaccinate(self)

    def join(self, o: object) -> None:
        """ Join a community. """
        if not isinstance(o, Community):
            raise TypeError("object must be a 'Community'")
        return o.add(self)

    def recover(self) -> None:
        """ 
        Recover from the disease. It also end the quarentine if this person 
        is currently in quarentine.
        """
        if self.state is not INFECTED:
            return

        self.state   = RECOVERED

        # some diseases may not give immunity after infection. so make 
        # people susceptible by the disease's probability to give immunity 
        # against another infection.  
        if random() > self.disease.pimt:
            self.state = EXPOSED

        self.disease = None # now the disease is cured!
    
        if self.at_quarentine:
            self.endQuarentine()

    def makeInfected(self, disease: Disease) -> None:
        """ Make a person infected. """
        if not self.state is EXPOSED or self.at_quarentine:
            return

        if not isinstance(disease, Disease):
            raise TypeError("disease must be a 'Disease'")

        self.disease = disease
        self.state   = INFECTED

        # if the person is part of a community having this disease, then use 
        # community specific recovery time and probability. otherwise use 
        # disease specific values. those community specific values will be 
        # spreaded about the disease specific values.
        trec, pinf = disease.trec, disease.pinf
        if self.community:
            if self.community.disease is disease:
                trec, pinf = self.community.tr(), self.community.pi()
        self.tminus_rec, self._pi = trec, pinf
        
    def infect(self, o: object) -> None:
        """
        Infect another person, if this person is already infected. He/she 
        can't infect others if in quarentine. Infection is transmitted with 
        a probability, means the other may survive the infection.
        """
        if not isinstance(o, Person):
            return
        if self.state is not INFECTED:
            return
        if random() < self.pi:
            o.makeInfected(self.disease)

    def infectOthers(self) -> None:
        """ 
        Infect other people in the neighborhood, if they are closer than the 
        infection radius property. Infection is transmitted only when the 
        person is diseased or not in quarentine.
        """
        if not self.community:
            return
        for person in self.community.neighboursOf(self):
            self.infect(person)
        return

    def walk(self) -> None:
        """ 
        Walk through the 2D space or specified region. If a region is linked 
        to this person (through the community), then limit the position and 
        velocity by its settings.
        """
        if not self.alive:
            return # person is dead - can be removed from the community

        dt   = self.clock.dt
        veff = self.v * self.v_modifier # effective velocity

        self.p = self.p + veff * dt
        
        # update the infection status of this person, as the clock has now
        # completed a tick!
        if self.state is INFECTED:
            self.tminus_rec = max(self.tminus_rec - dt, 0.0)
            if self.tminus_rec <= 0.0:
                self.recover() # recover and end quarentine

       
        if self.community:
            # if the person is a part of a community, apply community 
            # specific boundary conditions. 
            self.community.applyBC(self.p, self.v)
            
            # there may be a death-rate/probability. so, 'die' at that 
            # probability!
            if random() < self.community.pdeath:
                self.die()

class Community(u.Region):
    """ 
    A community class. This represents a community of randomly moving 
    people. A disease can be introdused into that community and allowed to 
    spread. People in the community has randomly distributed properties, 
    such as the radius of influence, infection probability, speed etc. 
    These can be specified using `Generator` objects with their mean value 
    specified by the `value` parameter.  
    """
    __slots__ = (
                    'size', 'ri', 'tr', 'pi', 'randvel','bc', 'disease', 
                    'mask_on', 'take_vaccine','force_quarentine', 
                    'pdeath', 'pbirth', 
                )

    def __init__(self, size: int, ri: float = ..., x_width: float = ..., y_width: float = ..., x0: float = 0, y0: float = 0, ri_generator: u.Generator = ..., pi_generator: u.Generator = ..., tr_generator: u.Generator = ..., v_generator: u.Generator = ..., speed: float = 10.0, p_death: float = 0.0, p_birth: float = 0.0, name: str = ...) -> None:
        # initialise the region of the community
        if x_width is ... :
            raise ValueError("x_width is a required argument")
        super().__init__(x_width, y_width, x0, y0, name)

        # various generators
        if ri_generator is ... :
            if ri is ... :
                raise EpidemicError("ri is required if no generator is specified")
            ri_generator = u.Generator(lambda __o: __o.value, value = ri)
        elif not isinstance(ri_generator, u.Generator):
            raise TypeError("ri_generator must be a 'Generator' object")
        self.ri = ri_generator

        if pi_generator is ... :
            pi_generator = u.Generator(lambda __o: __o.value, value = 0.0)
        elif not isinstance(pi_generator, u.Generator):
            raise TypeError("pi_generator must be a 'Generator' object")
        self.pi = pi_generator

        if tr_generator is ... :
            tr_generator = u.Generator(lambda __o: __o.value, value = 0.0)
        elif not isinstance(tr_generator, u.Generator):
            raise TypeError("tr_generator must be a 'Generator' object")
        self.tr = tr_generator

        if v_generator is ... :
            v_generator = u.Generator(lambda __o: __o.value * u.randdir(), value = speed) 
        elif not isinstance(v_generator, u.Generator):
            raise TypeError("v_generator must be a 'Generator' object")
        self.randvel = v_generator

        # community have a birth rate and death rate (probabilities)
        self.pdeath, self.pbirth = p_death, p_birth

        self.setPositionGetter(lambda __o: __o.p) # to get people position
        
        # initialise the objects/people
        for _ in range(size):
            p = Person(self.randpos(), self.randvel(), self.ri())
            p.community = self
            self.objects.append(p)
        self.prepareGrid()
        
        self.size = size

        self.setBC() # set periodic boundary conditions

        self.disease = None # disease spreading in the community

        # rules to follow:
        self.mask_on          = False
        self.take_vaccine     = False
        self.force_quarentine = False

    def __repr__(self) -> str:
        return f"<Community '{self.name}': size={self.size} region=[{self.vert0}, {self.vert1}]>"

    @property
    def clock(self) -> u.Clock:
        return getGlobalClock()

    @property
    def people(self) -> list:
        return self.objects

    def add(self, o: object) -> bool:
        """ Add a member to the community. """
        if not isinstance(o, Person):
            raise EpidemicError("object must be a 'Person'")
        if not o.alive:
            return 

        # if the person is infected by some disease, initialise an infection
        if o.state is INFECTED:
            self.putDisease(o.disease)
        o.community = self
        return self.placeObject(o)

    def setBC(self, bc: Any = 'periodic') -> None:
        """ 
        Set a boundary condition on the objects in this community. This can 
        be a string indicating a named boundary condition (available 
        options: `periodic`, `none`) or a callable of two arguments of type 
        `Vector2D`. Boundary condition functions will modify their arguments 
        and not return anything. 
        """
        def bc_none(p: u.Vector2D, v: u.Vector2D) -> None:
            """ no boundary conditions """
            ...

        def bc_periodic(p: u.Vector2D, v: u.Vector2D) -> None:
            """ periodic boundary conditions """
            p.x = (p.x - self.vert0.x) % self.width.x + self.vert0.x
            p.y = (p.y - self.vert0.y) % self.width.y + self.vert0.y

        if bc is None or bc == 'none':
            self.bc = bc_none
            return
        elif bc == 'periodic':
            self.bc = bc_periodic
            return

        if not callable(bc):
            raise TypeError("bc must be a callable")
        elif bc.__code__.co_argcount != 2:
            raise ValueError("bc must accept two arguments")
        self.bc = bc

    def applyBC(self, p: u.Vector2D, v: u.Vector2D) -> None:
        """ Apply the boundary conditions. """
        self.bc(p, v)

    def putDisease(self, disease: Disease) -> bool:
        """ Put a disease to the community. """
        if self.disease is not None:
            return False
        if not isinstance(disease, Disease):
            return False # not a disease, nothing to fear!
        self.pi.value = disease.pinf
        self.tr.value = disease.trec
        self.disease = disease
        return True
        
    def startInfection(self, disease: Disease, initial_cases: int = 1) -> None:
        """ 
        Start an infection in the community. This is done by taking some 
        random people and infecting them. 
        """
        if not self.putDisease(disease):
            raise EpidemicError("cannot start infection")
        elif not self.people:
            raise EpidemicError("community is empty")
        while initial_cases:
            p = choice(self.people)
            while p.state is INFECTED:
                p = choice(self.people)
            p.makeInfected(disease)
            initial_cases -= 1
        return

    def spreadInfection(self) -> None:
        """ Spread infection in the community. """
        if not self.people:
            return
        self.prepareGrid()

        if not self.disease:
            return
        for person in self.people:
            person.infectOthers()

    def prepareGrid(self) -> None:
        return super().prepareGrid(cellsize = max(map(lambda __o: __o.ri, self.people)))

    def evolve(self) -> None:
        """ Evolution of the community and linked people. """
        self.spreadInfection()
        for person in self.people:
            person.walk()

        # TODO: deaths and births
        return

    
