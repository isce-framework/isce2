#!/usr/bin/env python3
"""
section Platform Motion (geo.motion)

After all that, you still don't have platform motion. Enter the motion
module. It requires numpy, since you will have array_like attributes.
The SpaceCurve class is basically Vector which takes that into consider-
ation. (Recall the fundamental theorem of Space Curves? That curvature
and torision define a unique space curve? Yes, that one-- well space
curves define all that: velocity, normal, acceleration, angular velocity,
yadda yadda. They key property is you can define a local tangent frame,
with:

x   parallel to the curve's velocity
y   = z X x
z   is some "z" orthogonal to x. The default "z" is DOWN, but you can
                                 make it UP, or something else.

Hence, given a 3-DoF motion history, you get the transformation from
level cartesian space to the tangent frame. Now if you throw in attitude,
represented by any kind of rotation, boom, you have an affine
transformation to body coordinates.

But wait: these things all have array_like attributes, that means in
one object, you have the trasnformtion from a local pegged coordinate
system the body frame AT EVERY POINT IN THE MOTION HISTORY.

Now stop and think about that for a minute. IF you were still suffering
primative obession, using arrays for vectors, and using standalone
functions for coordinate transformations--you be in a real pickle.
All these arrays, which are JUST NUMBERS and have NO intrinsici meaning-
no you the developer has to keep it straight. Then, you have to pass
them to functions, and then to other functions-- how you deal with the
fact that the functions are not constant--- I don't know- but you do.

None of that. You got a gps history and an attitude history:

f = SpaceCurve(*gps).tld2body(imu)

f(look), f(target), etc...

does the whole affine transformation at every point.
"""
## \namespace ::geo::motion::motion SpaceCurve's and their Tangent Frames
__date__ = "10/30/2012"
__version__ = "1.21"
print ("importing %s version::%s"%(__name__, __version__))

import itertools
import operator
from functools import wraps
## you need numpy for this module to be useful
import numpy

## Note: this is an __all__ controlled import, so it does not pollute the
## namespace
from geo import euclid
from geo.euclid import charts
from geo.motion import dxdt


## Reative Body Frame Directions
UP = euclid.Z
DOWN = -UP
FORWARD = euclid.X
BACKWARD = -FORWARD


## tangent-left-up to tangent-right-down transformation
tlu2trd = charts.Roll(180)
## tangent-right-downto tangent-left-up  transformation
trd2tlu = ~tlu2trd

## This is a derivative decorator:\n
## \f$D_n(f) \rightarrow \frac{d^nf}{ds^n} \f$ \n
## If you can follow how this works, then you might be a python guru.
def derivative(n):
    """derivative(n) returns a function decorator--which takes an input function
    and returns a function that compute the n-th derivative of the function"""
    def dnf_dtn(func):
        """The is the decorator that decorates it's argument "func",
        making it into the n-th derivative of func, e.g:

        if func(x) --> x**4,  and n=2, then

        dnf_dtn --> 12*x**2
        """
        # Usage: decorators with arguments compute a new decorator
        # (here, dnf_dtn) which then decorates the method, turning it into
        # dfunc_dt. dfunc_dt calls the prime method with the derivative operator
        # raised to "n". Finally, functools.wraps decorates it so that the user
        # has access original method's docstring (that's all @wraps does).
        # That it's a deocrator that returns a decoraated decorator should not
        # be seen as "obscure"-- it's just the right way to do it.
        @wraps(func)
        def dfunc_dt(self):
            dt = self.Dx(self.t)
            return self.broadcast(func).prime(dt=dt**n)
        return dfunc_dt
    return dnf_dtn


## \f$ \vec{f(\vec{v})} \rightarrow \vec{f}/||v|| \f$ \n
## This decorator normalizes the output with abs(self):
def normalized(func):
    """ func() --> func()/||self|| decorator"""
    @wraps(func)
    def nfunc(self, *args, **kwargs):
        return func(self, *args, **kwargs)/abs(self)
    return nfunc

## \f$ \vec{f(\vec{v})} \rightarrow \vec{f}/||f|| \f$ \n This decorator
## normalizes the output with abs(output)
def hat(func):
    """func() --> func().hat() decorator"""
    @wraps(func)
    def hfunc(self, *args, **kwargs):
        return func(self, *args, **kwargs).hat()
    return hfunc


## \f$ f(\vec{v}) \rightarrow 1/f  \f$  \n This decorator returns the
## reciprocol of the function
def radius_of(func):
    """func --> 1/func decorator"""
    @wraps(func)
    def rfunc(self, *args, **kwargs):
        return func(self, *args, **kwargs)**(-1)
    return rfunc

##  \f$ \vec{f(\vec{v})} \rightarrow ||f||  \f$  \n This decorator returns
## the magnitude of the vector function
def magnitude(func):
    """func --> |func| decorator"""
    @wraps(func)
    def mfunc(self, *args, **kwargs):
        return abs(func(self, *args, **kwargs))
    return mfunc


## \f$ f(\vec{v}) \rightarrow O(*f) \f$ \n   A decorator for operators of
## functions-- now this is just silly.
def starfunc(op):
    """starfunc(op) decorator takes a binary operator "op"
    and wraps a method that MUST return 2 objects: left
    and right, and returns:

    op(left, right)
    """
    def starop(func):
        @wraps(func)
        def starF(self, *args, **kwargs):
            return op(*func(self, *args, **kwargs))
        return starF
    return starop

## Vector with <a href="http://mathworld.wolfram.com/SpaceCurve.html">
## Space Curve</a> capabilities
class SpaceCurve(euclid.Vector):
    """v = SpaceCurve(x, y, z)

    A space curve is a Vector, with the expectation
    that:

    x
    y
    z

    are 1 dimensional numpy.ndarray with at least 3 points, on which various
    numeric derivative operations are performed
    """

    ## \f$ D_x \f$ Differential operator wrt \f$x\f$\n see http://en.wikipedia.org/wiki/Differential_operator \n see http://docs.scipy.org/doc/scipy/reference/misc.html for other derivative functions \n This is an Inner Class used for differentiation, and if you don't like, you can overide it and use your own differentiater-- just set SpaceCurve.Dx = my favorite differental operator, of course, it may need to support composition (but not if you only need a tangent plane coordinate system)
    class Dx(object):
        """Derivative operator class, so for example:

        D = Dx(x)                                        -- D is d/dx

        So, for exmaple:

        D(special.jn(n,x))        = special.jvp(n,x,1)   --D(y)   = y'(x)
        pow(D,2,(special.jn(n,x)) = special.jvp(n,x,2)   --D^2(y) = y"(x)

        (D**2)(y) --> lammda y: pow(D,2,y) also...

        Of course, x must be defined sensibly

        NOTE: this class uses dxdt.deriv , so if you don't
        like it, change it.

        """
        ## x in \f$ \frac{d}{dx} \f$
        def __init__(self, x):
            self.x = x
            return None
        ## Dx(x)(y) = \f$ \frac{dy}{dx} \f$,  a wrapper deriv() .
        def __call__(self, y):
            return dxdt.deriv(self.x, y)
        ## \f$\frac{d^n y}{dx^n} =  {\rm pow}(x, n, y) \f$  = pow(x, n, y)
        def __pow__(self, n, y=None):
            return (
                lambda y: pow(self, n, y) if y is None else
                eval('self('*n+'y'+')'*n) if n > 0 else (~self)**(-n)
                )
        ## Add derivatives
        def __add__(self, other):
            return lambda y: self(y) + other(y)
        ## R to L composition
        def __mul__(self, other):
            return lambda y: self(other(y))
        ## Dilation
        def __rmul__(self, other):
            return lambda y: other*self(y)
        pass

    ## A Vector (elementwise) with optional t parameter
    def __init__(self, x, y, z, t=None):
        super(SpaceCurve, self).__init__(x, y, z)
        ## t is not in use yet, and when it is, it preculde the t decorator.
        self.t = np.arange(len(x)) if t is None else t
        return None

    ## len() --> length of the attributes (and they have to be equal)
    def __len__(self):
        try:
            lens = map(len, self.iter())
        except TypeError:
            raise TypeError("Spacecurve's attributes have no len()")
            pass
        if lens[0] == lens[1] == lens[2]:
            return lens[0]
        raise ValueError("SpaceCurve's attributes have differnent len()'s.")

    ## \f$ \vec{v}'_i = \frac{dv_i}{dt} \f$ \n Vector derivative with resept to argument or index, keywords specify \f$t\f$ and/or  \f$\partial \f$, or \f$\frac{d}{dt} \f$.
    def prime(self, t=None, dx=None, dt=None):
        """dT/dt = T.prime([t=None [, dx=None, [dt=None, [inplace=None]]])

        t defaults to (1,2,..., len(T)) if is None

        dx        controls numeric differentiation with respect to t, and
        defaults to Dx(t), evaulated as dx(t)-- so if you want to overide
        with a better (e.g. filtered derrivative), it has to be curried--
        that is:

        dx(t, x) = dx(t)(x).

        Now if you was dt, then

        inplace is not supported.
        """
        dt = self.Dx(self.t)
        return self.broadcast(dt)

    ## This is the 0-th order derivative, and it starts the decorator off
    def __call__(self, t=None, dx=None):
        return self

    ## \f$ \vec{v} \equiv \frac{d\vec{r}}{dt} \f$ \n
    ## <a href="http://mathworld.wolfram.com/Velocity.html for details.">
    ## Velocity</a>, as a derivative() decorated SpaceCurve.__call__
    @derivative(1)
    def velocity(self):
        """1st derivative of self()"""
        return self


    ## \f$ \vec{a} \equiv \frac{d\vec{v}}{dt} = \frac{d^2\vec{r}}{dt^2} \f$ \n
    ## <a href="http://mathworld.wolfram.com/Acceleration.html for details.">
    ## Velocity</a>, as a derivative() decorated SpaceCurve.__call__
    @derivative(2)
    def acceleration(self):
        """2nd derivative of self()"""
        return self

    ## \f$ \vec{j} \equiv \frac{d\vec{a}}{dt} = \frac{d^3\vec{r}}{dt^3} \f$ \n
    ## <a href="see http://en.wikipedia.org/wiki/Jerk_(physics)">Jerk</a>, as
    ## a derivative() decorated SpaceCurve.__call__
    @derivative(3)
    def jerk(self):
        """3rd derivative of self()"""
        return self

    ## \f$ \vec{s} \equiv \frac{d\vec{j}}{dt} = \frac{d^4\vec{r}}{dt^4} \f$\n
    ## <a href="see http://en.wikipedia.org/wiki/Jounce)">Jounce</a>, aka
    ## SpaceCurve.Snap, as a derivative() decorated SpaceCurve.__call__
    @derivative(4)
    def jounce(self):
        """4th derivative of self()"""
        return self

    ## Snap is Jounce
    snap = jounce

    ## \f$ \frac{{d^5}\vec{r}}{dt^5} \f$ \n
    ## <a hef="http://en.wikipedia.org/wiki/Jounce">Crackle</a>, asa a
    ## derivative() decorated SpaceCurve.__call__
    @derivative(5)
    def crackle(self):
        """5th derivative of self()"""
        return self

    ## \f$ \frac{{d^6}\vec{r}}{dt^6} \f$ \n
    ## <a hef="http://en.wikipedia.org/wiki/Jounce">Pop</a>, asa a derivative()
    ## decorated SpaceCurve.__call__
    @derivative(6)
    def pop(self):
        """6th derivative of self()"""
        return self

    ## \f$ \frac{{d^7}\vec{r}}{dt^7} \f$ \n
    ## <a hef="http://en.wikipedia.org/wiki/Jounce">Lock</a>, asa a
    ## derivative() decorated SpaceCurve.__call__
    @derivative(7)
    def lock(self):
        """7th derivative of self()"""
        return self

    ## \f$ \frac{{d^8}\vec{r}}{dt^8} \f$ \n
    ## <a hef="http://en.wikipedia.org/wiki/Jounce">Drop</a>, asa a
    ## derivative() decorated SpaceCurve.__call__
    @derivative(8)
    def drop(self):
        """8th derivative of self()"""
        return self

    ## \f$ \frac{{d^9}\vec{r}}{dt^9} \f$ \n
    ## <a hef="http://en.wikipedia.org/wiki/Jounce">Shot</a>, asa a
    ## derivative() decorated SpaceCurve.__call__
    @derivative(9)
    def shot(self):
        """9th derivative of self()"""
        return self

    ## \f$ \frac{{d^{10}}\vec{r}}{dt^{10}} \f$ \n
    ## <a hef="http://en.wikipedia.org/wiki/Jounce">Put</a>, asa a
    ## derivative() decorated SpaceCurve.__call__
    @derivative(10)
    def put(self):
        """10th derivative of self()"""
        return self


    ##  \f$ |\vec{v}| \f$ \n
    ## <a href="http://mathworld.wolfram.com/Speed.html"> Speed</a>, as a
    # magnitude() decorated velocity()
    @magnitude
    def speed(self):
        return self.velocity(t, dx, dt)


    ## \f$ \hat{T} \equiv \hat{v} \f$ \n
    ## <a href="http://mathworld.wolfram.com/TangentVector.html">Tangent Vector
    ## </a>, as a hat() decorated velocity()
    @hat
    def tangent(self):
        """Tangent is the 'hat'-ed Velocity"""
        return self.velocity(t, dx, dt)


    ## \f$ \hat{N} \equiv \hat{\dot{\hat{T}}} = \frac{d\hat{T}}{dt}/|\frac{d\hat{T}}{dt}| \f$
    ## <a href="http://mathworld.wolfram.com/NormalVector.html">Normal Vector
    ## </a>, as a starfunc() decorated binormal() and tangent().
    @starfunc(operator.__xor__)
    def normal(self):
        """Normal is the cross product of the binormal and the Tangent"""
        return self.binormal(), self.tangent()

    ## \f$ \vec{\omega} = \vec{v}/r \f$ \n
    ## <a href="http://mathworld.wolfram.com/AngularVelocity.html">
    ## Angular Velocity</a>, as a normalized() decorated velocity()
    @normalized
    def angular_velocity(self):
        """angular_velocity os the normalized velocity"""
        return self.velocity()

    ## \f$ \vec{\alpha} = \vec{\omega}' = \vec{a}/r \f$ \n
    ## <a href="http://mathworld.wolfram.com/AngularAcceleration.html">
    ## Angular Acceleration</a>, as a normalized() decorated acceleration().
    @normalized
    def angular_acceleration(self):
        """AngularAccelration os the normalized Acceleration"""
        return self.acceleration()


    ## \f$ \vec{B} = \vec{v} \times \vec{a} \f$ \n as a starfunc() __xor__ decorated velocity() and v.v
    @starfunc(operator.__xor__)
    def velocity_X_acceleration(self):
        """Velocity crossed with Acceleration"""
        v = self.Velocity(t, dx, dt)
        return v, v.velocity()

    ## \f$ \hat{B} = \vec{B}/|\vec{B}| \f$ \n
    ## <a href="http://mathworld.wolfram.com/BinormalVector.html">Binormal
    ## vector</a> as a hat() decorated velocity_X_acceleration()
    @hat
    def binormal(self):
        """Binormal is the 'hat'-ed velocity_X_acceleration"""
        return self.velocity_X_acceleration()

    ## \f$ \tau = [\vec{v},\vec{a}, \dot{a}]/\sigma^2  \f$ \n
    ## <a href="http://mathworld.wolfram.com/Torsion.html">Torsion</a> via
    ## euclid.scalar_triple_product()
    @starfunc(operator.__div__)
    def torsion(self, t=None, dx=None, dt=None):
        """torsion: -- it's along story"""
        v = self.velocity()
        a = self.acceleration()
        j = self.Jerk()
        return (
            euclid.scalar_triple_product(v, a, j),
            (self.velocity_X_acceleration())**2
            )

    ## \f$ \kappa = \frac{|\vec{v} \times \vec{a}|}{|\vec{v}|^3} \f$ \n
    ## <a href="http://mathworld.wolfram.com/Curvature.html">The Curvature</a>
    @starfunc(operator.__div__)
    def curvature(self):
        """||V X A||/||V||**3"""
        v = self.velocity()
        a = v.velocity()
        return abs(v^a), abs(v)**3

    ## \f$ \vec{C} = \tau \vec{T} + \kappa \vec{B} \f$ \n
    ## <a href=" http://mathworld.wolfram.com/Centrode.html">The Centrode</a>
    ## in terms of the torsion(), curvature(), tangent() vector and the
    ## binormal() vector;
    def centrode(self):
        return (
            self.torsion()*self.tangent() +
            self.curvature()*self.binormal()
            )

    ## The Darboux vector is the centrode
    Darboux = centrode

    ## \f$ \sigma = 1/\tau \f$ \n
    ## <a href="http://mathworld.wolfram.com/RadiusofTorsion.html">Radius of
    ## torsion</a> as a radius_of() decorated torsion().
    @radius_of
    def radius_of_torsion(self):
        """radius_of() torsion"""
        return self.torsion()

    ## \f$ \rho^2 = 1/ |v \times a |^2 \f$ \n <a href="http://mathworld.wolfram.com/RadiusofCurvature.html">Radius of curvature</a> as a radius_of() decorated curvature()
    @radius_of
    def radius_of_curvature(self):
        """radius_of() curvature"""
        return self.curvature()

    ## \f$ s = \int_{\gamma} ds = \int_{\gamma} |\dot{\vec{r}}| \f$\n
    ## <a href="http://mathworld.wolfram.com/ArcLength.html">Arc Length</a>
    ## (computed for fixed time steps).
    def arc_length(self, axis=None):
        """TODO: use scipy to integrate to make a nested function...."""
        return self.speed().cumsum(axis=axis)

    ## \f$ {\bf T} = \hat{x}\hat{T}+\hat{y}\hat{N} + \hat{z}\hat{B} \f$ \n
    ## <a href="http://mathworld.wolfram.com/Trihedron.html">The Trihedron
    ## Tensor</a>
    def trihedron(self):
        return euclid.ziprows(self.tangent(), self.normal(), self.binormal())

    ## TNB
    TNB = trihedron

    ## \f$ \kappa \f$, read-only curvature()
    @property
    def kappa(self):
        return self.curvature()

    ## \f$ \tau \f$, read-only torsion()
    @property
    def tau(self):
        return self.torsion()

    ## \f$ \vec{T} \f$, read-only tangent()
    @property
    def T(self):
        return self.tangent()

    ## \f$ \vec{B} \f$, read-only normal()
    @property
    def N(self):
        return self.normal()

    ## \f$ \vec{B} \f$ , read-only binormal()
    @property
    def B(self):
        return self.binormal()

    @property
    def Tprime(self):
        return self.kappa*self.N

    @property
    def Nprime(self):
        return -self.kappa*self.T + self.tau*self.B

    @property
    def Bprime(self):
        return -self.tau*self.B

    ## \f$ {\bf \vec{T}'} = {\bf \vec{\omega} \times \vec{T}} \f$ \n
    ## \f$ {\bf \vec{N}'} = {\bf \vec{\omega} \times \vec{N}} \f$ \n
    ## \f$ {\bf \vec{B}'} = {\bf \vec{\omega} \times \vec{B}} \f$ \n
    ## <a href="http://en.wikipedia.org/wiki/Frenet-Serret_formulas">
    ## Frenet-Serret</a> formuale.
    def TNBPrime(self):
        return euclid.ziprows(*map(self.Darboux().cross, self.TNB()))


    ## Matplotlib plotter
    def plot3d(self, f=0):
        import pylab
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d import axes3d
        fig = plt.figure(f)
        ax = Axes3D(fig)
        ax.plot(self.x, self.y, self.z)
        pylab.xlabel("x")
        pylab.ylabel("y")
        pylab.show()
        return ax

    ## Recast as a lowly Vector
    def vector(self):
        return euclid.Vector(self.x, self.y, self.z)

    ## return space curve
    def space_curve(self):
        return self

    ##  Compute tangent plane triplet from SpaceCurve \f${\bf \vec{r}}\f$:\n
    ## \f$ \bf{ \hat{x} } \propto {\bf \hat{r'}} \f$ \n
    ## \f$ {\bf \hat{z}} \propto {\bf \hat{z}} - {\bf (\hat{z}\cdot\hat{x})\hat{x}}  \f$ \n
    ## \f$ \bf{ \hat{y} }= {\bf \hat{z} \times \hat{x} } \f$ \n
    ## (where "prime" is differentiation via prime())\n
    ## keyword z defines body z coordinate.
    def tangent_frame_triplet(self, z=DOWN):
        """i, j, k = r.tangent_frame_triplet([z=DOWN])

        i, j, k for a right handed orthonormal triplet, with:

        i   parallel to r.velocity()
        j   is k^i (cross product)
        k   is keyword z's normalized vector rejection of i
        """
        i = self.tangent().vector()
        k = (z.VectorRejection(i)).hat()
        j = (k^i)

        return i, j, k


    ## Tangent, Level, Up' to Level \n: Rotation connecting
    ## tangent-to-ellipsoid to tangent-to-motion frame \n computed from
    ## itertools.product:\n take all dot product combinations (as numbers,
    ## not euclid.Scalar() ) and make a euclid.Matrix().
    def tlu2level(self):
        return euclid.Matrix(
            *[
                (e_body*e_level).w for e_body, e_level in itertools.product(
                    self.tangent_frame_triplet(z=UP),
                    euclid.BASIS
                    )
                ]
             ).versor()

    ## invert tlu2level()
    def level2tlu(self):
        return ~(self.tlu2level())

    ## compose level2tlu() and tlu2rd()
    def level2trd(self):
        return self.level2tlu.compose(self.tlu2trd())

    ## Compute level frame to body frame-- with keyword defined system
    ## (TLU or TRD)
    def level2body(self, imu, method=level2tlu):
        return (operator.methodcaller(method)(self)).compose(imu)

    ## To Be Debugged
    def level2body_affine(self, imu, method=level2tlu):
        from geo.euclid.affine import Affine
        print ("order not debugged")
        R = self.level2body(imu, method=method)
        T = -(~R)(self)
        return Affine(R, T)
    pass
