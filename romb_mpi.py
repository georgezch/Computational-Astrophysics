#trapParallel_2.py
#example to run: mpiexec -n 4 python26 trapParallel_2.py 0.0 1.0 10000
from mpi4py import MPI
import numpy as np
import time
import sys
from colossus.cosmology import cosmology

import numpy as np
from mpi4py import MPI
import sys

# define the cosmology we are going to work with (kept everything the same as in notes)
my_cosmo = {'flat': True, 'H0': 70.0, 'Om0': 0.27, 'Ob0': 0.045714, 'sigma8': 0.82, 'ns': 0.96}
z = 0.0

# set my_cosmo to be the current cosmology; 
# setCosmology returns a colossus cosmology object with useful functions
cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
# switch off internal colossus spline interpolation
cosmo.interpolation = False

# define the range of wavenumbers k 
kmin = 1.e-4; kmax = 1.e+3; dk = 1.e-2
lkmin = np.log10(kmin); lkmax = np.log10(kmax)

# set up vectors of k with even spacing and log10 spacing (klog)
k = np.arange(kmin, kmax, dk)
lk = np.arange(lkmin, lkmax, 0.01); klog = 10.0**lk

def Wth(x, R):
    # top-hat window function
    xi = R*x
    return 3.*(np.sin(xi)-xi*np.cos(xi)) / xi**3.
def Pw(x):
    # power spectrum from colossus
    return cosmo.matterPowerSpectrum(x, 'eh98')
def f(x, R):
    # integrant of ex 2 in log spacing
    return 10.**(3.*x)*Pw(10.**x)*Wth(10.**x, R)**2. * ( np.log(10) / (2.*np.pi**2.) )

def romb_km(func, a, b, k, m, *args):
    # calculates Rkm
    """
    Parameters:
    --------------------------------
    func - python function object
           function to integrate
    a, b - floats
           integration interval
    k, m - int 
           parameters in R_{k,m}
    args - list
           a list of parameters to pass to func
           parameters must be in order and number expected by func

    Returns:
    ---------------------------------
    Rkm  - (k+1)x(m+1) float matrix
           the R_{k,m} values for k,m=0,1,2,...
    """
    Rkm = np.array( [[0] * (m+1)] * (k+1), float ) # initialize Rkm matrix with zeroes
    h = b - a # we use only the two edge points at first
    Rkm[0,0] = 0.5 * h * ( func(a, *args) + func(b, *args) ) # trapezoidal with just the two edges
    for i in range(1, k+1):
        h *= 0.5 # the new step size is half the previous
        sd = 0. # sum for k>0
        for k in range(1, 2**i, 2): # step 2 to add only the new points in-between the old from previous subdivision
            sd += func( a+k*h, *args ) # the summation term in Rk0
            Rkm[i,0] = 0.5 * Rkm[i-1,0] + h * sd # evaluation of Rk0
        for j in range( 1, i + 1):
            fact = 4.**j
            Rkm[i,j] = ( fact*Rkm[i,j-1] - Rkm[i-1,j-1] ) / ( fact - 1 ) # recurance relation to evalutate Rkm         
    return Rkm


def romberg(func, a, b, rtol, *args):
    # calculates the integral to the required accuracy
    """
    Parameters:
    --------------------------------
    func - python function object
           function to integrate
    a, b - floats
           integration interval
    rtol - float 
           fractional tolerance of the integral estimate
    args - list
           a list of parameters to pass to func
           parameters must be in order and number expected by func

    Returns:
    ---------------------------------
    I    - float
           estimate of the integral for input f, [a,b] and rtol
    m    - int
           number of iterations to achive desired accuracy
    """
    # testing error tolerance
    mtol = 1
    Rkm = romb_km(func, a, b, mtol, mtol, *args)
    while np.abs( Rkm[mtol,mtol] - Rkm[mtol,mtol-1] ) > rtol: 
        mtol += 1
        Rkm = romb_km(func, a, b, mtol, mtol, *args)
    return Rkm[mtol,mtol], mtol

#==================================================================

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# h is the step size the separates each integration interval
h = (kmax-kmin)/size
rtol = 1.e-8; R = 1.e-1

# the interval that each process handles and the error tolerance of each process
local_a = kmin + rank*h
local_b = local_a + h
local_rtol = rtol*size

la = np.log10(local_a)
lb = np.log10(local_b)

#initializing variables
integral = np.zeros(1)
romb_int = np.zeros(1)

t1 = time.time()
# perform local computation. Each process integrates its own interval
integral, mtol = romberg(f, la, lb, local_rtol, R)
# communication
# root node receives results with a collective "reduce"
romb_int = comm.reduce(integral, MPI.SUM)
t2 = time.time()
dt = t2-t1

# root process prints results
if comm.rank == 0:
	print("Parallel Romberg integration in [a,b] = [%.3e, %.3e] gives result = %.9f with rtol = %.3e"%(kmin, kmax, np.sqrt(romb_int), rtol))
	#print("Fractional error compared to exact value of integral = %.3e"%(romb_int/exact_int - 1.))
	print("Time = %.3e"%(dt))

