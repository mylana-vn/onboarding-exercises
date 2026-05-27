import matplotlib.pyplot as plt
import numpy as np
from galpy.util import conversion
import astropy.units as u
from galpy.potential import (
    MWPotential2014, 
    vcirc,
    evaluateDensities,
    mass,
    HernquistPotential,
    NFWPotential,
    interpSphericalPotential,
    ChandrasekharDynamicalFrictionForce,
    MovingObjectPotential,
    NonInertialFrameForce,
    evaluateRforces,
    evaluatezforces,
    evaluatephitorques
    )
from galpy.orbit import Orbit
from galpy.df import (
    isotropicHernquistdf,
    constantbetaHernquistdf)
from scipy.optimize import minimize
import os
import ctypes
import time

def make_halo_potential(ro,vo):
    # Compute Menc(< r) for MWPotential2014 on a logarithmic grid from 0.1 to 300 kpc

    r_grid = np.logspace(np.log10(0.1),np.log10(300),300) * u.kpc

    menc_MW = np.array([mass(MWPotential2014, r, ro=ro, vo=vo, use_physical=True) for r in r_grid])
    # Find the best-fit (M,a)

    def objective(params):
        # use logM and loga as params instead of M and a so they have a more similar scale
        logM, loga = params

        M = 10**logM * u.Msun
        a = 10**loga * u.kpc

        hq = HernquistPotential(amp=M,a=a)

        menc_HQ = mass(hq, r_grid)

        return np.sum((np.log10(menc_HQ) - np.log10(menc_MW))**2)

    # set an initial guess (M = 1e12Msun, a = 10 kpc)
    guess = [12.0, 1.0]

    best_fit = minimize(objective,guess)

    best_M = 10**best_fit.x[0]
    best_a = 10**best_fit.x[1]

    return HernquistPotential(amp=best_M, a=best_a), best_M, best_a

def sample_halo(pot,N):
    hq_df = isotropicHernquistdf(pot=pot)
    return hq_df.sample(n=N,return_orbit=True)

def plot_number_density(orbits, best_M, best_a, filename="number_density.png"):
    # Sample number density
    # get r for each orbit (kpc)
    r_samples = orbits.r(use_physical=True)
    N = len(r_samples)
    r_min, r_max = 0.1, 100.0 
    bins = np.logspace(np.log10(r_min),np.log10(r_max),50)
    counts, edges = np.histogram(r_samples, bins=bins)

    r_centres = 0.5*(edges[:-1] + edges[1:])
    volumes = (4.0/3.0)*np.pi*(edges[1:]**3 - edges[:-1]**3) # kpc^3

    n_sample = counts / volumes # number of particles per spherical shell (kpc^-3)

    # Hernquist number density
    # formula : rho_Hernquist(r) = M / (2*pi) * a / r(r+a)^3

    M_total = best_M # Msun
    a = best_a # kpc 

    def rho_hern(r,M,a):
        return M / (2.0 * np.pi) * a / (r * (r+a)**3) #Msun / kpc^3

    r_fine = np.logspace(np.log10(r_min), np.log10(r_max), 500)
    rho_fine = rho_hern(r_fine, M_total, a)

    # normalize to match sample number density
    norm = N / np.trapezoid(4.0 * np.pi * r_fine**2 * rho_hern(r_fine, 1.0, a), r_fine)
    rho_norm = norm * rho_hern(r_fine, 1.0, a) # kpc^-3

    plt.plot(r_centres[counts>0], n_sample[counts>0], label="Sample number density")
    plt.plot(r_fine, rho_norm, label="Analytical Hernquist number density")

    plt.xscale("log")
    plt.yscale("log")

    plt.title("Sample number density vs. analytical Hernquist number density")
    plt.xlabel("r (kpc)")
    plt.ylabel("Number density ($1/kpc^3$)")
    plt.legend()

    plt.savefig(filename)
    plt.show()


def main(n=1000):
    os.environ["OMP_NUM_THREADS"] = "8"
    #omp = ctypes.CDLL('vcomp140.dll')

    ro, vo = 8.0 * u.kpc, 220.0 * u.km / u.s
    R0 = 8.0 * u.kpc

    MWPotential2014.turn_physical_on(ro=ro,vo=vo)

    halo_potential, best_M, best_a = make_halo_potential(ro,vo)
    orbits = sample_halo(halo_potential,n)
    plot_number_density(orbits, best_M, best_a)
    

if __name__ == "__main__":
    main()