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

# Exercise 1
def plot_mw_vcirc(R0,ro,vo,filename="mw_vcirc.png"):
    # get each component

    bulge = MWPotential2014[0]
    disk = MWPotential2014[1]
    halo = MWPotential2014[2]

        # get each component's vcirc plot in physical units
    MWPotential2014.plotRotcurve(
        Rrange=[0.1,100],
        ro=ro.value,
        vo=vo.value,
        grid=1001,
        use_physical=True,
        label="Total")

    bulge.plotRotcurve(
        Rrange=[0.1,100],
        ro=ro.value,
        vo=vo.value,
        grid=1001,
        use_physical=True,
        overplot=True,
        label="Bulge")

    disk.plotRotcurve(
        Rrange=[0.1,100],
        ro=ro.value,
        vo=vo.value,
        grid=1001,
        use_physical=True,
        overplot=True,
        label="Disk")
    
    halo.plotRotcurve(
        Rrange=[0.1,100],
        ro=ro.value,
        vo=vo.value,
        grid=1001,
        use_physical=True,
        overplot=True,
        label="Halo")
            
    # find circular velocity at R0 = 8kpc
    V0 = vcirc(MWPotential2014, R0, ro=ro, vo=vo)
    plt.scatter(R0.value, V0, label="$v_C(R_0=8kpc)$",color="violet",zorder=5)

    # 3. Add single Hernquist

    hp = HernquistPotential(
        amp=1e12 * u.Msun,
        a=30 * u.kpc,
        ro=ro, 
        vo=vo)

    hp.plotRotcurve(
        Rrange=[0.1,100],
        ro=ro.value,
        vo=vo.value,
        grid=1001,
        use_physical=True,
        overplot=True,
        label="Single Hernquist potential"
        )

    plt.legend()
    plt.title("Exercise 1.1: MWPotential2014 circular-velocity curve contributions")

    plt.savefig(filename)
    plt.close()

def plot_mw_density(ro,vo,r=np.logspace(-1,2,500),filename="mw_density.png"):
    # get r in galpy units
    r_galpy = r/ro.value

    rho = np.array([evaluateDensities(
        MWPotential2014,
        R,
        0.0,
        ro=ro.value,
        vo=vo.value,
        use_physical=True) 
        for R in r_galpy]) 
        
    plt.loglog(r, rho)

    plt.title("Logarithmic plot of ρ(r) for MWPotential2014 in the equatorial plane")
    plt.xlabel("r (kpc)")
    plt.ylabel("ρ(r,z=0) ($M_\odot\,{kpc}^{-3}$)")

    plt.savefig(filename)
    plt.close()

def plot_mw_menc(ro,vo,r=np.logspace(-1,2,500),filename="mw_menc.png"):

    r_galpy = r/ro.value

    menc = np.array([mass(
    MWPotential2014,
    R,
    ro=ro.value,
    vo=vo.value,
    use_physical=True,)
    for R in r_galpy])
    
    plt.loglog(r, menc)

    plt.title("Logarithmic plot of $M_{enc}(<r)$ for MWPotential2014")
    plt.xlabel("r (kpc)")
    plt.ylabel(r"$M_\mathrm{enc}(<r)\ (M_\odot)$")

    plt.savefig(filename)
    plt.close()

# Exercise 2
def integrate_orbit(ro,vo,orbit,ts=np.linspace(0.0,5.0,10000) * u.Gyr,method="symplec4_c"):
    orbit.turn_physical_on(ro=ro, vo=vo)
    orbit.integrate(ts,MWPotential2014,method=method)

def plot_orbits(orbit,filename1="sun_orbit_Rz.png",filename2="sun_orbit_xy.png"):
    orbit.plot(d1="R",d2="z")
    plt.title("Sun-like Orbit in the meridional plane, 5Gyr")
    plt.savefig(filename1)
    plt.close()

    orbit.plot(d1="x",d2="y")
    plt.title("Sun-like orbit in the (x,y) plane, 5Gyr")
    plt.savefig(filename2)
    plt.close()

def find_fractional_errors(orbit,ts=np.linspace(0.0,5.0,10000) * u.Gyr):
    # find the energy and z angular momentum for all times
    E = orbit.E(ts, pot=MWPotential2014, use_physical=True)
    Lz = orbit.Lz(ts, use_physical=True)

    # get the initial energy and angular momentum
    E0 = E[0]
    Lz0 = Lz[0]

    # find fractional errors at each time
    E_error = np.abs((E - E0) / E0)
    Lz_error = np.abs((Lz - Lz0) / Lz0)

    return E_error, Lz_error

def plot_fractional_errors(E_error, Lz_error, ts = np.linspace(0.0,5.0,10000) * u.Gyr,filename1="fractional_energy_error.png",
                            filename2="fractional_Lz_error.png"):
    plt.semilogy(ts, E_error)
    plt.title("Semilog plot of fractional energy error")
    plt.xlabel("Time (Gyr)")
    plt.ylabel("Fractional error")
    plt.ylim(1e-16,1e-8)
    plt.savefig(filename1)
    plt.close()

    plt.semilogy(ts, Lz_error)
    plt.title("Semilog plot of z angular momentum error")
    plt.xlabel("Time (Gyr)")
    plt.ylabel("Fractional error")
    plt.ylim(1e-16,1e-13)
    plt.savefig(filename2)
    plt.close()

def plot_compare_orbits_vT(vT_multiplier_1,vT_multiplier_2, ts = np.linspace(0.0,5.0,10000) * u.Gyr, R=8.0*u.kpc,
                           vR=-11.1*u.km/u.s,vT=232.24*u.km/u.s,z=0.0208*u.kpc,vz=7.25*u.km/u.s,phi=0.0*u.rad,
                           filename="sun_orbit_vT"):
    o_vTplus = Orbit([R,
                    vR,
                    vT_multiplier_1 * vT,
                    z,
                    vz,
                    phi])

    o_vTplus.integrate(ts,MWPotential2014)

    o_vTminus = Orbit([R,
                  vR,
                  vT_multiplier_2 * vT,
                  z,
                  vz,
                  phi])

    o_vTminus.integrate(ts,MWPotential2014)

    o_vTplus.plot(d1="R", d2="z", overplot=True, label="$v_T$ * "+str(vT_multiplier_1), alpha=0.8)
    o_vTminus.plot(d1="R", d2="z", overplot=True, label="$v_T$ * "+str(vT_multiplier_2), alpha=0.8)
    plt.legend()
    plt.title("Sun-like Orbits in the meridional plane for 5Gyr")
    plt.xlim(4, 15)
    plt.ylim(-0.15, 0.15)
    plt.xlabel("R (kpc)")
    plt.ylabel("z (kpc)")

    plt.savefig(filename)
    plt.close()

# Exercise 3 
def make_halo_potential(ro,vo,r_grid = np.logspace(np.log10(0.1),np.log10(300),300) * u.kpc):
    # Compute Menc(< r) for MWPotential2014 on a logarithmic grid from 0.1 to 300 kpc

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

    return HernquistPotential(amp=best_M * u.Msun, a=best_a * u.kpc), best_M, best_a

def make_two_component_halo_potential(ro, vo, r_grid=np.logspace(np.log10(0.1),np.log10(300),300) * u.kpc):
    menc_MW = np.array([mass(MWPotential2014, r, ro=ro, vo=vo, use_physical=True) for r in r_grid])

    def objective(params):
        logM_hq, loga_hq, logM_nfw, loga_nfw = params

        # Hernquist (bulge)
        M_hq = 10**logM_hq * u.Msun
        a_hq = 10**loga_hq * u.kpc
        hq = HernquistPotential(amp=M_hq,a=a_hq)
        menc_hq = mass(hq, r_grid)

        # NFW (Halo)
        M_nfw = 10**logM_nfw * u.Msun
        a_nfw = 10**loga_nfw * u.kpc
        nfw = NFWPotential(amp=M_nfw,a=a_nfw)
        menc_nfw = mass(nfw,r_grid)

        menc_total = menc_hq + menc_nfw

        return np.sum((np.log10(menc_total) - np.log10(menc_MW))**2)

    guess = [12.0, 1.0, 12.5, 1.2]

    best_fit = minimize(objective, guess, bounds=[(10, 12), (0, 1),  
                                                (11, 13), (1, 2.5)]) 

    best_M_hq = 10**best_fit.x[0]
    best_a_hq = 10**best_fit.x[1]
    best_M_nfw = 10**best_fit.x[2]
    best_a_nfw = 10**best_fit.x[3]

    hq_component = HernquistPotential(amp=best_M_hq * u.Msun, a=best_a_hq * u.kpc)
    nfw_component = NFWPotential(amp=best_M_nfw * u.Msun, a=best_a_nfw * u.kpc)

    return hq_component + nfw_component, best_M_hq, best_a_hq, best_M_nfw, best_a_nfw

def plot_residual(pot, ro, vo, r_grid=np.logspace(np.log10(0.1),np.log10(300),300) * u.kpc, filename="residual.png"):
    menc_MW = np.array([mass(MWPotential2014, r, ro=ro, vo=vo, use_physical=True) for r in r_grid])

    menc_pot = mass(pot,r_grid)

    residual = np.log10(menc_pot) - np.log10(menc_MW)

    plt.plot(r_grid,residual)
    plt.title("Residual for given Milky Way approximation")
    plt.xlabel("r (kpc)")
    plt.ylabel(r"$\Delta \log_{10} M$")

    plt.savefig(filename)
    plt.close()

def plot_compare_vcirc(halo_potential, two_component_halo_potential, r_grid=np.logspace(np.log10(0.1),np.log10(300),300) * u.kpc, 
                       filename="vcirc_comparsion.png"):
    single_vcirc = halo_potential.vcirc(r_grid)
    combined_vcirc = two_component_halo_potential.vcirc(r_grid)

    plt.plot(r_grid,single_vcirc,label="Single Hernquist")
    plt.plot(r_grid,combined_vcirc,label="Hernquist and NFW")
    plt.xlabel("$R$ (kpc)")
    plt.ylabel("$v_C(R)$ (km/s)")
    plt.title("Circular velocity of spherical proxies for MWPotential2014")
    plt.legend()
    plt.savefig(filename)
    plt.close()

def make_interpolated_potential(ro):
    r_grid_galpy = np.geomspace(0.001, 10000, 101) / ro.value  # kpc -> internal galpy units
    r_grid_phys = np.geomspace(0.001, 10000, 101) * u.kpc # in kpc

    return interpSphericalPotential(MWPotential2014, r_grid_galpy)

# Exercise 4 
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
    plt.close()

def main(n=1000):
    os.environ["OMP_NUM_THREADS"] = "8"
    #omp = ctypes.CDLL('vcomp140.dll')

    ro, vo = 8.0 * u.kpc, 220.0 * u.km / u.s
    R0 = 8.0 * u.kpc

    MWPotential2014.turn_physical_on(ro=ro,vo=vo)

    """plot_mw_vcirc(R0,ro,vo)
    plot_mw_density(ro,vo)
    plot_mw_menc(ro,vo)

    o_sun = Orbit()
    integrate_orbit(ro,vo,o_sun)
    plot_orbits(o_sun)
    E_error, Lz_error = find_fractional_errors(o_sun)
    plot_fractional_errors(E_error,Lz_error)
    plot_compare_orbits_vT(1.2,0.8)"""

    halo_potential, best_M, best_a = make_halo_potential(ro,vo)
    two_component_halo_potential, best_M_hq, best_a_hq, best_M_nfw, best_a_nfw = make_two_component_halo_potential(ro,vo)

    plot_residual(halo_potential,ro,vo,filename="residual_hernquist.png")
    plot_residual(two_component_halo_potential,ro,vo,filename="residual_hernquist_nfw.png")
    plot_compare_vcirc(halo_potential, two_component_halo_potential)

    mw_interp = make_interpolated_potential(ro)

    orbits = sample_halo(halo_potential,n)
    plot_number_density(orbits, best_M, best_a)
    

if __name__ == "__main__":
    main()