import os
os.environ["OMP_NUM_THREADS"] = "8"
import ctypes
gomp = ctypes.CDLL("libgomp.so.1")
gomp.omp_set_num_threads(8)

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
    PlummerPotential,
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
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
import copy


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
    plt.ylabel("ρ(r,z=0) ($M_{sun},{kpc}^{-3}$)")

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
def make_halo_potential(ro,vo):
    # Compute Menc(< r) for MWPotential2014 on a logarithmic grid from 0.1 to 300 kpc

    mwp = copy.deepcopy(MWPotential2014)
    mwp[2] *= 1.5
    mwp.turn_physical_on(ro=ro,vo=vo)

    r_grid = np.logspace(np.log10(0.1),np.log10(300),300) * u.kpc 

    menc_MW = np.array([mass(mwp, r, ro=ro, vo=vo, use_physical=True).value for r in r_grid])
    # Find the best-fit (M,a)

    def objective(params):
        # use logM and loga as params instead of M and a so they have a more similar scale
        logM, loga = params

        M = 10**logM * u.Msun
        a = 10**loga * u.kpc

        hq = HernquistPotential(amp=M,a=a)

        menc_HQ = mass(hq, r_grid).value

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

    return hq_component + nfw_component

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
    return hq_df, hq_df.sample(n=N,return_orbit=True)

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

def plot_velocity_dispersion(hq_df, orbits,filename="velocity_dispersion.png"):
    r_samples = orbits.r(use_physical=True)
    N = len(r_samples)
    r_min, r_max = 0.1, 100.0 
    bins = np.logspace(np.log10(r_min),np.log10(r_max),50)
    counts, edges = np.histogram(r_samples, bins=bins)

    r_centres = 0.5*(edges[:-1] + edges[1:])
    # velocity dispersion σ^2(r) = 1/3(<v^2> - <v>^2), but since it's isotropic <v> = 0

    vx = orbits.vx(use_physical=True)
    vy = orbits.vy(use_physical=True)
    vz = orbits.vz(use_physical=True)

    sigma2_sample = np.zeros(len(r_centres))
    for i, (r_lo, r_hi) in enumerate(zip(edges[:-1], edges[1:])): # get the inner and outer boundary of each bin
        mask = (r_samples >= r_lo) & (r_samples < r_hi) # make sure each particle is in the bin
        sigma2_sample[i] = (np.var(vx[mask]) + np.var(vy[mask]) + np.var(vz[mask])) / 3.0

    sigma_sample = np.sqrt(sigma2_sample) # km/s

    # get the analytical velocity dispersion
    r_fine_kpc = np.logspace(np.log10(r_min),np.log10(r_max),200)
    r_fine_internal = r_fine_kpc / 8.0
    sigma_analytic = np.array([hq_df.sigmar(r) for r in r_fine_internal]) 

    plt.title("Velocity-dispersion profile comparison")
    plt.plot(r_fine_kpc,sigma_analytic, label="Hernquist isotropic prediction")
    plt.scatter(r_centres,sigma_sample, color="orange", s=15,label="Sample")
    plt.xlabel("r (kpc)")
    plt.ylabel("$σ_r(r)$ (km/s)")
    plt.xscale("log")
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_beta_vdisp_curves(beta1,beta2,orbits,best_M,best_a,filename="v_disp_beta_comparsion.png"):
    hq_best = HernquistPotential(amp=best_M * u.Msun, a=best_a * u.kpc)
    r_samples = orbits.r(use_physical=True)
    r_min, r_max = 0.1, 100.0 

    r_fine_kpc = np.logspace(np.log10(r_min),np.log10(r_max),200)
    r_fine_internal = r_fine_kpc / 8.0

    hq_beta_neg = constantbetaHernquistdf(pot=hq_best,beta=beta1)
    sigma_beta_neg = np.array([hq_beta_neg.sigmar(r) for r in r_fine_internal])

    hq_beta_pos = constantbetaHernquistdf(pot=hq_best,beta=beta2)
    sigma_beta_pos = np.array([hq_beta_pos.sigmar(r) for r in r_fine_internal])

    hq_beta_zero = constantbetaHernquistdf(pot=hq_best,beta=0)
    sigma_beta_zero = np.array([hq_beta_zero.sigmar(r) for r in r_fine_internal])

    plt.title("Velocity-dispersion profile comparison for constantbetaHernquistdf")
    plt.xlabel("r (kpc)")
    plt.ylabel("$σ_r(r)$ (km/s)")
    plt.xscale("log")
    plt.plot(r_fine_kpc,sigma_beta_pos,label="β = "+str(beta1))
    plt.plot(r_fine_kpc,sigma_beta_zero, label = "β = 0")
    plt.plot(r_fine_kpc,sigma_beta_neg,label="β = "+str(beta2))
    plt.legend()
    plt.savefig(filename)
    plt.close()

# Exercise 5

def integrate_halo_sample_mw(hq_df,mw_interp,ro,vo,N):
    np.random.seed(0)
    stars = hq_df.sample(n=N)
    stars.turn_physical_on(ro=ro, vo=vo)
    r_stars = stars.r(use_physical=True)
    mask = (r_stars > 0.001) & (r_stars < 10000.)
    stars = stars[mask]
    ts = np.linspace(0., 4., 1000) * u.Gyr
    stars.integrate(ts, mw_interp, method="dop853_c")

    return stars

def plot_star_positions(stars,t,filename="star_positions", file_ext=''):
    plt.scatter(stars.x(t, use_physical=True),stars.y(0.0*u.Gyr, use_physical=True),s=3)
    plt.xlabel("x (kpc)")
    plt.ylabel("y (kpc)")
    plt.xlim(-2500,3500)
    plt.ylim(-4000,4500)
    plt.title("Initial positions of star samples (t = "+str(t)+")")
    filepath = filename + file_ext + '.png'
    plt.savefig(filename)
    plt.close()

# Exercise 6

def make_lmc_orbit(amp=1.5e11*u.Msun,a=10*u.kpc):
    lmc = PlummerPotential(amp=amp,b=a)
    o_lmc = Orbit.from_name("LMC")
    return lmc, o_lmc

def integrate_lmc_backward(hq_df,o_lmc,halo_potential):
    friction = ChandrasekharDynamicalFrictionForce(GMs=1.5e11* u.Msun,sigmar=lambda r: hq_df.sigmar(r,use_physical=False))
    ts_bckwd = np.linspace(0,-3,200)*u.Gyr
    o_lmc.integrate(ts_bckwd,[halo_potential,friction],method="dop853_c")

def plot_galactocentric_distance(o_lmc,filename="galactocentric_distance.png"):
    o_lmc.plot(d1='t',d2='r')
    plt.title("LMC galactocentric distance vs. time")
    plt.savefig(filename)
    plt.close()

def make_moving_object_potential(o_lmc,lmc,halo_potential):
    lmc_moving = MovingObjectPotential(o_lmc,pot=lmc)
    # ts_fwd = np.linspace(-3., 0., 2001) * u.Gyr
    # o_lmc.integrate(ts_fwd,[halo_potential,lmc_moving],method="dop853_c")
    return lmc_moving

def plot_unperturbed_star_positions(ro,vo,hq_df,N,mw_interp,filename="stars_unperturbed.png"):
    r_grid_galpy = np.geomspace(0.001, 10000, 101) / ro.value

    np.random.seed(0)
    stars_unperturbed = hq_df.sample(n=N)
    mask = (stars_unperturbed.r(use_physical=False) > r_grid_galpy.min()) & \
        (stars_unperturbed.r(use_physical=False) < r_grid_galpy.max())
    stars_unperturbed = stars_unperturbed[mask]
    stars_unperturbed.turn_physical_on(ro=ro, vo=vo)

    ts_fwd = np.linspace(-3., 0., 21) * u.Gyr
    stars_unperturbed.integrate(ts_fwd, mw_interp, method="dop853_c")

    plt.scatter(stars_unperturbed.x(ts_fwd[-1]), stars_unperturbed.y(ts_fwd[-1]), s=2)
    plt.title("N="+str(N)+" halo star sample, Milky Way only")
    plt.xlabel("x (kpc)")
    plt.ylabel("y (kpc)")
    plt.xlim(-500,500)
    plt.ylim(-500,500)
    plt.savefig(filename)
    plt.close()

def plot_perturbed_star_positions(ro,vo,hq_df,N,mw_interp,lmc_moving,filename="stars_perturbed_moving_lmc.png"):
    r_grid_galpy = np.geomspace(0.001, 10000, 101) / ro.value

    np.random.seed(0)
    stars_perturbed = hq_df.sample(n=N)
    mask = (stars_perturbed.r(use_physical=False) > r_grid_galpy.min()) & \
       (stars_perturbed.r(use_physical=False) < r_grid_galpy.max())
    stars_perturbed = stars_perturbed[mask]
    stars_perturbed.turn_physical_on(ro=ro, vo=vo)

    ts_fwd = np.linspace(0., 3., 2001) * u.Gyr
    stars_perturbed.integrate(ts_fwd, [mw_interp, lmc_moving])

    plt.scatter(stars_perturbed.x(ts_fwd[-1]), stars_perturbed.y(ts_fwd[-1]), s=2)
    plt.title("N="+str(N)+" halo star sample, Milky Way + moving LMC")
    plt.xlabel("x (kpc)")
    plt.ylabel("y (kpc)")
    plt.xlim(-500,500)
    plt.ylim(-500,500)
    plt.savefig(filename)
    plt.close()

# Exercise 7
def plot_inertial_acceleration(o_lmc,ax,ay,az,filename="inertial_acceleration.png"):
    t_galpy = o_lmc.time(use_physical=False)
    plt.plot(t_galpy, ax, label="$a_{MW,x}(t)$")
    plt.plot(t_galpy, ay, label="$a_{MW,y}(t)$")
    plt.plot(t_galpy, az, label="$a_{MW,z}(t)$")
    plt.xlabel("t (internal units)")
    plt.ylabel("acceleration (internal units)")
    plt.title("LMC's gravitational acceleration at the MW centre")
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_full_halo_response(stars_nif,ts_fwd,N_large,filename="full_halo_response.png"):
    plt.scatter(stars_nif.x(ts_fwd[-1]), stars_nif.y(ts_fwd[-1]), s=2)
    plt.title("N="+str(N_large)+" halo star sample, Milky Way + moving LMC + Noninertial frame force")
    plt.xlabel("x (kpc)")
    plt.ylabel("y (kpc)")
    plt.xlim(-500,500)
    plt.ylim(-500,500)
    plt.savefig(filename)
    plt.close()

def integrate_mw(stars_mw,halo_potential,N):
    ts_fwd = np.linspace(-3., 0., 21) * u.Gyr
    stars_mw.integrate(ts_fwd,halo_potential,method="dop853_c")
    return stars_mw

def integrate_full(stars_full,halo_potential,lmc_moving,nif,N):
    ts_fwd = np.linspace(-3., 0., 21) * u.Gyr
    stars_full.integrate(ts_fwd, [halo_potential,lmc_moving,nif], method="dop853_c")
    return stars_full

def get_coords(stars):
    ts_fwd = np.linspace(-3., 0., 21) * u.Gyr
    x = stars.x(ts_fwd).value
    y = stars.y(ts_fwd).value
    z = stars.z(ts_fwd).value
    return x,y,z

def get_velocities(stars):
    ts_fwd = np.linspace(-3., 0., 21) * u.Gyr
    vx = stars.vx(ts_fwd).value
    vy = stars.vy(ts_fwd).value
    vz = stars.vz(ts_fwd).value
    return vx,vy,vz

def get_vr_vt(x,y,z,vx,vy,vz):
        r = np.sqrt(x**2 + y**2 + z**2)
        vr = (x*vx + y*vy + z*vz) / r
        r_unit_x, r_unit_y, r_unit_z = x/r, y/r, z/r
        vt_x = vx - vr*r_unit_x
        vt_y = vy - vr*r_unit_y
        vt_z = vz - vr*r_unit_z
        return vr, vt_x, vt_y, vt_z

def create_modified_density(r_kpc,best_M,best_a):
    rho = (best_M / (2 * np.pi)) * best_a / (r_kpc * (r_kpc + best_a)**3)
    rho_r = rho * r_kpc

    return rho_r

def get_normalized_enclosed_mass(r_kpc,rho_r):
    p_r = 4 * np.pi * rho_r * r_kpc**2
    cumulative = cumulative_trapezoid(p_r, r_kpc, initial=0)
    cdf = cumulative / cumulative[-1]

    return cdf

def interpolate_samples(cdf,r_kpc,N):
    inverse_cdf = interp1d(cdf,r_kpc)

    u_samples = np.random.uniform(0,1,N)
    r_samples = inverse_cdf(u_samples)

    return r_samples

def sample_random_angles(r_samples,N):
    phi = np.random.uniform(0,2*np.pi,N)
    cos_theta = np.random.uniform(-1,1,N)
    sin_theta = np.sqrt(1 - cos_theta**2)

    R_samples = r_samples * sin_theta
    z_samples = r_samples * cos_theta

    return R_samples, z_samples

def make_plottable_lmc_orbit(hq_df,halo_potential,lmc_potential,rhm):
    ts_bckwd = np.linspace(0, -3, 200) * u.Gyr
    o_lmc_plot = Orbit.from_name("LMC")
    friction = ChandrasekharDynamicalFrictionForce(GMs=1.5e11 * u.Msun,sigmar=lambda r: 
                                                   hq_df.sigmar(r, use_physical=False), rhm=rhm, dens=halo_potential)
    o_lmc_plot.integrate(ts_bckwd, [halo_potential, lmc_potential, friction], method = "dop853_c")

    return o_lmc_plot

def main(n=50_000):
    gomp = ctypes.CDLL("libgomp.so.1")
    gomp.omp_set_num_threads(8)
    
    np.random.seed(0)

    ro, vo = 8.0 * u.kpc, 220.0 * u.km / u.s
    R0 = 8.0 * u.kpc

    MWPotential2014.turn_physical_on(ro=ro,vo=vo)

    halo_potential, best_M, best_a = make_halo_potential(ro,vo)
    
    # mw_interp = make_interpolated_potential(ro)

    hq_df, orbits = sample_halo(halo_potential,n)
    
    lmc, o_lmc = make_lmc_orbit()
    integrate_lmc_backward(hq_df,o_lmc,halo_potential) # was mw_interp
    plot_galactocentric_distance(o_lmc)
    
    lmc_moving = make_moving_object_potential(o_lmc,lmc,halo_potential) # was mw_interp
    
    loc_origin = 1e-4
    
    ts_fwd = np.linspace(-3., 0., 21) * u.Gyr
    t_galpy = o_lmc.time(use_physical=False)

    ax = np.array([evaluateRforces(lmc_moving, loc_origin, 0., phi=0., t=t, use_physical=False) for t in t_galpy])
    ay = np.array([evaluatephitorques(lmc_moving, loc_origin, 0., phi=0., t=t, use_physical=False) for t in t_galpy]) / loc_origin
    az = np.array([evaluatezforces(lmc_moving, loc_origin, 0., phi=0., t=t, use_physical=False) for t in t_galpy])
    
    time_unit = conversion.time_in_Gyr(vo=vo.value, ro=ro.value)
    ts_fwd_galpy = ts_fwd.value / time_unit  # convert Gyr -> galpy units

    ax_int = lambda t: np.interp(t,t_galpy,ax)
    ay_int = lambda t: np.interp(t,t_galpy,ay)
    az_int = lambda t: np.interp(t,t_galpy,az)
    
    nif = NonInertialFrameForce(a0=[ax_int,ay_int,az_int])
    ts_bckwd = np.linspace(0, -3, 21) * u.Gyr
    o_lmc_plot = make_plottable_lmc_orbit(hq_df,halo_potential, lmc, 10*u.kpc)
    np.save('results/o_lmc_x_plummer', o_lmc_plot.x(ts_bckwd).value[::-1])
    np.save('results/o_lmc_y_plummer', o_lmc_plot.y(ts_bckwd).value[::-1])
    np.save('results/o_lmc_z_plummer', o_lmc_plot.z(ts_bckwd).value[::-1])
    # unbiased sample

    np.random.seed(0)
    stars_mw_sample = hq_df.sample(n=n)
    np.random.seed(0)
    stars_full_sample = hq_df.sample(n=n)

    stars_mw = integrate_mw(stars_mw_sample,halo_potential,n) # was mw_interp
    r_mw_final = (np.sqrt(stars_mw.x(ts_fwd[-1])**2 + stars_mw.y(ts_fwd[-1])**2 + stars_mw.z(ts_fwd[-1])**2)).value

    np.save('results/r_mw_final_plummer', r_mw_final)
    #x_mw, y_mw, z_mw = get_coords(stars_mw)
    pos_mw = get_coords(stars_mw)
    np.save('results/pos_mw_plummer', pos_mw)

    #vx_mw, vy_mw, vz_mw = get_velocities(stars_mw)
    vel_mw_cart = get_velocities(stars_mw)
    np.save('results/vel_mw_cart_plummer', vel_mw_cart)
    
    # vr_mw, vt_x_mw, vt_y_mw, vt_z_mw = get_vr_vt(x_mw,y_mw,z_mw,vx_mw,vy_mw,vz_mw)
    vel_mw_cyl = get_vr_vt(*pos_mw, *vel_mw_cart)
    np.save('results/vel_mw_cyl_plummer', vel_mw_cyl)
    
    stars_full = integrate_full(stars_full_sample,halo_potential,lmc_moving,nif,n) # was mw_interp
    r_full_final = (np.sqrt(stars_full.x(ts_fwd[-1])**2 + stars_full.y(ts_fwd[-1])**2 + stars_full.z(ts_fwd[-1])**2)).value

    np.save('results/r_full_final_plummer', r_full_final)
    
    pos_full = get_coords(stars_full)
    np.save('results/pos_full_plummer', pos_full)

    #vx_full, vy_full, vz_full = get_velocities(stars_full)
    vel_full_cart = get_velocities(stars_full)
    np.save('results/vel_full_cart_plummer', vel_full_cart)

    #vr_full, vt_x_full, vt_y_full, vt_z_full = get_vr_vt(x_full,y_full,z_full,vx_full,vy_full,vz_full)
    vel_full_cyl = get_vr_vt(*pos_full, *vel_full_cart)
    np.save('results/vel_full_cyl_plummer', vel_full_cyl)
    # plot_radial_density_comparison(r_mw_final,r_full_final,n)

    np.save('results/times_plummer', ts_fwd.value)
    
if __name__ == "__main__":
    main()