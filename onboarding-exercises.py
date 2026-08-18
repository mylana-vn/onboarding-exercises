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
    mass,
    HernquistPotential,
    PlummerPotential,
    ChandrasekharDynamicalFrictionForce,
    MovingObjectPotential,
    NonInertialFrameForce,
    evaluateRforces,
    evaluatezforces,
    evaluatephitorques
    )
from galpy.orbit import Orbit
from galpy.df import (
    isotropicHernquistdf)
from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
import copy

def make_halo_potential(ro,vo):
    # Compute Menc(< r) for MWPotential2014 on a logarithmic grid from 0.1 to 300 kpc

    mwp = copy.deepcopy(MWPotential2014)
    mwp[2] *= 1.5
    mwp.turn_physical_on(ro=ro,vo=vo)

    r_grid = np.logspace(np.log10(0.1),np.log10(300),300) * u.kpc 

    menc_MW = np.array([mass(mwp[2], r, ro=ro, vo=vo, use_physical=True).value for r in r_grid])
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

def sample_halo(pot,N):
    hq_df = isotropicHernquistdf(pot=pot)
    return hq_df, hq_df.sample(n=N,return_orbit=True)

def make_lmc_orbit(amp=1.5e11*u.Msun,a=10*u.kpc):
    lmc = HernquistPotential(amp=amp,b=a)
    o_lmc = Orbit.from_name("LMC")
    return lmc, o_lmc

def integrate_lmc_backward(hq_df,o_lmc,halo_potential,lmc_potential):
    friction = ChandrasekharDynamicalFrictionForce(GMs=1.5e11* u.Msun,sigmar=lambda r: hq_df.sigmar(r,use_physical=False))
    ts_bckwd = np.linspace(0,-3,2001)*u.Gyr
    o_lmc.integrate(ts_bckwd,[halo_potential,friction,lmc_potential],method="dop853_c")

def make_moving_object_potential(o_lmc,lmc):
    lmc_moving = MovingObjectPotential(o_lmc,pot=lmc)
    return lmc_moving

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
    o_lmc_plot.integrate(ts_bckwd, [halo_potential, friction, lmc_potential], method = "dop853_c")

    return o_lmc_plot

def main(n=250_000):
    gomp = ctypes.CDLL("libgomp.so.1")
    gomp.omp_set_num_threads(8)
    
    np.random.seed(0)

    ro, vo = 8.0 * u.kpc, 220.0 * u.km / u.s

    MWPotential2014.turn_physical_on(ro=ro,vo=vo)

    halo_potential, best_M, best_a = make_halo_potential(ro,vo)
    np.save('results/best_M', best_M)
    np.save('results/best_a', best_a)
    
    # mw_interp = make_interpolated_potential(ro)

    hq_df, orbits = sample_halo(halo_potential,n)
    
    lmc, o_lmc = make_lmc_orbit()
    lmc.turn_physical_on(ro=ro,vo=vo)
    integrate_lmc_backward(hq_df,o_lmc,halo_potential,lmc) # was mw_interp
    
    lmc_moving = make_moving_object_potential(o_lmc,lmc) 
    
    loc_origin = 1e-4
    
    ts_fwd = np.linspace(-3., 0., 200) * u.Gyr
    ts_halo = np.linspace(-3., 0., 21) * u.Gyr
    t_galpy = o_lmc.time(use_physical=False)[::-1]

    ax = np.array([evaluateRforces(lmc_moving, loc_origin, 0., phi=0., t=t, use_physical=False) for t in t_galpy])
    ay = np.array([evaluatephitorques(lmc_moving, loc_origin, 0., phi=0., t=t, use_physical=False) for t in t_galpy]) / loc_origin
    az = np.array([evaluatezforces(lmc_moving, loc_origin, 0., phi=0., t=t, use_physical=False) for t in t_galpy])
    
    time_unit = conversion.time_in_Gyr(vo=vo.value, ro=ro.value)
    ts_fwd_galpy = ts_fwd.value / time_unit  # convert Gyr -> galpy units

    ax_int = lambda t: np.interp(t,t_galpy,ax)
    ay_int = lambda t: np.interp(t,t_galpy,ay)
    az_int = lambda t: np.interp(t,t_galpy,az)
    
    nif = NonInertialFrameForce(a0=[ax_int,ay_int,az_int])
    ts_bckwd = np.linspace(0, -3, 200) * u.Gyr
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

    np.save('results/times_lmc', ts_fwd.value)
    np.save('results/times_halo', ts_halo.value)
    
if __name__ == "__main__":
    main()