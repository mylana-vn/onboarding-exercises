import matplotlib.pyplot as plt
import numpy as np
from galpy.potential import ChandrasekharDynamicalFrictionForce
import astropy.units as u
from galpy.orbit import Orbit

def load_results(path_to_results):
    x_mw, y_mw, z_mw = np.load(path_to_results + 'pos_mw.npy')
    vx_mw, vy_mw, vz_mw = np.load(path_to_results + 'vel_mw_cart.npy')
    vr_mw, vt_x_mw, vt_y_mw, vt_z_mw = np.load(path_to_results + 'vel_mw_cyl.npy')

    x_full,y_full,z_full = np.load(path_to_results + 'pos_full.npy')
    vx_full, vy_full, vz_full = np.load(path_to_results + 'vel_full_cart.npy')
    vr_full, vt_x_full, vt_y_full, vt_z_full = np.load(path_to_results + 'vel_full_cyl.npy')
    
    r_mw_final = np.load(path_to_results + 'r_mw_final.npy')
    r_full_final = np.load(path_to_results + 'r_full_final.npy')

    o_lmc_x = np.load(path_to_results + 'o_lmc_x.npy')
    o_lmc_y = np.load(path_to_results + 'o_lmc_y.npy')
    o_lmc_z = np.load(path_to_results + 'o_lmc_z.npy')


    return (x_mw, y_mw, z_mw,
            vx_mw, vy_mw, vz_mw, 
            vr_mw, vt_x_mw, vt_y_mw, vt_z_mw,
            x_full, y_full, z_full,
            vx_full, vy_full, vz_full, 
            vr_full, vt_x_full, vt_y_full, vt_z_full,
            r_mw_final, r_full_final, o_lmc_x, o_lmc_y, o_lmc_z)

def make_plots(path_to_results='results/'):
    (x_mw, y_mw, z_mw,
    vx_mw, vy_mw, vz_mw, 
    vr_mw, vt_x_mw, vt_y_mw, vt_z_mw,
    x_full, y_full, z_full,
    vx_full, vy_full, vz_full, 
    vr_full, vt_x_full, vt_y_full, vt_z_full,
    r_mw_final, r_full_final, o_lmc_x, o_lmc_y, o_lmc_z) = load_results(path_to_results)
    n = len(x_mw)


    plot_radial_density_comparison(r_mw_final,r_full_final,n)
    
    plot_density_slices(20,x_mw, y_mw, z_mw, x_full, y_full, z_full, o_lmc_y, o_lmc_z,n)
    v_disp_mw, v_disp_full, counts_mw, counts_full = plot_vdisp_profile_comparison(vr_mw,vr_full,r_mw_final,r_full_final,n)
    vt_disp_mw, vt_disp_full, counts_t_mw, counts_t_full = plot_tangential_disp_comparison(vt_x_mw,vt_y_mw,vt_z_mw, \
        vt_x_full,vt_y_full,vt_z_full,r_mw_final,r_full_final,n)
    plot_orbital_anisotropy_comparison(v_disp_mw,vt_disp_mw,counts_mw,counts_t_mw,v_disp_full,vt_disp_full,counts_full,counts_t_full,n)

    sigma_r_mw_2d, sigma_r_full_2d, sigma_t_mw_2d, sigma_t_full_2d, beta_mw_2d, beta_full_2d = \
          get_dispersion_slices(20,x_mw,y_mw,z_mw,vr_mw,vt_x_mw,vt_y_mw,vt_z_mw,x_full,y_full,z_full,vr_full,vt_x_full,vt_y_full,vt_z_full)

    plot_vdisp_slices(20,sigma_r_mw_2d,sigma_r_full_2d,o_lmc_y, o_lmc_z,n)
    plot_tdisp_slices(20,sigma_t_mw_2d,sigma_t_full_2d,o_lmc_y, o_lmc_z,n)
    plot_oa_slices(20,beta_mw_2d,beta_full_2d,o_lmc_y, o_lmc_z,n)

    plot_density_ratio(20,x_mw, y_mw, z_mw, x_full, y_full, z_full,o_lmc_y, o_lmc_z,n)
    plot_radial_dispersion_ratio(20,sigma_r_mw_2d,sigma_r_full_2d,o_lmc_y, o_lmc_z,n)
    plot_tangential_dispersion_ratio(20,sigma_t_mw_2d,sigma_t_full_2d,o_lmc_y, o_lmc_z,n)
    
    plot_density_difference(20,x_mw, y_mw, z_mw, x_full, y_full, z_full,o_lmc_y, o_lmc_z,n)
    plot_radial_dispersion_difference(20,sigma_r_mw_2d,sigma_r_full_2d,o_lmc_y, o_lmc_z,n)
    plot_tangential_dispersion_difference(20,sigma_t_mw_2d,sigma_t_full_2d,o_lmc_y, o_lmc_z,n)
    

def plot_radial_density_comparison(r_mw_final,r_full_final,N,weights_mw=None,weights_full=None,filename="radial_density_comparison_function.png"):
    bins = np.geomspace(0.1,1000,50)
    r_centres = 0.5 * (bins[:-1] + bins[1:])
    volumes = (4.0/3.0) * np.pi * (bins[1:]**3 - bins[:-1]**3)

    counts_mw, edges_mw = np.histogram(r_mw_final,bins=bins,weights=weights_mw) 
    counts_full, edges_full = np.histogram(r_full_final,bins=bins,weights=weights_mw)

    n_mw = counts_mw / volumes
    n_full = counts_full / volumes

    plt.loglog(r_centres[counts_mw > 0], n_mw[counts_mw > 0], label="MW only")
    plt.loglog(r_centres[counts_full > 0], n_full[counts_full > 0], label="MW + LMC + NIF")
    plt.xlabel("r (kpc)")
    plt.ylabel("number density (kpc$^{-3}$)")
    plt.title("Unperturbed vs perturbed radial density profile, N="+str(N))
    plt.legend()
    plt.savefig(filename)
    plt.close()


def plot_density_slices(dx,x_mw,y_mw,z_mw,
                        x_full,y_full,z_full,
                        lmc_y, lmc_z ,N,filename1="density_slice_mw_function.png",filename2="density_slice_full_function.png"):
    bins_yz = np.linspace(-500,500,30)
    # ts_bckwd = np.linspace(0, -3, 200) * u.Gyr

    # lmc_y = o_lmc_plot.y(ts_bckwd)
    # lmc_z = o_lmc_plot.z(ts_bckwd)

    # Plotting slice of the density in Y-Z at X=0
    # Density slice in Y-Z, X=0

    mask_mw_slice = np.abs(x_mw) < dx
    mask_full_slice = np.abs(x_full) < dx

    counts_mw, y_edges_mw, z_edges_mw = np.histogram2d(y_mw[mask_mw_slice],z_mw[mask_mw_slice],bins=bins_yz)
    counts_full, y_edges_full, z_edges_full = np.histogram2d(y_full[mask_full_slice],z_full[mask_full_slice],bins=bins_yz)

    counts_mw_nonzero = np.where(counts_mw>0, counts_mw, np.nan)
    counts_full_nonzero = np.where(counts_full>0, counts_full, np.nan)

    log_counts_mw = np.log10(counts_mw_nonzero)
    log_counts_full = np.log10(counts_full_nonzero)

    vmin = np.nanmin([log_counts_mw,log_counts_full])
    vmax = np.nanmax([log_counts_mw,log_counts_full])

    plt.pcolormesh(y_edges_mw, z_edges_mw, log_counts_mw.T, cmap="Blues",vmin=vmin,vmax=vmax)
    plt.colorbar(label="star count (log)")
    plt.xlabel("y (kpc)")
    plt.ylabel("z (kpc)")
    plt.title("MW only, y-z slice at |x| < "+str(dx)+" kpc, N="+str(N))

    plt.savefig(filename1)
    plt.close()

    plt.pcolormesh(y_edges_full, z_edges_full, log_counts_full.T, cmap="Blues",vmin=vmin,vmax=vmax)
    plt.colorbar(label="star count (log)")
    plt.xlabel("y (kpc)")
    plt.ylabel("z (kpc)")
    plt.title("MW + LMC + NIF, y-z slice at |x| < "+str(dx)+" kpc, N="+str(N))
    plt.scatter(lmc_y[-1], lmc_z[-1],color="orange", s=100, marker='*', zorder=5, label="LMC current position")
    plt.plot(lmc_y,lmc_z,color="orange",linestyle="--",label="LMC orbit")
    plt.legend()

    plt.savefig(filename2)
    plt.close()



def plot_vdisp_profile_comparison(vr_mw,vr_full,r_mw_final,r_full_final,N,filename="vdisp_comparison_function.png"):
    # vr_mw = vr_mw.value
    # vr_full = vr_full.value
    bins = np.geomspace(0.1,1000,15)
    r_centres = 0.5 * (bins[:-1] + bins[1:])

    v_disp_mw = np.zeros(len(r_centres))
    counts_mw = np.zeros(len(r_centres))
    v_disp_full = np.zeros(len(r_centres))
    counts_full = np.zeros(len(r_centres))

    for i, (r_lo, r_hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask_mw_bin = (r_mw_final >= r_lo) & (r_mw_final < r_hi)
        mask_full_bin = (r_full_final >= r_lo) & (r_full_final < r_hi)

        counts_mw[i] = mask_mw_bin.sum()
        counts_full[i] = mask_full_bin.sum()

        if counts_mw[i] > 1:
            v_disp_mw[i] = np.std(vr_mw[mask_mw_bin])

        if counts_full[i] > 1:
            v_disp_full[i] = np.std(vr_full[mask_full_bin])

    plt.title("Velocity dispersion profile comparison, N="+str(N))
    plt.plot(r_centres[counts_mw > 1],v_disp_mw[counts_mw > 1], label="MW only")
    plt.plot(r_centres[counts_full > 1],v_disp_full[counts_full > 1], color="orange",label="MW+LMC+NIF")
    plt.xlabel("r (kpc)")
    plt.ylabel("$σ_r(r)$ (km/s)")
    plt.xscale("log")
    plt.legend()
    plt.savefig(filename)
    plt.close()

    return v_disp_mw, v_disp_full, counts_mw, counts_full

def plot_tangential_disp_comparison(vt_x_mw,vt_y_mw,vt_z_mw,vt_x_full,vt_y_full,vt_z_full,r_mw_final,r_full_final,N,filename="tangential_disp_comparison_function.png"):
    bins = np.geomspace(0.1,1000,15)
    r_centres = 0.5 * (bins[:-1] + bins[1:])

    vt_disp_mw = np.zeros(len(r_centres))
    counts_t_mw = np.zeros(len(r_centres))
    vt_disp_full = np.zeros(len(r_centres))
    counts_t_full = np.zeros(len(r_centres))

    for i, (r_lo, r_hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask_mw_bin = (r_mw_final >= r_lo) & (r_mw_final < r_hi)
        mask_full_bin = (r_full_final >= r_lo) & (r_full_final < r_hi)

        counts_t_mw[i] = mask_mw_bin.sum()
        counts_t_full[i] = mask_full_bin.sum()

        if counts_t_mw[i] > 1:
            sigma_t2 = np.var(vt_x_mw[mask_mw_bin]) + np.var(vt_y_mw[mask_mw_bin]) + np.var(vt_z_mw[mask_mw_bin])
            vt_disp_mw[i] = np.sqrt(sigma_t2)

        if counts_t_full[i] > 1:
            sigma_t2 = np.var(vt_x_full[mask_full_bin]) + np.var(vt_y_full[mask_full_bin]) + np.var(vt_z_full[mask_full_bin])
            vt_disp_full[i] = np.sqrt(sigma_t2)

    plt.title("Tangential dispersion profile comparison, N="+str(N))
    plt.plot(r_centres[counts_t_mw > 1],vt_disp_mw[counts_t_mw > 1], label="MW only")
    plt.plot(r_centres[counts_t_full > 1],vt_disp_full[counts_t_full > 1], color="orange",label="MW+LMC+NIF")
    plt.xlabel("r (kpc)")
    plt.ylabel("$σ_t(r)$ (km/s)")
    plt.xscale("log")
    plt.legend()
    plt.savefig(filename)
    plt.close()

    return vt_disp_mw, vt_disp_full, counts_t_mw, counts_t_full

def plot_orbital_anisotropy_comparison(v_disp_mw, vt_disp_mw,
                                       counts_mw, counts_t_mw,
                                       v_disp_full, vt_disp_full,
                                       counts_full, counts_t_full,
                                       N, filename="orbital_anisotropy_comparison_function.png"):
    bins = np.geomspace(0.1,1000,15)
    r_centres = 0.5 * (bins[:-1] + bins[1:])

    beta_mw = 1 - (vt_disp_mw[counts_t_mw > 5]**2 / (2 * (v_disp_mw[counts_mw > 5])**2))
    beta_full = 1 - (vt_disp_full[counts_t_full > 5]**2 / (2 * (v_disp_full[counts_full > 5])**2))

    plt.semilogx(r_centres[counts_t_mw > 5], beta_mw, label = "MW")
    plt.semilogx(r_centres[counts_t_full > 5], beta_full, color = "orange", label="MW+LMC+NIF")
    plt.xlabel("r (kpc)")
    plt.ylabel("β(r)")
    plt.ylim(-0.5,1)
    plt.title("Orbital anisotropy profile comparison, N="+str(N))
    plt.legend()
    plt.savefig(filename)
    plt.close()

def get_dispersion_slices(dx,x_mw,y_mw,z_mw,vr_mw,vt_x_mw,vt_y_mw,vt_z_mw,x_full,y_full,z_full,vr_full,vt_x_full,vt_y_full,vt_z_full):

    mask_mw_slice = np.abs(x_mw) < dx
    mask_full_slice = np.abs(x_full) < dx

    bins_yz = np.linspace(-500, 500, 30)

    def dispersion_2d_radial(y, z, v, bins):
        # get bin indices for each star
        y_idx = np.digitize(y, bins) - 1
        z_idx = np.digitize(z, bins) - 1
        n_bins = len(bins) - 1
        sigma = np.full((n_bins, n_bins), np.nan)
        
        for i in range(n_bins):
            for j in range(n_bins):
                mask = (y_idx == j) & (z_idx == i)
                if mask.sum() > 1:
                    sigma[i, j] = np.std(v[mask])
        return sigma
    
    def dispersion_2d_tangential(y, z, vx, vy, vz, bins):
        y_idx = np.digitize(y, bins) - 1
        z_idx = np.digitize(z, bins) - 1
        n_bins = len(bins) - 1
        sigma = np.full((n_bins, n_bins), np.nan)

        for i in range(n_bins):
            for j in range(n_bins):
                mask = (y_idx == j) & (z_idx == i)
                if mask.sum() > 1:
                    sigma_2 = np.var(vx[mask]) + np.var(vy[mask]) + np.var(vz[mask])
                    sigma[i, j] = np.sqrt(sigma_2)
        
        return sigma

    sigma_r_mw_2d = dispersion_2d_radial(y_mw[mask_mw_slice], z_mw[mask_mw_slice], vr_mw[mask_mw_slice], bins_yz)
    sigma_r_full_2d = dispersion_2d_radial(y_full[mask_full_slice], z_full[mask_full_slice], vr_full[mask_full_slice], bins_yz)

    sigma_t_mw_2d = dispersion_2d_tangential(y_mw[mask_mw_slice], z_mw[mask_mw_slice], vt_x_mw[mask_mw_slice], \
                                             vt_y_mw[mask_mw_slice], vt_z_mw[mask_mw_slice], bins_yz)
    sigma_t_full_2d = dispersion_2d_tangential(y_full[mask_full_slice], z_full[mask_full_slice], vt_x_full[mask_full_slice], \
                                               vt_y_full[mask_full_slice], vt_z_full[mask_full_slice], bins_yz)

    beta_mw_2d = 1 - (sigma_t_mw_2d**2) / (2*(sigma_r_mw_2d**2))
    beta_full_2d = 1 - (sigma_t_full_2d**2) / (2*(sigma_r_full_2d**2))

    return sigma_r_mw_2d, sigma_r_full_2d, sigma_t_mw_2d, sigma_t_full_2d, beta_mw_2d, beta_full_2d

def plot_vdisp_slices(dx,sigma_r_mw_2d,sigma_r_full_2d,lmc_y, lmc_z,N,filename1="vdisp_mw_slices_function.png",filename2="vdisp_full_slices_function.png"):
    bins_yz = np.linspace(-500, 500, 30)
    ts_bckwd = np.linspace(0, -3, 200) * u.Gyr

    log_sigma_r_mw_2d = np.where(sigma_r_mw_2d>0,np.log10(sigma_r_mw_2d),np.nan)
    log_sigma_r_full_2d = np.where(sigma_r_full_2d>0,np.log10(sigma_r_full_2d),np.nan)

    vmin = np.nanmin([log_sigma_r_mw_2d,log_sigma_r_full_2d])
    vmax = np.nanmax([log_sigma_r_mw_2d,log_sigma_r_full_2d])

    plt.pcolormesh(bins_yz, bins_yz, log_sigma_r_mw_2d, cmap='Blues',vmin=vmin,vmax=vmax)
    plt.colorbar(label="velocity dispersion (km/s) (log)")
    plt.xlabel("y (kpc)")
    plt.ylabel("z (kpc)")
    plt.title("MW, velocity dispersion at |x| < "+str(dx)+" kpc, N="+str(N))
    plt.savefig(filename1)
    plt.close()

    plt.pcolormesh(bins_yz, bins_yz, log_sigma_r_full_2d, cmap='Blues',vmin=vmin,vmax=vmax)
    plt.colorbar(label="velocity dispersion (km/s) (log)")
    plt.xlabel("y (kpc)")
    plt.ylabel("z (kpc)")
    plt.title("MW+LMC+NIF, velocity dispersion at |x| < "+str(dx)+" kpc, N="+str(N))
    plt.scatter(lmc_y[-1], lmc_z[-1],color="orange", s=100, marker='*', zorder=5, label="LMC current position")
    plt.plot(lmc_y,lmc_z,color="orange",linestyle="--",label="LMC orbit")
    plt.legend()
    plt.savefig(filename2)
    plt.close()

def plot_tdisp_slices(dx, sigma_t_mw_2d, sigma_t_full_2d, lmc_y, lmc_z,N,filename1="tdisp_mw_slices_function.png",filename2="tdisp_full_slices.png"):
    bins_yz = np.linspace(-500, 500, 30)
    ts_bckwd = np.linspace(0, -3, 200) * u.Gyr
    log_sigma_t_mw_2d = np.where(sigma_t_mw_2d>0,np.log10(sigma_t_mw_2d),np.nan)
    log_sigma_t_full_2d = np.where(sigma_t_full_2d>0,np.log10(sigma_t_full_2d),np.nan)

    vmin = np.nanmin([log_sigma_t_mw_2d,log_sigma_t_full_2d])
    vmax = np.nanmax([log_sigma_t_mw_2d,log_sigma_t_full_2d])

    plt.pcolormesh(bins_yz, bins_yz, log_sigma_t_mw_2d, cmap='Blues',vmin=vmin,vmax=vmax)
    plt.colorbar(label="tangential dispersion (log)")
    plt.xlabel("y (kpc)")
    plt.ylabel("z (kpc)")
    plt.title("MW, tangential dispersion at |x| < "+str(dx)+" kpc, N="+str(N))
    plt.savefig(filename1)
    plt.close()

    plt.pcolormesh(bins_yz, bins_yz, log_sigma_t_full_2d, cmap='Blues',vmin=vmin,vmax=vmax)
    plt.colorbar(label="tangential dispersion (log)")
    plt.xlabel("y (kpc)")
    plt.ylabel("z (kpc)")
    plt.title("MW+LMC+NIF, tangential dispersion at |x| < "+str(dx)+" kpc, N="+str(N))
    plt.scatter(lmc_y[-1], lmc_z[-1], color="orange", s=100, marker='*', zorder=5, label="LMC current position")
    plt.plot(lmc_y,lmc_z,color="orange",linestyle="--",label="LMC orbit")
    plt.legend()
    plt.savefig(filename2)
    plt.close()

def plot_oa_slices(dx, beta_mw_2d, beta_full_2d, lmc_y, lmc_z,N,filename1="oa_mw_slices_function.png", filename2="oa_full_slices_function.png"):
    bins_yz = np.linspace(-500, 500, 30)
    ts_bckwd = np.linspace(0, -3, 200) * u.Gyr

    plt.pcolormesh(bins_yz, bins_yz, beta_mw_2d, cmap='Blues',vmin=0,vmax=1)
    plt.colorbar(label="β")
    plt.xlabel("y (kpc)")
    plt.ylabel("z (kpc)")
    plt.title("MW, orbital anisotropy at |x| < "+str(dx)+" kpc, N="+str(N))
    plt.savefig(filename1)
    plt.close()

    plt.pcolormesh(bins_yz, bins_yz, beta_full_2d, cmap='Blues',vmin=0,vmax=1)
    plt.colorbar(label="β")
    plt.xlabel("y (kpc)")
    plt.ylabel("z (kpc)")
    plt.title("MW+LMC+NIF, orbital anisotropy at |x| < "+str(dx)+" kpc, N="+str(N))
    plt.scatter(lmc_y[-1], lmc_z[-1],color="orange", s=100, marker='*', zorder=5, label="LMC current position")
    plt.plot(lmc_y,lmc_z,color="orange",linestyle="--",label="LMC orbit")
    plt.legend()
    plt.savefig(filename2)
    plt.close()

def plot_density_ratio(dx,x_mw,y_mw,z_mw,
                       x_full,y_full,z_full,
                       lmc_y, lmc_z,N,filename="density_ratio_function.png"):
    bins_yz = np.linspace(-500, 500, 30)
    ts_bckwd = np.linspace(0, -3, 200) * u.Gyr


    mask_mw_slice = np.abs(x_mw) < dx
    mask_full_slice = np.abs(x_full) < dx

    counts_mw, y_edges, z_edges = np.histogram2d(y_mw[mask_mw_slice],z_mw[mask_mw_slice],bins=bins_yz)
    counts_full, _, _ = np.histogram2d(y_full[mask_full_slice],z_full[mask_full_slice],bins=bins_yz)

    counts_mw_nonzero = np.where(counts_mw>0, counts_mw, np.nan)
    counts_full_nonzero = np.where(counts_full>0, counts_full, np.nan)

    ratio = np.log10((counts_full_nonzero) / (counts_mw_nonzero))

    max_deviation = np.nanmax(np.abs(ratio))
    vmin = -max_deviation
    vmax = max_deviation

    """display_ratio = np.where((counts_mw > 0) & (counts_full > 0),
                          counts_full / counts_mw,
                    np.where((counts_mw == 0) & (counts_full > 0),
                          vmax,  
                    np.where((counts_mw > 0) & (counts_full == 0),
                          vmin,  
                          np.nan)))"""

    plt.pcolormesh(y_edges, z_edges, ratio.T, cmap="RdBu_r",vmin=vmin,vmax=vmax)
    plt.colorbar(label = "perturbed/unperturbed density ratio (log)")
    plt.xlabel("y (kpc)")
    plt.ylabel("z (kpc)")
    plt.scatter(lmc_y[-1], lmc_z[-1],color="orange", s=100, marker='*', zorder=5, label="LMC current position")
    plt.plot(lmc_y,lmc_z,color="orange",linestyle="--",label="LMC orbit")
    plt.title("Perturbed to unperturbed density ratio at |x| < "+str(dx)+" kpc, N="+str(N))
    plt.savefig(filename)
    plt.close()

def plot_radial_dispersion_ratio(dx,sigma_r_mw_2d, sigma_r_full_2d,lmc_y, lmc_z,N,filename="radial_dispersion_ratio_function.png"):
    bins_yz = np.linspace(-500, 500, 30)


    sigma_mw_nonzero = np.where(sigma_r_mw_2d>0, sigma_r_mw_2d, np.nan)
    sigma_full_nonzero = np.where(sigma_r_full_2d>0, sigma_r_full_2d, np.nan)

    ratio = np.log10(sigma_full_nonzero / sigma_mw_nonzero)

    max_deviation = np.nanmax(np.abs(ratio))
    vmin = -max_deviation
    vmax = max_deviation

    plt.pcolormesh(bins_yz, bins_yz,ratio.T,cmap="RdBu_r",vmin=vmin,vmax=vmax)
    plt.colorbar(label="perturbed/unperturbed $σ_r$ ratio (log)")
    plt.xlabel("y (kpc)")
    plt.ylabel("z (kpc)")
    plt.scatter(lmc_y[-1], lmc_z[-1],color="orange", s=100, marker='*', zorder=5, label="LMC current position")
    plt.plot(lmc_y,lmc_z,color="orange",linestyle="--",label="LMC orbit")
    plt.title("Perturbed to unperturbed velocity dispersion ratio, |x| <"+str(dx)+" kpc, N="+str(N))
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_tangential_dispersion_ratio(dx,sigma_t_mw_2d,sigma_t_full_2d,lmc_y, lmc_z,N,filename="tangential_dispersion_ratio_function.png"):
    bins_yz = np.linspace(-500, 500, 30)

    sigma_mw_nonzero = np.where(sigma_t_mw_2d>0, sigma_t_mw_2d, np.nan)
    sigma_full_nonzero = np.where(sigma_t_full_2d>0, sigma_t_full_2d, np.nan)

    ratio = np.log10(sigma_full_nonzero / sigma_mw_nonzero)

    max_deviation = np.nanmax(np.abs(ratio))
    vmin = -max_deviation
    vmax = max_deviation

    plt.pcolormesh(bins_yz, bins_yz,ratio.T,cmap="RdBu_r",vmin=vmin,vmax=vmax)
    plt.colorbar(label="perturbed/unperturbed $σ_t$ ratio (log)")
    plt.xlabel("y (kpc)")
    plt.ylabel("z (kpc)")
    plt.scatter(lmc_y[-1], lmc_z[-1],color="orange", s=100, marker='*', zorder=5, label="LMC current position")
    plt.plot(lmc_y,lmc_z,color="orange",linestyle="--",label="LMC orbit")
    plt.title("Perturbed to unperturbed tangential dispersion ratio, |x| <"+str(dx)+" kpc, N="+str(N))
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_density_difference(dx,x_mw,y_mw,z_mw,
                            x_full,y_full,z_full,
                            lmc_y, lmc_z,N,filename="density_difference_function.png"):
    bins_yz = np.linspace(-500, 500, 30)


    mask_mw_slice = np.abs(x_mw) < dx
    mask_full_slice = np.abs(x_full) < dx

    counts_mw, y_edges, z_edges = np.histogram2d(y_mw[mask_mw_slice],z_mw[mask_mw_slice],bins=bins_yz)
    counts_full, _, _ = np.histogram2d(y_full[mask_full_slice],z_full[mask_full_slice],bins=bins_yz)

    difference_percent = np.where((counts_full > 1) & (counts_mw > 1),(counts_full - counts_mw) / ((counts_full + counts_mw) / 2) * 100, \
                                np.where((counts_full <= 1) & (counts_mw > 1), -200.0, \
                                np.where((counts_full > 1) & (counts_mw <= 1), 200.0, np.nan)))

    max_deviation = np.nanmax(np.abs(difference_percent))
    vmin = -max_deviation
    vmax = max_deviation

    plt.pcolormesh(y_edges, z_edges, difference_percent.T, cmap="RdBu_r",vmin=vmin,vmax=vmax)
    plt.colorbar(label = "percent difference")
    plt.xlabel("y (kpc)")
    plt.ylabel("z (kpc)")
    plt.scatter(lmc_y[-1], lmc_z[-1],color="orange", s=100, marker='*', zorder=5, label="LMC current position")
    plt.plot(lmc_y,lmc_z,color="orange",linestyle="--",label="LMC orbit")
    plt.title("Perturbed - unperturbed density difference at |x| < "+str(dx)+" kpc, N="+str(N))
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_radial_dispersion_difference(dx,sigma_r_mw_2d, sigma_r_full_2d, lmc_y, lmc_z,N,filename="radial_dispersion_difference_function.png"):
    bins_yz = np.linspace(-500, 500, 30)

    mw_empty = np.isnan(sigma_r_mw_2d)
    full_empty = np.isnan(sigma_r_full_2d)

    difference_percent = np.where(~full_empty & ~mw_empty, \
                                (sigma_r_full_2d - sigma_r_mw_2d) / ((sigma_r_full_2d + sigma_r_mw_2d) / 2) * 100, \
                                np.where(full_empty & ~mw_empty, -200.0, \
                                np.where(~full_empty & mw_empty, 200.0, np.nan)))

    max_deviation = np.nanmax(np.abs(difference_percent))
    vmin = -max_deviation
    vmax = max_deviation

    plt.pcolormesh(bins_yz, bins_yz,difference_percent,cmap="RdBu_r",vmin=vmin,vmax=vmax)
    plt.colorbar(label="perturbed - unperturbed percent difference")
    plt.xlabel("y (kpc)")
    plt.ylabel("z (kpc)")
    plt.scatter(lmc_y[-1], lmc_z[-1],color="orange", s=100, marker='*', zorder=5, label="LMC current position")
    plt.plot(lmc_y,lmc_z,color="orange",linestyle="--",label="LMC orbit")
    plt.title("Perturbed to unperturbed velocity dispersion difference, |x| <"+str(dx)+" kpc, N="+str(N))
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_tangential_dispersion_difference(dx,sigma_t_mw_2d, sigma_t_full_2d,lmc_y, lmc_z,N,filename="tangential_dispersion_difference_function.png"):
    bins_yz = np.linspace(-500, 500, 30)

    mw_empty = np.isnan(sigma_t_mw_2d)
    full_empty = np.isnan(sigma_t_full_2d)

    difference_percent = np.where(~full_empty & ~mw_empty, \
                                (sigma_t_full_2d - sigma_t_mw_2d) / ((sigma_t_full_2d + sigma_t_mw_2d) / 2) * 100, \
                                np.where(full_empty & ~mw_empty, -200.0, \
                                np.where(~full_empty & mw_empty, 200.0, np.nan)))

    max_deviation = np.nanmax(np.abs(difference_percent))
    vmin = -max_deviation
    vmax = max_deviation

    plt.pcolormesh(bins_yz, bins_yz,difference_percent,cmap="RdBu_r",vmin=vmin,vmax=vmax)
    plt.colorbar(label="perturbed - unperturbed percent difference")
    plt.xlabel("y (kpc)")
    plt.ylabel("z (kpc)")
    plt.scatter(lmc_y[-1], lmc_z[-1],color="orange", s=100, marker='*', zorder=5, label="LMC current position")
    plt.plot(lmc_y,lmc_z,color="orange",linestyle="--",label="LMC orbit")
    plt.title("Perturbed to unperturbed tangential dispersion difference, |x| <"+str(dx)+" kpc, N="+str(N))
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_displacement_field(x_mw,y_mw,x_full,y_full,lmc_x, lmc_y,filename="xy_position_field.png"):
    plotted_stars = np.random.choice(len(x_mw), 500, replace=False)

    delta_x = x_full - x_mw
    delta_y = y_full - y_mw
    mag = np.sqrt(delta_x**2 + delta_y**2)

    plt.quiver(x_mw[plotted_stars],y_mw[plotted_stars],delta_x[plotted_stars],delta_y[plotted_stars],mag[plotted_stars],
               cmap="viridis",alpha=0.8)
    plt.colorbar(label="displacement magnitude (kpc)")
    plt.scatter(lmc_x[-1], lmc_y[-1],color="red", s=80, marker='*', zorder=5, label="LMC current position")
    plt.plot(lmc_x,lmc_y,color="red",linestyle="--",label="LMC orbit",alpha=0.8)
    plt.xlabel("x (kpc)")
    plt.ylabel("y (kpc)")
    plt.title("Unperturbed vs. perturbed displacement field")
    plt.legend()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    make_plots()