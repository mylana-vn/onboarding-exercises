import matplotlib.pyplot as plt
import numpy as np
from galpy.potential import ChandrasekharDynamicalFrictionForce
import astropy.units as u
from galpy.orbit import Orbit
import matplotlib as mpl
from matplotlib.colors import LogNorm, SymLogNorm, CenteredNorm
from matplotlib import animation

BINS = 75
LIMS = 200

def load_results(path_to_results, filename_ext=''):
    x_mw, y_mw, z_mw = np.load(path_to_results + 'pos_mw'+filename_ext+'.npy')
    vx_mw, vy_mw, vz_mw = np.load(path_to_results + 'vel_mw_cart'+filename_ext+'.npy')
    vr_mw, vt_x_mw, vt_y_mw, vt_z_mw = np.load(path_to_results + 'vel_mw_cyl'+filename_ext+'.npy')

    x_full,y_full,z_full = np.load(path_to_results + 'pos_full'+filename_ext+'.npy')
    vx_full, vy_full, vz_full = np.load(path_to_results + 'vel_full_cart'+filename_ext+'.npy')
    vr_full, vt_x_full, vt_y_full, vt_z_full = np.load(path_to_results + 'vel_full_cyl'+filename_ext+'.npy')
    
    r_mw_final = np.load(path_to_results + 'r_mw_final'+filename_ext+'.npy')
    r_full_final = np.load(path_to_results + 'r_full_final'+filename_ext+'.npy')

    o_lmc_x = np.load(path_to_results + 'o_lmc_x'+filename_ext+'.npy')
    o_lmc_y = np.load(path_to_results + 'o_lmc_y'+filename_ext+'.npy')
    o_lmc_z = np.load(path_to_results + 'o_lmc_z'+filename_ext+'.npy')

    return (x_mw.T, y_mw.T, z_mw.T,
            vx_mw.T, vy_mw.T, vz_mw.T, 
            vr_mw.T, vt_x_mw.T, vt_y_mw.T, vt_z_mw.T,
            x_full.T, y_full.T, z_full.T,
            vx_full.T, vy_full.T, vz_full.T, 
            vr_full.T, vt_x_full.T, vt_y_full.T, vt_z_full.T,
            r_mw_final.T, r_full_final.T, o_lmc_x.T, o_lmc_y.T, o_lmc_z.T)

def make_plots(path_to_results='results/', output_path='outputs/plots/', file_ext=''):
    (x_mw_all, y_mw_all, z_mw_all,
    vx_mw_all, vy_mw_all, vz_mw_all, 
    vr_mw_all, vt_x_mw_all, vt_y_mw_all, vt_z_mw_all,
    x_full_all, y_full_all, z_full_all,
    vx_full_all, vy_full_all, vz_full_all, 
    vr_full_all, vt_x_full_all, vt_y_full_all, vt_z_full_all,
    r_mw_final, r_full_final, o_lmc_x, o_lmc_y, o_lmc_z) = load_results(path_to_results, filename_ext=file_ext)

    (x_mw, y_mw, z_mw,
    vx_mw, vy_mw, vz_mw, 
    vr_mw, vt_x_mw, vt_y_mw, vt_z_mw,
    x_full, y_full, z_full,
    vx_full, vy_full, vz_full, 
    vr_full, vt_x_full, vt_y_full, vt_z_full) = (x_mw_all[-1], y_mw_all[-1], z_mw_all[-1],
        vx_mw_all[-1], vy_mw_all[-1], vz_mw_all[-1], 
        vr_mw_all[-1], vt_x_mw_all[-1], vt_y_mw_all[-1], vt_z_mw_all[-1],
        x_full_all[-1], y_full_all[-1], z_full_all[-1],
        vx_full_all[-1], vy_full_all[-1], vz_full_all[-1], 
        vr_full_all[-1], vt_x_full_all[-1], vt_y_full_all[-1], vt_z_full_all[-1])
    n = len(x_mw)
    print(vx_mw_all.shape, vr_mw_all.shape)
    # unbiased sample

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
    times = np.load('results/times.npy')

    plot_halo_response_slice(x_full_all, y_full_all, z_full_all, o_lmc_y, o_lmc_z, times, tind=-1)
    
def plot_radial_density_comparison(r_mw_final,r_full_final,N,weights_mw=None,weights_full=None,filename="radial_density_comparison_function.png"):
    bins = np.geomspace(0.1,1000,50)
    r_centres = 0.5 * (bins[:-1] + bins[1:])
    volumes = (4.0/3.0) * np.pi * (bins[1:]**3 - bins[:-1]**3)

    counts_mw, edges_mw = np.histogram(r_mw_final,bins=bins,weights=weights_mw) 
    counts_full, edges_full = np.histogram(r_full_final,bins=bins,weights=weights_mw)

    n_mw = counts_mw / volumes
    n_full = counts_full / volumes

    fig = plt.figure()
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
    bins_yz = np.linspace(-LIMS,LIMS,BINS)

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
    plt.xlim(-LIMS, LIMS)
    plt.ylim(-LIMS, LIMS)
    plt.savefig(filename1)
    plt.close()

    plt.pcolormesh(y_edges_full, z_edges_full, log_counts_full.T, cmap="Blues",vmin=vmin,vmax=vmax)
    plt.colorbar(label="star count (log)")
    plt.xlabel("y (kpc)")
    plt.ylabel("z (kpc)")
    plt.title("MW + LMC + NIF, y-z slice at |x| < "+str(dx)+" kpc, N="+str(N))
    plt.scatter(lmc_y[0], lmc_z[0],color="orange", s=100, marker='*', zorder=5, label="LMC current position")
    plt.plot(lmc_y,lmc_z,color="orange",linestyle="--",label="LMC orbit")
    plt.legend()
    plt.xlim(-LIMS, LIMS)
    plt.ylim(-LIMS, LIMS)
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

    bins_yz = np.linspace(-LIMS, LIMS, BINS)

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
    bins_yz = np.linspace(-LIMS, LIMS, BINS)

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
    plt.scatter(lmc_y[0], lmc_z[0],color="orange", s=100, marker='*', zorder=5, label="LMC current position")
    plt.plot(lmc_y,lmc_z,color="orange",linestyle="--",label="LMC orbit")
    plt.legend()
    plt.xlim(-LIMS, LIMS)
    plt.ylim(-LIMS, LIMS)
    plt.savefig(filename2)
    plt.close()

def plot_tdisp_slices(dx, sigma_t_mw_2d, sigma_t_full_2d, lmc_y, lmc_z,N,filename1="tdisp_mw_slices_function.png",filename2="tdisp_full_slices.png"):
    bins_yz = np.linspace(-LIMS, LIMS, BINS)
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
    plt.scatter(lmc_y[0], lmc_z[0], color="orange", s=100, marker='*', zorder=5, label="LMC current position")
    plt.plot(lmc_y,lmc_z,color="orange",linestyle="--",label="LMC orbit")
    plt.legend()
    plt.xlim(-LIMS, LIMS)
    plt.ylim(-LIMS, LIMS)
    plt.savefig(filename2)
    plt.close()

def plot_oa_slices(dx, beta_mw_2d, beta_full_2d, lmc_y, lmc_z,N,filename1="oa_mw_slices_function.png", filename2="oa_full_slices_function.png"):
    bins_yz = np.linspace(-LIMS, LIMS, BINS)

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
    plt.scatter(lmc_y[0], lmc_z[0],color="orange", s=100, marker='*', zorder=5, label="LMC current position")
    plt.plot(lmc_y,lmc_z,color="orange",linestyle="--",label="LMC orbit")
    plt.legend()
    plt.xlim(-LIMS, LIMS)
    plt.ylim(-LIMS, LIMS)
    plt.savefig(filename2)
    plt.close()

def plot_density_ratio(dx,x_mw,y_mw,z_mw,
                       x_full,y_full,z_full,
                       lmc_y, lmc_z,N,filename="density_ratio_function.png"):
    bins_yz = np.linspace(-LIMS, LIMS, BINS)

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
    plt.scatter(lmc_y[0], lmc_z[0],color="orange", s=100, marker='*', zorder=5, label="LMC current position")
    plt.plot(lmc_y,lmc_z,color="orange",linestyle="--",label="LMC orbit")
    plt.title("Perturbed to unperturbed density ratio at |x| < "+str(dx)+" kpc, N="+str(N))
    plt.xlim(-LIMS, LIMS)
    plt.ylim(-LIMS, LIMS)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_radial_dispersion_ratio(dx,sigma_r_mw_2d, sigma_r_full_2d,lmc_y, lmc_z,N,filename="radial_dispersion_ratio_function.png"):
    bins_yz = np.linspace(-LIMS, LIMS, BINS)

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
    plt.scatter(lmc_y[0], lmc_z[0],color="orange", s=100, marker='*', zorder=5, label="LMC current position")
    plt.plot(lmc_y,lmc_z,color="orange",linestyle="--",label="LMC orbit")
    plt.title("Perturbed to unperturbed velocity dispersion ratio, |x| <"+str(dx)+" kpc, N="+str(N))
    plt.legend()
    plt.xlim(-LIMS, LIMS)
    plt.ylim(-LIMS, LIMS)
    plt.savefig(filename)
    plt.close()

def plot_tangential_dispersion_ratio(dx,sigma_t_mw_2d,sigma_t_full_2d,lmc_y, lmc_z,N,filename="tangential_dispersion_ratio_function.png"):
    bins_yz = np.linspace(-LIMS, LIMS, BINS)

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
    plt.scatter(lmc_y[0], lmc_z[0],color="orange", s=100, marker='*', zorder=5, label="LMC current position")
    plt.plot(lmc_y,lmc_z,color="orange",linestyle="--",label="LMC orbit")
    plt.title("Perturbed to unperturbed tangential dispersion ratio, |x| <"+str(dx)+" kpc, N="+str(N))
    plt.legend()
    plt.xlim(-LIMS, LIMS)
    plt.ylim(-LIMS, LIMS)
    plt.savefig(filename)
    plt.close()

def plot_density_difference(dx,x_mw,y_mw,z_mw,
                            x_full,y_full,z_full,
                            lmc_y, lmc_z,N,filename="density_difference_function.png"):
    bins_yz = np.linspace(-LIMS, LIMS, BINS)

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
    plt.scatter(lmc_y[0], lmc_z[0],color="orange", s=100, marker='*', zorder=5, label="LMC current position")
    plt.plot(lmc_y,lmc_z,color="orange",linestyle="--",label="LMC orbit")
    plt.title("Perturbed - unperturbed density difference at |x| < "+str(dx)+" kpc, N="+str(N))
    plt.legend()
    plt.xlim(-LIMS, LIMS)
    plt.ylim(-LIMS, LIMS)
    plt.savefig(filename)
    plt.close()

def plot_radial_dispersion_difference(dx,sigma_r_mw_2d, sigma_r_full_2d, lmc_y, lmc_z,N,filename="radial_dispersion_difference_function.png"):
    bins_yz = np.linspace(-LIMS, LIMS, BINS)

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
    plt.scatter(lmc_y[0], lmc_z[0],color="orange", s=100, marker='*', zorder=5, label="LMC current position")
    plt.plot(lmc_y,lmc_z,color="orange",linestyle="--",label="LMC orbit")
    plt.title("Perturbed to unperturbed velocity dispersion difference, |x| <"+str(dx)+" kpc, N="+str(N))
    plt.xlim(-LIMS, LIMS)
    plt.ylim(-LIMS, LIMS)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_tangential_dispersion_difference(dx,sigma_t_mw_2d, sigma_t_full_2d,lmc_y, lmc_z,N,filename="tangential_dispersion_difference_function.png"):
    bins_yz = np.linspace(-LIMS, LIMS, BINS)

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
    plt.scatter(lmc_y[0], lmc_z[0],color="orange", s=100, marker='*', zorder=5, label="LMC current position")
    plt.plot(lmc_y,lmc_z,color="orange",linestyle="--",label="LMC orbit")
    plt.title("Perturbed to unperturbed tangential dispersion difference, |x| <"+str(dx)+" kpc, N="+str(N))
    plt.legend()
    plt.xlim(-LIMS, LIMS)
    plt.ylim(-LIMS, LIMS)
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
    plt.scatter(lmc_x[0], lmc_y[0],color="red", s=80, marker='*', zorder=5, label="LMC current position")
    plt.plot(lmc_x,lmc_y,color="red",linestyle="--",label="LMC orbit",alpha=0.8)
    plt.xlabel("x (kpc)")
    plt.ylabel("y (kpc)")
    plt.title("Unperturbed vs. perturbed displacement field")
    plt.legend()
    plt.savefig(filename)
    plt.xlim(-LIMS, LIMS)
    plt.ylim(-LIMS, LIMS)
    plt.close()

ro = 8.
def make_gif(full_x, full_y, full_z, lmc_x, lmc_y, lmc_z, sat_a, filepath, file_ext='', frames=60, bins=60):
    # (x_mw, y_mw, z_mw,
    #             vx_mw, vy_mw, vz_mw, 
    #             vr_mw, vt_x_mw, vt_y_mw, vt_z_mw,
    #             x_full, y_full, z_full,
    #             vx_full, vy_full, vz_full, 
    #             vr_full, vt_x_full, vt_y_full, vt_z_full,
    #             r_mw_final, r_full_final, o_lmc_x, o_lmc_y, o_lmc_z) = load_results()
    skip = len(full_x) // frames
    extent = (-LIMS, LIMS, -LIMS, LIMS)
    xedges = np.linspace(extent[0], extent[1], bins + 1)
    yedges = np.linspace(extent[2], extent[3], bins + 1)

    H0, _, _ = np.histogram2d(full_x[0], full_y[0], bins=[xedges, yedges])
    norm = LogNorm(vmin=1, vmax=H0.max())

    fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    empty = np.zeros((bins, bins))
    cmap = mpl.colormaps['magma'].copy()
    cmap.set_bad('black')
    im_kw = dict(origin='lower', extent=extent, norm=norm, cmap=cmap, aspect='equal')
    im_xy = ax[0].imshow(empty, **im_kw)
    im_xz = ax[1].imshow(empty, **im_kw)
    im_yz = ax[2].imshow(empty, **im_kw)

    circle_xy = plt.Circle((0, 0), sat_a, color='orange', fill=False, lw=4, label='Scale Radius')
    circle_xz = plt.Circle((0, 0), sat_a, color='orange', fill=False, lw=4)
    circle_yz = plt.Circle((0, 0), sat_a, color='orange', fill=False, lw=4)
    ax[0].add_patch(circle_xy)
    ax[1].add_patch(circle_xz)
    ax[2].add_patch(circle_yz)

    title = ax[1].set_title('', fontsize=30)
    ax[0].set_xlim(extent[:2])
    ax[0].set_ylim(extent[2:])

    ax[0].set_xlabel('x [kpc]', fontsize=25)
    ax[0].set_ylabel('y [kpc]', fontsize=25)
    ax[1].set_xlabel('x [kpc]', fontsize=25)
    ax[1].set_ylabel('z [kpc]', fontsize=25)
    ax[2].set_xlabel('y [kpc]', fontsize=25)
    ax[2].set_ylabel('z [kpc]', fontsize=25)

    def update(frame):
        frame = frame * skip
        x = full_x[frame]
        y = full_y[frame]
        z = full_z[frame]

        x_sat = float(np.ravel(lmc_x[frame])[0])
        y_sat = float(np.ravel(lmc_y[frame])[0])
        z_sat = float(np.ravel(lmc_z[frame])[0])

        H_xy, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
        H_xz, _, _ = np.histogram2d(x, z, bins=[xedges, yedges])
        H_yz, _, _ = np.histogram2d(y, z, bins=[xedges, yedges])

        im_xy.set_data(H_xy.T)
        im_xz.set_data(H_xz.T)
        im_yz.set_data(H_yz.T)

        circle_xy.center = (x_sat, y_sat)
        circle_xz.center = (x_sat, z_sat)
        circle_yz.center = (y_sat, z_sat)

        title.set_text(f't = {frame:.2f}')
        return im_xy, im_xz, im_yz, circle_xy, circle_xz, circle_yz

    ani = animation.FuncAnimation(fig, update, frames=frames)
    ani.save(filepath+'halo_density' + file_ext + '.gif', fps=5)

    xedges = np.linspace(extent[0], extent[1], bins + 1)
    yedges = np.linspace(extent[2], extent[3], bins + 1)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    empty = np.zeros((bins, bins))
    x0 = full_x[0]
    y0 = full_y[0]
    z0 = full_z[0]
    H_xy_0, _, _ = np.histogram2d(x0, y0, bins=[xedges, yedges])
    H_xz_0, _, _ = np.histogram2d(x0, z0, bins=[xedges, yedges])
    H_yz_0, _, _ = np.histogram2d(y0, z0, bins=[xedges, yedges])
    xN, yN, zN = full_x[-1], full_y[-1], full_z[-1]
    H_xyN, _, _ = np.histogram2d(xN, yN, bins=[xedges, yedges])
    H_xzN, _, _ = np.histogram2d(xN, zN, bins=[xedges, yedges])
    H_yzN, _, _ = np.histogram2d(yN, zN, bins=[xedges, yedges])
    halfrange = max(np.abs((H_xyN - H_xy_0)).max(),
                    np.abs((H_xzN - H_xz_0)).max(),
                    np.abs((H_yzN - H_yz_0)).max())

    #norm = CenteredNorm(halfrange=halfrange)
    from matplotlib.colors import SymLogNorm
    norm = SymLogNorm(linthresh=1, vmin=-halfrange, vmax=halfrange, base=10)
    im_kw = dict(origin='lower', extent=extent, norm = norm, cmap='RdBu', aspect='equal')
    im_xy = ax[0].imshow(empty, **im_kw)
    im_xz = ax[1].imshow(empty, **im_kw)
    im_yz = ax[2].imshow(empty, **im_kw)

    circle_xy = plt.Circle((0, 0), sat_a, color='cyan', fill=False, lw=4, label='Scale Radius')
    circle_xz = plt.Circle((0, 0), sat_a, color='cyan', fill=False, lw=4)
    circle_yz = plt.Circle((0, 0), sat_a, color='cyan', fill=False, lw=4)
    ax[0].add_patch(circle_xy)
    ax[1].add_patch(circle_xz)
    ax[2].add_patch(circle_yz)

    title = ax[1].set_title('', fontsize=30)
    ax[0].set_xlim(extent[:2])
    ax[0].set_ylim(extent[2:])

    ax[0].set_xlabel('x [kpc]', fontsize=25)
    ax[0].set_ylabel('y [kpc]', fontsize=25)
    ax[1].set_xlabel('x [kpc]', fontsize=25)
    ax[1].set_ylabel('z [kpc]', fontsize=25)
    ax[2].set_xlabel('y [kpc]', fontsize=25)
    ax[2].set_ylabel('z [kpc]', fontsize=25)

    def update(frame):
        frame = frame * skip
        x = full_x[frame]
        y = full_y[frame]
        z = full_z[frame]

        x_sat = float(np.ravel(lmc_x[frame])[0])
        y_sat = float(np.ravel(lmc_y[frame])[0])
        z_sat = float(np.ravel(lmc_z[frame])[0])

        H_xy, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
        H_xz, _, _ = np.histogram2d(x, z, bins=[xedges, yedges])
        H_yz, _, _ = np.histogram2d(y, z, bins=[xedges, yedges])
        im_xy.set_data((H_xy - H_xy_0).T)
        im_xz.set_data((H_xz - H_xz_0).T)
        im_yz.set_data((H_yz - H_yz_0).T)

        circle_xy.center = (x_sat, y_sat)
        circle_xz.center = (x_sat, z_sat)
        circle_yz.center = (y_sat, z_sat)

        title.set_text(f't = {frame:.2f}')
        return im_xy, im_xz, im_yz, circle_xy, circle_xz, circle_yz

    ani = animation.FuncAnimation(fig, update, frames=frames)
    ani.save(filepath+'halo_density_contrast' + file_ext + '.gif', fps=5)

def make_density_profile_gif(full_x, full_y, full_z, lmc_x, lmc_y, lmc_z, halo_pot, filepath, eps, r_min=0.5, r_max=300.0,
                             nbins=40, skip=2, fps=5):
    '''
    Animate the spherically-averaged radial density profile of the halo as a
    function of time. The analytic target profile (the Hernquist fit used to
    generate the ICs) is drawn as an opaque reference line in the background.
    '''
    import astropy.units as u

    # log-spaced spherical shells
    r_edges = np.geomspace(r_min, r_max, nbins + 1)
    r_cent = np.sqrt(r_edges[:-1] * r_edges[1:])
    shell_vol = 4.0 / 3.0 * np.pi * (r_edges[1:]**3 - r_edges[:-1]**3)

    def density_at(frame):
        r = np.sqrt(np.ravel(full_x[frame])**2
                    + np.ravel(full_y[frame])**2
                    + np.ravel(full_z[frame])**2)
        m_shell, _ = np.histogram(r, bins=r_edges, weights=masses)
        return m_shell / shell_vol  # Msun / kpc^3

    # analytic target density profile. galpy returns physical density as a
    # bare ndarray in Msun/pc^3 (no astropy unit attached, even with
    # turn_physical_on), so attach the unit explicitly and convert to the
    # Msun/kpc^3 used by the particle-based profile above.
    target = np.asarray(halo_pot.dens(r_cent * u.kpc, 0), dtype=float)
    target = (target * u.Msun / u.pc**3).to(u.Msun / u.kpc**3).value

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(r_cent, target, color='gray', lw=4, alpha=1.0, zorder=1,
            label='target (Hernquist)')
    line, = ax.plot([], [], color='tab:red', lw=2.5, marker='o', ms=4,
                    zorder=2, label='simulation')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('r [kpc]', fontsize=20)
    ax.set_ylabel(r'$\rho(r)$  [$\mathrm{M}_\odot\,\mathrm{kpc}^{-3}$]', fontsize=20)
    ax.set_xlim(r_min, r_max)
    #ax.axhline(sim.halo.mass[0], lw=1, c='k', alpha=0.25, label='halo particle mass')
    ax.axvline(eps, lw=1, c='k', alpha=0.25, label='smoothing length')
    pos_target = target[target > 0]
    ax.set_ylim(pos_target.min() * 0.3, pos_target.max() * 3.0)
    title = ax.set_title('', fontsize=22)
    ax.legend(fontsize=14)

    def update(frame):
        frame = frame * skip
        line.set_data(r_cent, density_at(frame))
        title.set_text(f't = {frame:.2f} Gyr')
        return line, title

    ani = animation.FuncAnimation(fig, update, frames=len(full_x) // skip,
                                  blit=False)
    ani.save(filepath + 'halo_density_profile.gif', fps=fps)
    plt.close(fig)


def make_enclosed_mass_gif(full_x, halo_pot, filepath, r_min=0.5, r_max=300.0,
                           nbins=40, skip=2, fps=5):
    '''
    Animate the enclosed-mass profile M(<r) of the halo as a function of time.
    The analytic target (the Hernquist fit used to generate the ICs) is drawn
    as an opaque reference line in the background, and a horizontal line marks
    the mass of a single halo particle (the resolution floor).
    '''
    import astropy.units as u
    from galpy.potential import mass as galpy_mass

    r_grid = np.geomspace(r_min, r_max, nbins)

    def enclosed_mass_at(frame):
        r = np.sqrt(np.ravel(full_x[frame])**2
                    + np.ravel(full_y[frame])**2
                    + np.ravel(full_z[frame])**2)
        order = np.argsort(r)
        r_sorted = r[order]
        m_cum = np.cumsum(masses[order])          # cumulative mass with radius
        # mass enclosed within each plotted radius (counts every particle below
        # it, including any inside r_min)
        k = np.searchsorted(r_sorted, r_grid, side='right')
        return np.where(k > 0, m_cum[np.clip(k - 1, 0, len(m_cum) - 1)], 0.0)

    # analytic target M(<r). galpy returns physical enclosed mass as a bare
    # ndarray in Msun (no astropy unit attached), so use it directly.
    target = np.asarray(galpy_mass(halo_pot, r_grid * u.kpc), dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(r_grid, target, color='gray', lw=4, alpha=1.0, zorder=1,
            label='target (Hernquist)')
    line, = ax.plot([], [], color='tab:red', lw=2.5, marker='o', ms=4,
                    zorder=2, label='simulation')
    ax.axhline(m_particle, lw=1, c='k', alpha=0.25, label='halo particle mass')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('r [kpc]', fontsize=20)
    ax.set_ylabel(r'$M(<r)$  [$\mathrm{M}_\odot$]', fontsize=20)
    ax.set_xlim(r_min, r_max)
    ax.set_ylim(m_particle * 0.3, target.max() * 3.0)
    title = ax.set_title('', fontsize=22)
    ax.legend(fontsize=14)

    def update(frame):
        frame = frame * skip
        line.set_data(r_grid, enclosed_mass_at(frame))
        title.set_text(f't = {frame:.2f} Gyr')
        return line, title

    ani = animation.FuncAnimation(fig, update, frames=len(full_x) // skip,
                                  blit=False)
    ani.save(filepath + 'halo_enclosed_mass_profile.gif', fps=fps)
    plt.close(fig)


def make_kinematics_profile_gifs(full_x, full_y, full_z, full_vx, full_vy, full_vz,  halo_pot, filepath, beta=None, r_min=0.5, r_max=300.0,
                                 nbins=40, skip=2, fps=5):
    '''
    Animate the halo's radial velocity dispersion sigma_r(r), tangential
    velocity dispersion sigma_t(r), and velocity anisotropy
    beta(r) = 1 - sigma_t^2 / (2 sigma_r^2) as functions of time. Each panel
    shows the isotropic-Hernquist target (from the fitted potential) as an
    opaque reference line in the background.

    Conventions: sigma_r^2 = <v_r^2> - <v_r>^2 and sigma_t^2 = <v_theta^2> +
    <v_phi^2> = <|v|^2 - v_r^2>, so an isotropic model has sigma_t = sqrt(2)
    sigma_r and beta = 0.
    '''
    import astropy.units as u
    from galpy.df import isotropicHernquistdf, osipkovmerrittHernquistdf

    r_edges = np.geomspace(r_min, r_max, nbins + 1)
    r_cent = np.sqrt(r_edges[:-1] * r_edges[1:])

    def stats_at(frame):
        x = np.ravel(full_x[frame])
        y = np.ravel(full_y[frame])
        z = np.ravel(full_z[frame])
        vx = np.ravel(full_vx)
        vy = np.ravel(full_vy)
        vz = np.ravel(full_vz)
        r = np.sqrt(x**2 + y**2 + z**2)
        with np.errstate(invalid='ignore', divide='ignore'):
            v_r = (x * vx + y * vy + z * vz) / r          # radial velocity
        v_t2 = np.clip(vx**2 + vy**2 + vz**2 - v_r**2, 0.0, None)  # v_theta^2+v_phi^2

        sig_r = np.full(nbins, np.nan)
        sig_t = np.full(nbins, np.nan)
        beta = np.full(nbins, np.nan)
        which = np.digitize(r, r_edges) - 1
        for i in range(nbins):
            sel = which == i
            if np.count_nonzero(sel) > 1:
                sr2 = np.var(v_r[sel])         # <v_r^2> - <v_r>^2
                st2 = np.mean(v_t2[sel])       # <v_theta^2> + <v_phi^2>
                sig_r[i] = np.sqrt(sr2)
                sig_t[i] = np.sqrt(st2)
                if sr2 > 0:
                    beta[i] = 1.0 - st2 / (2.0 * sr2)
        return sig_r, sig_t, beta

    # isotropic Hernquist targets. galpy returns bare floats in its physical
    # velocity unit (km/s) and does not vectorise over r, so loop.
    if beta==None:
        df = isotropicHernquistdf(pot=halo_pot)
    elif beta=='OM':
        df = osipkovmerrittHernquistdf(pot=halo_pot)
    sig_r_target = np.array([float(df.sigmar(rr * u.kpc)) for rr in r_cent])
    beta_target = np.array([float(df.beta(rr * u.kpc)) for rr in r_cent])
    sig_t_target =  np.sqrt(2 * (1 - beta_target)) * sig_r_target
    
    # robust y-limits from the target plus a few sampled snapshots, so the
    # animated curve does not run off the frame.
    n_t = len(full_x)
    sample_frames = sorted(set([0, n_t // 2, n_t - 1]))
    samples = [stats_at(f) for f in sample_frames]

    def _ylim(idx, target, pad=0.15, floor0=True):
        vals = np.concatenate([target[np.isfinite(target)]] +
                              [s[idx][np.isfinite(s[idx])] for s in samples])
        lo, hi = np.min(vals), np.max(vals)
        span = hi - lo
        lo = 0.0 if floor0 else lo - pad * span
        return lo, hi + pad * span

    panels = [
        dict(idx=0, target=sig_r_target,
             ylabel=r'$\sigma_r$  [$\mathrm{km\,s^{-1}}$]',
             outname='halo_sigma_r_profile.gif', ylim=_ylim(0, sig_r_target)),
        dict(idx=1, target=sig_t_target,
             ylabel=r'$\sigma_t$  [$\mathrm{km\,s^{-1}}$]',
             outname='halo_sigma_t_profile.gif', ylim=_ylim(1, sig_t_target)),
        dict(idx=2, target=beta_target,
             ylabel=r'$\beta(r) = 1 - \sigma_t^2 / 2\sigma_r^2$',
             outname='halo_beta_profile.gif',
             ylim=(-2.0, 1.1)),
    ]

    for p in panels:
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        ax.plot(r_cent, p['target'], color='gray', lw=4, alpha=1.0, zorder=1,
                label='target')
        line, = ax.plot([], [], color='tab:red', lw=2.5, marker='o', ms=4,
                        zorder=2, label='simulation')
        ax.set_xscale('log')
        ax.set_xlabel('r [kpc]', fontsize=20)
        ax.set_ylabel(p['ylabel'], fontsize=20)
        ax.set_xlim(r_min, r_max)
        ax.set_ylim(*p['ylim'])
        title = ax.set_title('', fontsize=22)
        ax.legend(fontsize=14)

        idx = p['idx']

        def update(frame, line=line, title=title, idx=idx):
            frame = frame * skip
            line.set_data(r_cent, stats_at(frame)[idx])
            title.set_text(f't = {frame:.2f} Gyr')
            return line, title

        ani = animation.FuncAnimation(fig, update, frames=n_t // skip, blit=False)
        ani.save(filepath + p['outname'], fps=fps)
        plt.close(fig)


# def make_virial_ratio_plot(sim, filepath, eps=0.01):
#     '''
#     Plot the global virial ratio 2T/|W| of the (isolated) halo versus time.
#     T is the total kinetic energy and W = (1/2) sum_i m_i Phi_i the total
#     gravitational potential energy from the particles' mutual self-gravity
#     (falcON; the 1/2 avoids double-counting pairs). A self-gravitating system
#     in equilibrium satisfies 2T + W = 0, i.e. 2T/|W| = 1, drawn as a dashed
#     reference line. Pass the same eps used for the run so that W reflects the
#     actual (softened) energy of the simulated system.
#     '''
#     virial = np.empty(len(sim.times))
#     for t_idx in range(len(sim.times)):
#         T = np.sum(sim.KE(t_idx))
#         W = 0.5 * np.sum(sim.PE(t_idx, method='falcON', eps=eps))  # < 0
#         virial[t_idx] = 2.0 * T / np.abs(W)

#     fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
#     ax.plot(sim.times, virial, color='tab:red', lw=2.5)
#     ax.axhline(1.0, ls='--', color='gray', lw=1, label='virial equilibrium')
#     ax.set_xlabel('Time [Gyr]', fontsize=20)
#     ax.set_ylabel(r'$2T/|W|$', fontsize=20)
#     ax.legend(fontsize=14)
#     fig.savefig(filepath + 'virial_ratio.png', dpi=200, bbox_inches='tight')
#     plt.close(fig)

def plot_halo_response_slice(full_x, full_y, full_z, lmc_y, lmc_z, times, tind=-1,
                             slice_halfwidth=20.0, bins=100, extent=(-300, 300, -300, 300),
                             filename="halo_response_slice_plummer.png"):
    '''
    Static map of the halo density change in a slab around the y-z plane, between the
    initial snapshot and snapshot tind. The initial slab is centred on x=0; the final
    slab is centred on the median x of that snapshot (|x - x_med| < slice_halfwidth),
    and the final y,z are recentred to remove the halo's bulk COM drift so the map
    shows the internal deformation (wake) rather than the overall translation.

    full_x/full_y/full_z : (n_times, n_stars) time-major arrays
    lmc_y/lmc_z          : 1-D LMC tracks of length n_times, same time ordering as `times`
    '''
    xedges = np.linspace(extent[0], extent[1], bins + 1)
    yedges = np.linspace(extent[2], extent[3], bins + 1)

    # initial snapshot, x slice around 0
    x0 = np.ravel(full_x[0]); y0a = np.ravel(full_y[0]); z0a = np.ravel(full_z[0])
    s0 = np.abs(x0) < slice_halfwidth
    y0 = y0a[s0]; z0 = z0a[s0]
    H0, _, _ = np.histogram2d(y0, z0, bins=[xedges, yedges])

    # final snapshot, x slice centred on the median x, with y/z bulk drift removed
    xN = np.ravel(full_x[tind]); yNa = np.ravel(full_y[tind]); zNa = np.ravel(full_z[tind])
    x_med = np.median(xN)
    sl = (xN > x_med - slice_halfwidth) & (xN < x_med + slice_halfwidth)
    yN = yNa[sl] - (np.median(yNa[sl]) - np.median(y0))
    zN = zNa[sl] - (np.median(zNa[sl]) - np.median(z0))
    HN, _, _ = np.histogram2d(yN, zN, bins=[xedges, yedges])

    dH = HN - H0
    halfrange = max(float(np.abs(dH).max()), 1.0)
    norm = SymLogNorm(linthresh=1, vmin=-halfrange, vmax=halfrange, base=10)
    cmap = plt.get_cmap('RdBu').copy()
    cmap.set_bad(color='k')

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(dH.T, origin='lower', extent=extent, norm=norm, cmap=cmap, aspect='equal')

    # progenitor (LMC) trajectory up to t_final and its final location
    t_final = float(np.asarray(times)[tind])
    sat_mask = np.asarray(times) <= t_final
    sat_y = np.asarray(lmc_y)[sat_mask]
    sat_z = np.asarray(lmc_z)[sat_mask]
    ax.plot(sat_y, sat_z, c='k', lw=3, label='Progenitor orbit')
    ax.scatter(sat_y[-1], sat_z[-1], s=400, marker='*', color='gold',
               edgecolor='k', zorder=5, label='Final location')

    ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
    ax.set_xlabel('y [kpc]', fontsize=20)
    ax.set_ylabel('z [kpc]', fontsize=20)
    ax.set_title(f't = {t_final:.2f} Gyr   ($|x - x_{{med}}| < {slice_halfwidth:.0f}$ kpc)', fontsize=18)
    ax.legend(loc='upper right', fontsize=12)
    fig.colorbar(im, ax=ax, label=r'$\Delta N$ per bin')
    fig.savefig(filename, dpi=200)
    plt.close(fig)

def make_halo_response_slice_gif(full_x, full_y, full_z, lmc_y, lmc_z, times, filepath, file_ext='',
                                 slice_halfwidth=20.0, bins=100, extent=(-300, 300, -300, 300),
                                 skip=1, fps=5):
    '''
    Animated version of plot_halo_response_slice: the halo density change in a slab
    around the y-z plane vs the initial snapshot, as a function of time. At each frame
    the slab is centred on the median x of that snapshot (|x - x_med| < slice_halfwidth)
    and the y,z are recentred to remove the halo's bulk COM drift, so the map shows the
    internal deformation (wake). The LMC trajectory up to the current time and its
    current position are overlaid.

    full_x/full_y/full_z : (n_times, n_stars) time-major arrays
    lmc_y/lmc_z          : 1-D LMC tracks of length n_times, same ordering as `times`
    '''
    xedges = np.linspace(extent[0], extent[1], bins + 1)
    yedges = np.linspace(extent[2], extent[3], bins + 1)
    times = np.asarray(times)
    lmc_y = np.asarray(lmc_y); lmc_z = np.asarray(lmc_z)

    # initial-snapshot reference histogram (x slice around 0)
    x0 = np.ravel(full_x[0]); y0a = np.ravel(full_y[0]); z0a = np.ravel(full_z[0])
    s0 = np.abs(x0) < slice_halfwidth
    y0 = y0a[s0]; z0 = z0a[s0]
    H0, _, _ = np.histogram2d(y0, z0, bins=[xedges, yedges])

    def slice_hist(frame):
        x = np.ravel(full_x[frame]); y = np.ravel(full_y[frame]); z = np.ravel(full_z[frame])
        x_med = np.median(x)
        sl = (x > x_med - slice_halfwidth) & (x < x_med + slice_halfwidth)
        ys = y[sl] - (np.median(y[sl]) - np.median(y0))
        zs = z[sl] - (np.median(z[sl]) - np.median(z0))
        H, _, _ = np.histogram2d(ys, zs, bins=[xedges, yedges])
        return H - H0

    # fix the symmetric colour scale from the final (most perturbed) frame
    dH_ref = slice_hist(len(full_x) - 1)
    halfrange = max(float(np.abs(dH_ref).max()), 1.0)
    norm = SymLogNorm(linthresh=1, vmin=-halfrange, vmax=halfrange, base=10)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
    ax.set_xlabel('y [kpc]', fontsize=20)
    ax.set_ylabel('z [kpc]', fontsize=20)
    title = ax.set_title('', fontsize=24)
    im = ax.imshow(slice_hist(0).T, origin='lower', extent=extent, norm=norm,
                   cmap='RdBu', aspect='equal')
    (traj_line,) = ax.plot([], [], c='w', lw=3)
    sat_marker = ax.scatter([], [], s=400, marker='*', color='w', edgecolor='k', zorder=5)
    fig.colorbar(im, ax=ax, label=r'$\Delta N$ per bin')

    def update(frame):
        fr = frame * skip
        t = times[fr]
        im.set_data(slice_hist(fr).T)
        sat_mask = times <= t
        traj_line.set_data(lmc_y[sat_mask], lmc_z[sat_mask])
        if np.any(sat_mask):
            sat_marker.set_offsets([[lmc_y[sat_mask][-1], lmc_z[sat_mask][-1]]])
        title.set_text(f't = {t:.2f} Gyr')
        return im, traj_line, sat_marker, title

    ani = animation.FuncAnimation(fig, update, frames=len(full_x) // skip, interval=300)
    ani.save(filepath + 'halo_response_slice' + file_ext + '.gif', fps=fps)
    plt.close(fig)

def make_gifs(path_to_results='results/', output_path='outputs/anims/', file_ext=''):
    (x_mw_all, y_mw_all, z_mw_all,
    vx_mw_all, vy_mw_all, vz_mw_all,
    vr_mw_all, vt_x_mw_all, vt_y_mw_all, vt_z_mw_all,
    x_full_all, y_full_all, z_full_all,
    vx_full_all, vy_full_all, vz_full_all,
    vr_full_all, vt_x_full_all, vt_y_full_all, vt_z_full_all,
    r_mw_final, r_full_final, o_lmc_x, o_lmc_y, o_lmc_z) = load_results(path_to_results, filename_ext=file_ext)
    #make_gif(x_full_all, y_full_all, z_full_all, o_lmc_x, o_lmc_y, o_lmc_z, 3, '', frames=20, bins=30)
    # load_results already returns time-major (n_times, n_stars) arrays, so no .T here
    #make_slab_density_change_gif(x_full_all, y_full_all, z_full_all, o_lmc_y, o_lmc_z, '', frames=20, bins=60, thickness=20.0)
    times = np.load(path_to_results + 'times' + file_ext + '.npy')
    make_halo_response_slice_gif(x_full_all, y_full_all, z_full_all, o_lmc_y, o_lmc_z, times, output_path, file_ext, slice_halfwidth=20.0, bins=50)


if __name__ == "__main__":
    make_plots(file_ext='_plummer')
    make_gifs(file_ext='_plummer')