# ------------------------------------------------
# Name: tranutils.py
# Author: Brian R. Greene and Robert M. Frost
# University of Oklahoma
# Created: 19 April 2023
# Purpose: Functions for calculating turbulence 
# statistics, interpolating to a polar grid, 
# and calculating roll factor.
# ------------------------------------------------

import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar

# ---------------------------------
# Calculate statistics
# --------------------------------- 
def calc_stats_tran(dnc, t0, t1, dt, delta_t):
    """Calculate statistics timeseries using netCDF simulation output from sim2netcdf

    :param str dnc: absolute path to directory for saving output netCDF files
    :param int t0: first timestep for stats to be computed
    :param int t1: final timestep for stats to be computed
    :param int dt: number of timesteps between files to load
    :param float delta_t: dimensional timestep in simulation (seconds)
    """
    # directories and configuration
    timesteps = np.arange(t0, t1+1, dt, dtype=np.int32)
    # determine files to read from timesteps
    fall = [f"{dnc}all_{tt:07d}.nc" for tt in timesteps]
    nf = len(fall)
    # calculate array of times represented by each file
    times = np.array([i*delta_t*dt for i in range(nf)])

    # Load files and clean up
    print("Reading files...")
    dd = xr.open_mfdataset(fall, combine="nested", concat_dim="time")
    dd.coords["time"] = times
    dd.time.attrs["units"] = "s"

    # Calculate statistics
    print("Beginning calculations")
    # create empty dataset that will hold everything
    dd_stat = xr.Dataset()
    # list of base variables
    base = ["u", "v", "w", "theta"]
    # calculate means
    for s in base:
        dd_stat[f"{s}_mean"] = dd[s].mean(dim=("x", "y"))
    # calculate covars
    with ProgressBar():
        # covariances
        dd_stat["uw_cov_res"] = xr.cov(dd.u, dd.w, dim=("x", "y"))
        dd_stat["uw_cov_tot"] = dd_stat.uw_cov_res + dd.txz.mean(dim=("x","y"))
        dd_stat["vw_cov_res"] = xr.cov(dd.v, dd.w, dim=("x", "y"))
        dd_stat["vw_cov_tot"] = dd_stat.vw_cov_res + dd.tyz.mean(dim=("x","y"))
        dd_stat["tw_cov_res"] = xr.cov(dd.theta, dd.w, dim=("x", "y")).compute()
        dd_stat["tw_cov_tot"] = dd_stat.tw_cov_res + dd.q3.mean(dim=("x","y")).compute()
        # variances
        dd_stat["uu_var"] = xr.cov(dd.u, dd.u, dim=("x", "y"))
        dd_stat["vv_var"] = xr.cov(dd.v, dd.v, dim=("x", "y"))
        dd_stat["ww_var"] = xr.cov(dd.w, dd.w, dim=("x", "y"))
        dd_stat["tt_var"] = xr.cov(dd.theta, dd.theta, dim=("x", "y"))
        # ustar
        dd_stat["ustar"] = ((dd_stat.uw_cov_tot**2) + (dd_stat.vw_cov_tot**2))**0.25
        dd_stat["ustar0"] = dd_stat.ustar.isel(z=0).compute()
        # zi
        idx = dd_stat.tw_cov_res.argmin(axis=1)
        dd_stat["zi"] = dd_stat.z[idx]
        # L
        dd_stat["theta_mean"] = dd.theta.mean(dim=("x","y")).compute()
        dd_stat["L"] = -1*(dd_stat.ustar0**3) * dd_stat.theta_mean.isel(z=0) / (.4 * 9.81 * dd_stat.tw_cov_tot.isel(z=0))
        # -zi/L
        dd_stat["zi_L"] = -1*(dd_stat.zi / dd_stat.L)
        # wstar
        dd_stat["wstar"] = ((9.81/dd_stat.theta_mean.isel(z=0))*dd_stat.tw_cov_tot.isel(z=0)*dd_stat.zi)**(1/3)
        # calculate vars
        for s in base[:-1]:
            dd_stat[f"{s}_var"] = dd[s].var(dim=("x", "y"))
        # rotate u_mean and v_mean so <v> = 0
        angle = np.arctan2(dd_stat.v_mean, dd_stat.u_mean)
        dd_stat["alpha"] = angle
        dd_stat["u_mean_rot"] = dd_stat.u_mean*np.cos(angle) + dd_stat.v_mean*np.sin(angle)
        dd_stat["v_mean_rot"] =-dd_stat.u_mean*np.sin(angle) + dd_stat.v_mean*np.cos(angle)
        # rotate instantaneous u and v
        u_rot = dd.u*np.cos(angle) + dd.v*np.sin(angle)
        v_rot =-dd.u*np.sin(angle) + dd.v*np.cos(angle)
        # recalculate u_var_rot, v_var_rot
        dd_stat["u_var_rot"] = u_rot.var(dim=("x", "y"))
        dd_stat["v_var_rot"] = v_rot.var(dim=("x", "y"))

    # Add attributes
    # copy from dd
    dd_stat.attrs = dd.attrs
    dd_stat.attrs["delta"] = (dd.dx * dd.dy * dd.dz) ** (1./3.)

    # Save output file
    fsave = f"{dnc}{t0}_{t1}_stats.nc"
    print(f"Saving file: {fsave}")
    with ProgressBar():
        dd_stat.to_netcdf(fsave, mode="w")
    print("Finished!")
    return

# ---------------------------------
# Convert to polar coords
# --------------------------------- 
def polar_grid(df, dnc, height, Lx, ntbin, nrbin):
    """Interpolate 2D autocorrelation to polar grid
    
    :param Dataset df: 4d (time,x,y,z) xarray Dataset for calculating
    :param str dnc: absolute path to directory for saving output netCDF files
    :param float height: dimensionless height to be rotated
    :param float Lx: Size of horizontal domain in meters
    :param int ntbin: number of angular bins
    :param int nrbin: number of radial bins
    """

    # calculate t'w'
    df["tw_cov_res"] = xr.cov(df.theta, df.w, dim=("x", "y")).compute()
    # calculate zi
    idx = df.tw_cov_res.argmin(axis=1)
    jz = np.zeros(df.time.size)
    for jt in range(0, df.time.size):
        # find jz for defined z/zi
        jz[jt] = abs(df.z/df.z[idx].isel(time=jt) - height).argmin()

    # read in autocorrelation dataset
    R = xr.open_dataset(f"{dnc}R_2d.nc")
    # calculate 2d arrays of theta=theta(x,y), r=r(x,y)
    theta = np.arctan2(R.y, R.x)
    r = (R.x**2. + R.y**2.) ** 0.5
    # grab sizes of x and y dimensions for looping
    nx, ny = R.x.size, R.y.size
    rbin = np.linspace(0, Lx//2, nrbin)
    tbin = np.linspace(-np.pi, np.pi, ntbin)
    # intiialize empty arrays for storing values and counter for normalizing
    wall, count = [np.zeros((R.time.size, ntbin, nrbin), dtype=np.float64) for _ in range(2)]

    print("Rotating to polar coordinates")
    # loop over x, y pairs
    for jx in range(nx):
        for jy in range(ny):
            # find nearest bin center for each r(jx,jy) and theta(jx,jy)
            jr = abs(rbin - r.isel(x=jx,y=jy).values).argmin()
            jt = abs(tbin - theta.isel(x=jx,y=jy).values).argmin()
            for t in range(0, R.time.size):
                w = R.w.isel(time=t, z=jz[t].astype(int))
                # store w[jt,jr] in wall, increment count
                wall[t,jt,jr] += w[jx,jy]
                count[t,jt,jr] += 1
    # set up dimensial array for wmean
    wmean = np.zeros((R.time.size, ntbin, nrbin))
    for t in range(0, R.time.size):
        # normalize wall by count
        wmean[t,:,:] = wall[t,:,:] / count[t,:,:]

    # convert polar Rww to xarray data array
    w_pol = xr.DataArray(data=wmean,
                        coords=dict(time=R.time, theta=tbin, r=rbin),
                        dims=["time", "theta", "r"])
    # output polar Rww data
    w_pol.to_netcdf(f"{dnc}R_pol_zzi{int(height*100)}.nc")
    return

# ---------------------------------
# Calculate roll factor timeseries
# --------------------------------- 
def roll_factor(dnc, height, stats):
    """Calculate roll factor using polar grid autocorrelation of w
    
    :param Dataset stats: output from calc_stats_tran
    :param str dnc: absolute path to directory for saving output netCDF files
    :param float height: dimensionless height
    """

    # read in polar autocorrelation
    w_pol = xr.open_dataarray(f"{dnc}R_pol_zzi{int(height*100)}.nc")
    # set parameters for calculation
    time = w_pol.time
    theta = w_pol.theta
    r = w_pol.r

    # set up dimensional arrays to store roll factor stats
    Rmax_r = np.zeros((time.size, r.size))
    rbin_zi = np.zeros((time.size, r.size))
    RR = np.zeros((time.size))

    print("Calculating roll factor")
    # set up dimensial array for wmean
    wmean = np.zeros((time.size, theta.size, r.size))
    # loop over time
    for jt in range(0, time.size):
        # calculate roll factor
        Rmax_r[jt,:] = np.nanmax(wmean[jt,:,:], axis=0) - np.nanmin(wmean[jt,:,:], axis=0)
        rbin_zi[jt,:] = r / stats.zi.isel(time=jt).values
        RR[jt] = np.nanmax(Rmax_r[jt, rbin_zi[jt,:] >= 0.5])
    print("Roll factor calculation complete!")
    # create xarray data array
    roll = xr.DataArray(data=RR,
                        coords=dict(time=time),
                        dims=["time"])
    
    # save data
    fsave = f"{dnc}rollfactor_zzi{int(height*100)}_raw.nc"
    # output to netCDF
    print(f"Saving file: {fsave}")
    with ProgressBar():
        roll.to_netcdf(fsave, mode="w")
    print("Finished!")
    return

# ---------------------------------
# Calculate integral length scales
# ---------------------------------
def length_scales(dnc, t0, t1, dt, height):
    """Calculate integral length scales along alpha and alpha+pi/2

    :param str dnc: absolute path to directory for saving output netCDF files
    :param int t0: first timestep for stats to be computed
    :param int t1: final timestep for stats to be computed
    :param int dt: number of timesteps between files to load
    :param float height: dimensionless height
    """

    # z/zi value to be plotted
    h = height

    # read in polar autocorrelation
    Rww = xr.open_dataarray(f"{dnc}R_pol_zzi{int(h*100)}.nc")
    # number of theta and r lags
    ntheta = Rww.theta.size
    nr = Rww.r.size

    # import stats
    s = xr.open_dataset(f"{dnc}{t0}_{t1}_stats.nc")
    # number of time points
    ntime = int((t1 - t0) / dt + 1)

    jz = np.zeros(ntime)
    for jt in range(ntime):
        # find jz for defined z/zi
        jz[jt] = abs(s.z/s.zi[jt] - h).argmin()
    # calculate mean wind angle
    alpha = np.zeros(ntime)
    for jt in range(ntime):
        alpha[jt] = np.arctan2(s.v_mean[jt,int(jz[jt])], s.u_mean[jt,int(jz[jt])])
    # find angular lag closest to alpha
    ja, ja2 = [np.zeros(ntime) for _ in range(2)]
    for jt in range(ntime):
        ja[jt] = abs(Rww.theta - alpha[jt]).argmin()
        ja2[jt] = abs(Rww.theta - (alpha[jt] + np.pi/2)).argmin()
    
    # find the indice of the first zero
    tazero, ta2zero = [np.zeros(ntime) for _ in range(2)]
    for jt in range(ntime):
        tazero[jt] = np.where(Rww.isel(time=jt, theta=int(ja[jt])) < 0)[0][0] + 1
        ta2zero[jt] = np.where(Rww.isel(time=jt, theta=int(ja2[jt])) < 0)[0][0] + 1

    # calculate length scale along alpha
    Lwa = np.zeros((Rww.time.size))
    for jt in range(Rww.time.size):
        Lwa[jt] = Rww.isel(time=jt, r=range(0,int(tazero[jt])), theta=int(ja[jt])).integrate("r")

    # calculate length scale along alpha + pi/2
    Lwa2 = np.zeros((Rww.time.size))
    for jt in range(Rww.time.size):
        Lwa2[jt] = Rww.isel(time=jt, r=range(0,int(ta2zero[jt])), theta=int(ja2[jt])).integrate("r")

    # back to xarray
    Lw = xr.Dataset(
        data_vars=dict(
            rolls = (["time"], Lwa),
            normal = (["time"], Lwa2)),
        coords=dict(
            time=Rww.time)
        )
    # save out data
    fsave = f"{dnc}{t0}_{t1}_length_scale.nc"
    Lw.to_netcdf(fsave, mode="w")
    return