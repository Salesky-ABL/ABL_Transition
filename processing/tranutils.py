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
import sys
sys.path.append("/home/rfrost/LES-utils/")
from spec import autocorr_2d
from dask.diagnostics import ProgressBar

# ---------------------------------
# plotting functions
# --------------------------------- 
def rasterize(cf):
    """Rasterize a matplotlib.ContourSet object (contour fill plot)

    :param ContourSet cf: ContourSet object from plt.contourf function
    """
    for collection in cf.collections:
        collection.set_rasterized(True)

# ---------------------------------
# Calculate statistics
# --------------------------------- 
def calc_stats_tran(dnc, t0, t1, dt, delta_t, zi_mode):
    """Calculate statistics timeseries using netCDF simulation output from sim2netcdf

    :param str dnc: absolute path to directory for saving output netCDF files
    :param int t0: first timestep for stats to be computed
    :param int t1: final timestep for stats to be computed
    :param int dt: number of timesteps between files to load
    :param float delta_t: dimensional timestep in simulation (seconds)
    :param float zi_mode: 0 for minima in heat flux, 1 for maxima in vertical potential temperature gradient
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
        dd_stat[s] = dd[s]
    # calculate covars
    with ProgressBar():
        # covariances
        dd_stat["uw_cov_res"] = xr.cov(dd.u, dd.w, dim=("x", "y"))
        dd_stat["uw_cov_tot"] = dd_stat.uw_cov_res + dd.txz.mean(dim=("x","y"))
        dd_stat["vw_cov_res"] = xr.cov(dd.v, dd.w, dim=("x", "y"))
        dd_stat["vw_cov_tot"] = dd_stat.vw_cov_res + dd.tyz.mean(dim=("x","y"))
        dd_stat["tw_cov_res"] = xr.cov(dd.theta, dd.w, dim=("x", "y")).compute()
        dd_stat["tw_cov_tot"] = dd_stat.tw_cov_res + dd.q3.mean(dim=("x","y")).compute()
        dd_stat["q3"] = dd.q3
        # ustar
        dd_stat["ustar"] = ((dd_stat.uw_cov_tot**2) + (dd_stat.vw_cov_tot**2))**0.25
        dd_stat["ustar0"] = dd_stat.ustar.isel(z=0).compute()
        # zi
        if zi_mode == 0:
            idx = dd_stat.tw_cov_res.argmin(axis=1)
            dd_stat["zi"] = dd_stat.z[idx]
        if zi_mode == 1:
            idx = dd_stat.theta_mean.differentiate(coord="z").argmax(axis=1)
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
# Calculate the vorticity terms
# ---------------------------------
def calc_vorticity(dnc, t0, t1, dt, delta_t):
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
    dd_ac = xr.Dataset()
    dd_ls = xr.Dataset()

    with ProgressBar():
        # calculate horizontal vorticity
        dd_stat["zeta1"] = dd.w.differentiate(coord="y") - dd.v.differentiate(coord="z")
        dd_stat["zeta2"] = dd.w.differentiate(coord="x") - dd.u.differentiate(coord="z")
        # calculate vertical vorticity
        dd_stat["zeta3"] = dd.v.differentiate(coord="x") - dd.u.differentiate(coord="y")

        # vorticity averages
        dd_stat["zeta1_abs"] = abs(dd_stat.zeta1)
        dd_stat["zeta1_abs_mean"] = dd_stat.zeta1_abs.mean(dim=("x","y"))
        dd_stat["zeta2_abs"] = abs(dd_stat.zeta2)
        dd_stat["zeta2_abs_mean"] = dd_stat.zeta2_abs.mean(dim=("x","y"))
        dd_stat["zeta3_abs"] = abs(dd_stat.zeta3)
        dd_stat["zeta3_abs_mean"] = dd_stat.zeta3_abs.mean(dim=("x","y"))
        # positive vertical vorticity
        dd_stat["zeta3_pos"] = dd_stat.zeta3.where(dd_stat.zeta3 > 0, drop=True)
        dd_stat["zeta3_pos_mean"] = dd_stat.zeta3_pos.mean(dim=("x","y"))

        # 2d autocorrelation
        R_ds = autocorr_2d("naw", dd_stat, ["zeta1", "zeta2", "zeta3"], timeavg=False, output=False)
        dd_ac["zeta1_autocorr2d"] = R_ds["zeta1"]
        dd_ac["zeta2_autocorr2d"] = R_ds["zeta3"]
        dd_ac["zeta3_autocorr2d"] = R_ds["zeta3"]

        # integral length scales
        Lw_ds_zeta1 = ls_rot(R_ds["zeta1"], t0, t1, dt, read_in=False, output=False)
        dd_ls["ls_zeta1_rolls"] = Lw_ds_zeta1["rolls"]
        dd_ls["ls_zeta1_normal"] = Lw_ds_zeta1["normal"]
        Lw_ds_zeta2 = ls_rot(R_ds["zeta2"], t0, t1, dt, read_in=False, output=False)
        dd_ls["ls_zeta2_rolls"] = Lw_ds_zeta2["rolls"]
        dd_ls["ls_zeta2_normal"] = Lw_ds_zeta2["normal"]
        Lw_ds_zeta3 = ls_rot(R_ds["zeta3"], t0, t1, dt, read_in=False, output=False)
        dd_ls["ls_zeta3_rolls"] = Lw_ds_zeta3["rolls"]
        dd_ls["ls_zeta3_normal"] = Lw_ds_zeta3["normal"]

    # Add attributes
    # copy from dd
    dd_stat.attrs, dd_ac.attrs, dd_ls.attrs = [dd.attrs for _ in range(3)]
    dd_stat.attrs["delta"], dd_ac.attrs["delta"], dd_ls.attrs["delta"] = [(dd.dx * dd.dy * dd.dz) ** (1./3.) for _ in range(3)]

    # save output
    fs_stat = f"{dnc}{t0}_{t1}_vort.nc"
    fs_ac = f"{dnc}{t0}_{t1}_vort_autocorr.nc"
    fs_ls = f"{dnc}{t0}_{t1}_vort_ls.nc"
    with ProgressBar():
        # save vorticity stats
        print(f"Saving file: {fs_stat}")
        dd_stat.to_netcdf(fs_stat, mode="w")
        # save vorticity autocorrelation
        print(f"Saving file: {fs_ac}")
        dd_ac.to_netcdf(fs_ac, mode="w")
        # save vorticity length scales
        print(f"Saving file: {fs_ls}")
        dd_ac.to_netcdf(fs_ls, mode="w")
    print("Finished!")
    
    return
    
# ---------------------------------
# Convert to polar coords
# --------------------------------- 
def polar_grid(df, dnc, heights, Lx, ntbin, nrbin):
    """Interpolate 2D autocorrelation to polar grid
    
    :param Dataset df: 4d (time,x,y,z) xarray Dataset
    :param str dnc: absolute path to directory for saving output netCDF files
    :param float height: dimensionless height to be rotated
    :param float Lx: Size of horizontal domain in meters
    :param int ntbin: number of angular bins
    :param int nrbin: number of radial bins
    """

    # number of heights
    nh, ntime = len(heights), df.time.size

    # calculate t'w'
    df["tw_cov_res"] = xr.cov(df.theta, df.w, dim=("x", "y")).compute()
    # calculate zi
    idx = df.tw_cov_res.argmin(axis=1)
    jz = np.zeros((nh, ntime))
    for jt in range(ntime):
        for jh in range(nh):
            # find jz for defined z/zi
            jz[jh,jt] = abs(df.z/df.z[idx].isel(time=jt) - heights[jh]).argmin()

    # read in autocorrelation dataset
    R = xr.open_dataset(f"{dnc}R_2d.nc")
    # grab sizes of x and y dimensions for looping
    nx, ny = R.x.size, R.y.size

    # calculate 2d arrays of theta=theta(x,y), r=r(x,y)
    theta = np.arctan2(R.y, R.x)
    r = (R.x**2. + R.y**2.) ** 0.5
    # arrays for theta and r bins
    rbin = np.linspace(0, Lx//2, nrbin)
    tbin = np.linspace(-np.pi, np.pi, ntbin)
    # intiialize empty arrays for storing values and counter for normalizing
    wall, count = [np.zeros((ntime, ntbin, nrbin, nh), dtype=np.float64) for _ in range(2)]

    # loop over desired heights
    for jh in range(nh):
        # loop over x, y pairs
        for jx in range(nx):
            for jy in range(ny):
                # find nearest bin center for each r(jx,jy) and theta(jx,jy)
                jr = abs(rbin - r.isel(x=jx,y=jy).values).argmin()
                jt = abs(tbin - theta.isel(x=jx,y=jy).values).argmin()
                for t in range(ntime):
                    w = R.w.isel(time=t, z=jz[jh,t].astype(int))
                    # store w[jt,jr] in wall, increment count
                    wall[t,jt,jr,jh] += w[jx,jy]
                    count[t,jt,jr,jh] += 1

    # set up dimensial array for wmean
    wmean = np.zeros((ntime, ntbin, nrbin, nh))
    for jh in range(nh):
        for t in range(ntime):
            # normalize wall by count
            wmean[t,:,:,jh] = wall[t,:,:,jh] / count[t,:,:,jh]

    # convert polar Rww to xarray data array
    w_pol = xr.DataArray(data=wmean,
                        coords=dict(time=df.time, 
                                    theta=tbin, 
                                    r=rbin, 
                                    height=heights),
                        dims=["time", "theta", "r", "height"])

    # output polar Rww data
    dout = f"{dnc}R_pol.nc"
    print(f"Saving to {dout}")
    w_pol.to_netcdf(dout)
    return

# ---------------------------------
# Calculate roll factor timeseries
# --------------------------------- 
def roll_factor(dnc, df, heights):
    """Calculate roll factor using polar grid autocorrelation of w
    
    :param Dataset stats: output from calc_stats_tran
    :param str dnc: absolute path to directory for saving output netCDF files
    :param float height: dimensionless height
    :param int avg_method: 0 = no average, 1 = rolling average, 2 = coarsen
    :param int avg_time: number of timesteps to average over (6 step default)
    """

    # calculate t'w'
    df["tw_cov_res"] = xr.cov(df.theta, df.w, dim=("x", "y")).compute()
    # calculate zi
    idx = df.tw_cov_res.argmin(axis=1)

    # read in polar autocorrelation
    w_pol = xr.open_dataarray(f"{dnc}R_pol.nc")

    # set parameters for calculation
    time = w_pol.time
    r = w_pol.r
    height = w_pol.height
    # number of dims
    ntime = time.size
    nr = r.size
    nh = height.size

    # set up dimensional arrays to store roll factor stats
    Rmax_r = np.zeros((nh, ntime, nr))
    rbin_zi = np.zeros((nh, ntime, nr))
    RR = np.zeros((nh, ntime))

    print("Calculating roll factor")
    # set up dimensial array for wmean
    wmean = w_pol.values
    # loop over heights
    for jh in range(nh):
        # loop over time
        for jt in range(ntime):
            # calculate roll factor
            Rmax_r[jh,jt,:] = np.nanmax(wmean[jt,:,:,jh], axis=0) - np.nanmin(wmean[jt,:,:,jh], axis=0)
            rbin_zi[jh,jt,:] = r / df.z[idx].isel(time=jt).values
            RR[jh,jt] = np.nanmax(Rmax_r[jh,jt, rbin_zi[jh,jt,:] >= 0.5])
    print("Roll factor calculation complete!")
    # create xarray data array
    roll = xr.DataArray(data=RR,
                        coords=dict(height=height.values,
                                    time=time),
                        dims=["height", "time"])
    
    # save data
    fsave = f"{dnc}rollfactor.nc"

    # output to netCDF
    print(f"Saving file: {fsave}")
    with ProgressBar():
        roll.to_netcdf(fsave, mode="w")
    print("Finished!")
    return

# ---------------------------------
# Calculate integral length scales
# ---------------------------------
def length_scales(dnc, t0, t1, dt, height, avg_method, avg_time=6):
    """Calculate integral length scales normal to and along alpha

    :param str dnc: absolute path to directory for saving output netCDF files
    :param int t0: first timestep for stats to be computed
    :param int t1: final timestep for stats to be computed
    :param int dt: number of timesteps between files to load
    :param float height: dimensionless height
    :param int avg_method: 0 = no average, 1 = rolling average, 2 = coarsen
    :param int avg_time: number of timesteps to average over (6 step default)
    """

    # z/zi value to be plotted
    h = height

    # read in polar autocorrelation
    if avg_method == 0:
        Rww = xr.open_dataarray(f"{dnc}R_pol_zzi{int(h*100)}.nc")
    if avg_method == 1:
        Rww = xr.open_dataarray(f"{dnc}R_pol_zzi{int(h*100)}_rolling{avg_time}.nc")
    if avg_method == 2:
        Rww = xr.open_dataarray(f"{dnc}R_pol_zzi{int(h*100)}_coarsen{avg_time}.nc")

    # fill nans with zeros
    Rww = Rww.fillna(0)

    # import stats
    s = xr.open_dataset(f"{dnc}{t0}_{t1}_stats.nc")
    # number of time points
    ntime = int((t1 - t0) / (dt) + 1) # remove hard coding later

    # array to hold indicies of z/zi
    jz = np.zeros(ntime)
    # loop over time
    for jt in range(ntime):
        # find jz for defined z/zi
        jz[jt] = abs(s.z/s.zi[jt] - h).argmin()

    # array to hold alpha indices
    alpha = np.zeros(ntime)
    # loop over time
    for jt in range(ntime):
        # calculate mean wind angle
        alpha[jt] = np.arctan2(s.v_mean[jt,int(jz[jt])], s.u_mean[jt,int(jz[jt])])

    # arrays to hold indices of angular lags
    ja, ja2 = [np.zeros(ntime) for _ in range(2)]
    # loop over time
    for jt in range(ntime):
        # find angular lag closest to alpha and normal
        ja[jt] = abs(Rww.theta - alpha[jt]).argmin()
        ja2[jt] = abs(Rww.theta - (alpha[jt] + np.pi/2)).argmin()

    # arrays to hold indice of first negative value
    tazero, ta2zero = [np.zeros(ntime) for _ in range(2)]
    # loop over time
    for jt in range(ntime):
        # find the indice of the first zero
        ta_array = np.where(Rww.isel(time=jt, theta=int(ja[jt])) < 0)[0]
        ta2_array = np.where(Rww.isel(time=jt, theta=int(ja2[jt])) < 0)[0]
        # exclude timesteps where all values are positive
        if len(ta_array) > 0:
            tazero[jt] = ta_array[0]
        if len(ta2_array) > 0:
            ta2zero[jt] = ta2_array[0]
    
    # load data into memory
    Rww.load()

    # array for length scale along alpha
    Lwa = np.zeros((Rww.time.size))
    # loop over time
    for jt in range(Rww.time.size):
        # calculate length scale along alpha
        Lwa[jt] = Rww.isel(time=jt, r=range(0,int(tazero[jt])), theta=int(ja[jt])).integrate("r")

    # array for length scale normal to alpha
    Lwa2 = np.zeros((Rww.time.size))
    # loop over time
    for jt in range(Rww.time.size):
        # calculate length scale normal to alpha
        Lwa2[jt] = Rww.isel(time=jt, r=range(0,int(ta2zero[jt])), theta=int(ja2[jt])).integrate("r")

    # back to xarray
    Lw = xr.Dataset(
        data_vars=dict(
            rolls = (["time"], Lwa),
            normal = (["time"], Lwa2)),
        coords=dict(
            time=Rww.time)
        )

    # replace zeroes with NaN
    Lw = Lw.where(Lw != 0, np.nan)

    # save out data
    if avg_method == 0:
        fsave = f"{dnc}{t0}_{t1}_length_scale_zzi{int(h*100)}.nc"
    if avg_method == 1:
        fsave = f"{dnc}{t0}_{t1}_length_scale_zzi{int(h*100)}_rolling{avg_time}.nc"
    if avg_method == 2:
        fsave = f"{dnc}{t0}_{t1}_length_scale_zzi{int(h*100)}_coarsen{avg_time}.nc"
    Lw.to_netcdf(fsave, mode="w")
    return

# ---------------------------------
# Calculate rotated integral length scales
# ---------------------------------
def ls_rot(dnc, t0, t1, dt, read_in=True, output=True):
    """Calculate integral length scales normal to and along alpha
    using rotated cartesian dataset

    :param str dnc: absolute path to directory for saving output 
    netCDF files if read_in=True. autocorr data if False.
    :param int t0: first timestep for stats to be computed
    :param int t1: final timestep for stats to be computed
    :param int dt: number of timesteps between files to load
    """
    
    if read_in:
        # read in rotated autocorrelation
        Rww = xr.open_dataarray(dnc)
    else:
        Rww = dnc
    # remove negative lags
    Rww = Rww.where(Rww.x >= 0, drop=True)
    Rww = Rww.where(Rww.y >= 0, drop=True)

    # number of time points
    ntime = int((t1 - t0) / dt + 1)
    # number of z levels
    nz = int(Rww.z.size)

    # arrays to hold indice of first negative value
    tazero, ta2zero, Lwa, Lwa2 = [np.empty((ntime, nz)) for _ in range(4)]

    # load data into memory
    Rww.load()

    # loop over time and z
    for jt in range(ntime):
        for jz in range(nz):
            # find the indices of the first zero
            ta_array = np.where(Rww[jt,:,0,jz] <= 0)[0]
            ta2_array = np.where(Rww[jt,0,:,jz] <= 0)[0]
            # exclude nans
            if ta_array.size > 0:
                tazero[jt,jz] = ta_array[0]
            else:
                tazero[jt,jz] = np.argmin(Rww[jt,:,0,jz].values)
            if ta2_array.size > 0:
                ta2zero[jt,jz] = ta2_array[0]
            else:
                ta2zero[jt,jz] = np.argmin(Rww[jt,0,:,jz].values)

    # loop over time and z
    for jt in range(ntime):
        for jz in range(nz):
            # calculate length scale along alpha
            Lwa[jt,jz] = Rww.isel(time=jt, y=0, z=jz, x=range(0,int(tazero[jt,jz]))).integrate("x")
            # calculate length scale normal to alpha
            Lwa2[jt,jz] = Rww.isel(time=jt, x=0, z=jz, y=range(0,int(ta2zero[jt,jz]))).integrate("y")

    # back to xarray
    Lw = xr.Dataset(
        data_vars=dict(
            rolls = (["time","z"], Lwa),
            normal = (["time","z"], Lwa2)),
        coords=dict(
            time=Rww.time,
            z=Rww.z)
        )
    
    if output:
        # save out data
        fsave = f"{dnc}{t0}_{t1}_length_scale_rot.nc"
        Lw.to_netcdf(fsave, mode="w")
        return
    else:
        return Lw

# ---------------------------------
# Add new statistic to stats
# ---------------------------------
def add_stat(dnc, t0, t1):
    """Calculate new statistic and add to stats.nc file. Only to be used
    for testing, intentionally not added to runtran.py

    :param str dnc: absolute path to directory for saving/writing netCDF files
    :param int t0: first timestep for stats to be computed
    :param int t1: final timestep for stats to be computed
    :param int dt: number of timesteps between files to load
    """
    # read in current stats file
    fstat = f"{dnc}{t0}_{t1}_stats.nc"
    ds = xr.open_dataset(fstat)

    # calculate vertical vorticity
    ds["zeta"] = ds.v.differentiate(coord="x") - ds.u.differentiate(coord="y")

    fstat_new = f"{dnc}{t0}_{t1}_stats_new.nc"
    print(f"Saving file: {fstat_new}")
    with ProgressBar():
        ds.to_netcdf(f"{fstat_new}", mode="w")
    print("Finished!")