#!/home/bgreene/anaconda3/bin/python
# --------------------------------
# Name: LESutils.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 1 Febuary 2023
# Purpose: collection of functions for use in
# processing raw output by Salesky LES code as well as
# calculating commonly-used statistics
# --------------------------------
import os
# import xrft
import xarray as xr
import numpy as np
from numba import njit
from scipy.signal import detrend
from dask.diagnostics import ProgressBar
# --------------------------------
# Begin Defining Functions
# --------------------------------
def read_f90_bin(path, nx, ny, nz, precision):
    """Read raw binary files output by LES fortran code

    :param str path: absolute path to binary file
    :param int nx: number of points in streamwise dimension
    :param int ny: number of points in spanwise dimension
    :param int nz: number of points in wall-normal dimension
    :param int precision: precision of floating-point values (must be 4 or 8)
    :return: dat
    """
    print(f"Reading file: {path}")
    f = open(path, "rb")
    if (precision == 4):
        dat = np.fromfile(f, dtype=np.float32, count=nx*ny*nz)
    elif (precision == 8):
        dat = np.fromfile(f, dtype=np.float64, count=nx*ny*nz)
    else:
        raise ValueError("Precision must be 4 or 8")
    dat = np.reshape(dat, (nx,ny,nz), order="F")
    f.close()

    return dat
# ---------------------------------------------
def sim2netcdf(dout, dnc, resolution, dimensions, scales, t0, t1, dt, 
               use_dissip, simlabel, units=None):
    """Read binary output files from LES code and combine into one netcdf
    file per timestep using xarray for future reading and easier analysis

    :param str dout: absolute path to directory with LES output binary files
    :param str dnc: absolute path to directory for saving output netCDF files
    :param tuple<int> resolution: simulation resolution (nx, ny, nz)
    :param tuple<Quantity> dimensions: simulation dimensions (Lx, Ly, Lz)
    :param tuple<Quantity> scales: dimensional scales from LES code\
        (uscale, Tscale)
    :param int t0: first timestep to process
    :param int t1: last timestep to process
    :param int dt: number of timesteps between files to load
    :param bool use_dissip: flag for loading dissipation rate files (SBL only)
    :param str simlabel: unique identifier for batch of files
    :param dict units: dictionary of units corresponding to each loaded\
        variable. Default values hard-coded in this function
    """
    # check if dnc exists
    if not os.path.exists(dnc):
        os.mkdir(dnc)
    # grab relevent parameters
    nx, ny, nz = resolution
    Lx, Ly, Lz = dimensions
    dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
    u_scale = scales[0]
    theta_scale = scales[1]
    # define timestep array
    timesteps = np.arange(t0, t1+1, dt, dtype=np.int32)
    nt = len(timesteps)
    # dimensions
    x, y = np.linspace(0., Lx, nx), np.linspace(0, Ly, ny)
    # u- and w-nodes are staggered
    # zw = 0:Lz:nz
    # zu = dz/2:Lz-dz/2:nz-1
    # interpolate w, txz, tyz, q3 to u grid
    zw = np.linspace(0., Lz, nz)
    zu = np.linspace(dz/2., Lz+dz/2., nz)
    # --------------------------------
    # Loop over timesteps to load and save new files
    # --------------------------------
    for i in range(nt):
        # load files - DONT FORGET SCALES!
        f1 = f"{dout}u_{timesteps[i]:07d}.out"
        u_in = read_f90_bin(f1,nx,ny,nz,8) * u_scale
        f2 = f"{dout}v_{timesteps[i]:07d}.out"
        v_in = read_f90_bin(f2,nx,ny,nz,8) * u_scale
        f3 = f"{dout}w_{timesteps[i]:07d}.out"
        w_in = read_f90_bin(f3,nx,ny,nz,8) * u_scale
        f4 = f"{dout}theta_{timesteps[i]:07d}.out"
        theta_in = read_f90_bin(f4,nx,ny,nz,8) * theta_scale
        f5 = f"{dout}txz_{timesteps[i]:07d}.out"
        txz_in = read_f90_bin(f5,nx,ny,nz,8) * u_scale * u_scale
        f6 = f"{dout}tyz_{timesteps[i]:07d}.out"
        tyz_in = read_f90_bin(f6,nx,ny,nz,8) * u_scale * u_scale
        f7 = f"{dout}q3_{timesteps[i]:07d}.out"
        q3_in = read_f90_bin(f7,nx,ny,nz,8) * u_scale * theta_scale
        # interpolate w, txz, tyz, q3 to u grid
        # create DataArrays
        w_da = xr.DataArray(w_in, dims=("x", "y", "z"), coords=dict(x=x, y=y, z=zw))
        txz_da = xr.DataArray(txz_in, dims=("x", "y", "z"), coords=dict(x=x, y=y, z=zw))
        tyz_da = xr.DataArray(tyz_in, dims=("x", "y", "z"), coords=dict(x=x, y=y, z=zw))
        q3_da = xr.DataArray(q3_in, dims=("x", "y", "z"), coords=dict(x=x, y=y, z=zw))
        # perform interpolation
        w_interp = w_da.interp(z=zu, method="linear", 
                               kwargs={"fill_value": "extrapolate"})
        txz_interp = txz_da.interp(z=zu, method="linear", 
                                   kwargs={"fill_value": "extrapolate"})
        tyz_interp = tyz_da.interp(z=zu, method="linear", 
                                   kwargs={"fill_value": "extrapolate"})
        q3_interp = q3_da.interp(z=zu, method="linear", 
                                 kwargs={"fill_value": "extrapolate"})
        # construct dictionary of data to save -- u-node variables only!
        data_save = {
                        "u": (["x","y","z"], u_in),
                        "v": (["x","y","z"], v_in),
                        "theta": (["x","y","z"], theta_in),
                    }
        # check fo using dissipation files
        if use_dissip:
            # read binary file
            f8 = f"{dout}dissip_{timesteps[i]:07d}.out"
            diss_in = read_f90_bin(f8,nx,ny,nz,8) * u_scale * u_scale * u_scale / Lz
            # interpolate to u-nodes
            diss_da = xr.DataArray(diss_in, dims=("x", "y", "z"), 
                                   coords=dict(x=x, y=y, z=zw))
            diss_interp = diss_da.interp(z=zu, method="linear", 
                                         kwargs={"fill_value": "extrapolate"})
        # construct dataset from these variables
        ds = xr.Dataset(
            data_save,
            coords={
                "x": x,
                "y": y,
                "z": zu
            },
            attrs={
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "Lx": Lx,
                "Ly": Ly,
                "Lz": Lz,
                "dx": dx,
                "dy": dy,
                "dz": dz,
                "label": simlabel
            })
        # now assign interpolated arrays that were on w-nodes
        ds["w"] = w_interp
        ds["txz"] = txz_interp
        ds["tyz"] = tyz_interp
        ds["q3"] = q3_interp
        if use_dissip:
            ds["dissip"] = diss_interp
        # hardcode dictionary of units to use by default
        if units is None:
            units = {
                "u": "m/s",
                "v": "m/s",
                "w": "m/s",
                "theta": "K",
                "txz": "m^2/s^2",
                "tyz": "m^2/s^2",
                "q3": "K m/s",
                "dissip": "m^2/s^3",
                "x": "m",
                "y": "m",
                "z": "m"
            }

        # loop and assign attributes
        for var in list(data_save.keys())+["x", "y", "z"]:
            ds[var].attrs["units"] = units[var]
        # save to netcdf file and continue
        fsave = f"{dnc}all_{timesteps[i]:07d}.nc"
        print(f"Saving file: {fsave.split(os.sep)[-1]}")
        ds.to_netcdf(fsave)

    print("Finished saving all files!")
    return
# ---------------------------------------------
def calc_stats(dnc, t0, t1, dt, delta_t, use_dissip, detrend_stats, tavg):
    """Read multiple output netcdf files created by sim2netcdf() to calculate
    averages in x, y, t and save as new netcdf file

    :param str dnc: absolute path to directory for loading netCDF files
    :param int t0: first timestep to process
    :param int t1: last timestep to process
    :param int dt: number of timesteps between files to load
    :param float delta_t: dimensional timestep in simulation (seconds)
    :param bool use_dissip: flag for loading dissipation rate files (SBL only)
    :param bool detrend_stats: flag for detrending fields in time when\
        calculating statistics
    :param str tavg: label denoting length of temporal averaging (e.g. 1h)
    """
    # directories and configuration
    timesteps = np.arange(t0, t1, dt, dtype=np.int32)
    # determine files to read from timesteps
    fall = [f"{dnc}all_{tt:07d}.nc" for tt in timesteps]
    nf = len(fall)
    # calculate array of times represented by each file
    times = np.array([i*delta_t*dt for i in range(nf)])
    # --------------------------------
    # Load files and clean up
    # --------------------------------
    print("Reading files...")
    dd = xr.open_mfdataset(fall, combine="nested", concat_dim="time")
    dd.coords["time"] = times
    dd.time.attrs["units"] = "s"
    # --------------------------------
    # Calculate statistics
    # --------------------------------
    print("Beginning calculations")
    # create empty dataset that will hold everything
    dd_stat = xr.Dataset()
    # list of base variables
    base = ["u", "v", "w", "theta"]
    base1 = ["u", "v", "w", "theta"] # use for looping over vars in case dissip not used
    # check for dissip
    if use_dissip:
        base.append("dissip")
    # calculate means
    for s in base:
        dd_stat[f"{s}_mean"] = dd[s].mean(dim=("time", "x", "y"))
    # calculate covars
    # u'w'
    dd_stat["uw_cov_res"] = xr.cov(dd.u, dd.w, dim=("time", "x", "y"))
    dd_stat["uw_cov_tot"] = dd_stat.uw_cov_res + dd.txz.mean(dim=("time","x","y"))
    # v'w'
    dd_stat["vw_cov_res"] = xr.cov(dd.v, dd.w, dim=("time", "x", "y"))
    dd_stat["vw_cov_tot"] = dd_stat.vw_cov_res + dd.tyz.mean(dim=("time","x","y"))
    # theta'w'
    dd_stat["tw_cov_res"] = xr.cov(dd.theta, dd.w, dim=("time", "x", "y"))
    dd_stat["tw_cov_tot"] = dd_stat.tw_cov_res + dd.q3.mean(dim=("time","x","y"))
    # calculate vars
    for s in base1:
        if detrend_stats:
            vv = np.var(detrend(dd[s], axis=0, type="linear"), axis=(0,1,2))
            dd_stat[f"{s}_var"] = xr.DataArray(vv, dims=("z"), coords=dict(z=dd.z))
        else:
            dd_stat[f"{s}_var"] = dd[s].var(dim=("time", "x", "y"))
    # rotate u_mean and v_mean so <v> = 0
    angle = np.arctan2(dd_stat.v_mean, dd_stat.u_mean)
    dd_stat["u_mean_rot"] = dd_stat.u_mean*np.cos(angle) + dd_stat.v_mean*np.sin(angle)
    dd_stat["v_mean_rot"] =-dd_stat.u_mean*np.sin(angle) + dd_stat.v_mean*np.cos(angle)
    # rotate instantaneous u and v for variances 
    # (not sure if necessary by commutative property but might as well)
    angle_inst = np.arctan2(dd.v.mean(dim=("x","y")), dd.u.mean(dim=("x","y")))
    u_rot = dd.u*np.cos(angle_inst) + dd.v*np.sin(angle_inst)
    v_rot =-dd.u*np.sin(angle_inst) + dd.v*np.cos(angle_inst)
    # recalculate u_var_rot, v_var_rot
    if detrend_stats:
        uvar_rot = np.var(detrend(u_rot, axis=0, type="linear"), axis=(0,1,2))
        dd_stat["u_var_rot"] = xr.DataArray(uvar_rot, dims=("z"), coords=dict(z=dd.z))
        vvar_rot = np.var(detrend(v_rot, axis=0, type="linear"), axis=(0,1,2))
        dd_stat["v_var_rot"] = xr.DataArray(vvar_rot, dims=("z"), coords=dict(z=dd.z))
    else:
        dd_stat["u_var_rot"] = u_rot.var(dim=("time", "x", "y"))
        dd_stat["v_var_rot"] = v_rot.var(dim=("time", "x", "y"))
    # --------------------------------
    # Add attributes
    # --------------------------------
    # copy from dd
    dd_stat.attrs = dd.attrs
    dd_stat.attrs["delta"] = (dd.dx * dd.dy * dd.dz) ** (1./3.)
    dd_stat.attrs["tavg"] = tavg
    # --------------------------------
    # Save output file
    # --------------------------------
    fsave = f"{dnc}mean_stats_xyt_{tavg}.nc"
    print(f"Saving file: {fsave}")
    with ProgressBar():
        dd_stat.to_netcdf(fsave, mode="w")
    print("Finished!")
    return
# ---------------------------------------------
def load_stats(fstats, SBL=False, display=False):
    """Reading function for average statistics files created from calc_stats()
    Load netcdf files using xarray and calculate numerous relevant parameters
    :param str fstats: absolute path to netcdf file for reading
    :param bool SBL: denote whether sim data is SBL or not to calculate\
        appropriate ABL depth etc., default=False
    :param bool display: print statistics from files, default=False
    :return dd: xarray dataset
    """
    print(f"Reading file: {fstats}")
    dd = xr.load_dataset(fstats)
    # calculate ustar and h
    dd["ustar"] = ((dd.uw_cov_tot**2.) + (dd.vw_cov_tot**2.)) ** 0.25
    dd["ustar2"] = dd.ustar ** 2.
    if SBL:
        dd["h"] = dd.z.where(dd.ustar2 <= 0.05*dd.ustar2[0], drop=True)[0] / 0.95
    else:
        # CBL
        dd["h"] = dd.z.isel(z=dd.tw_cov_tot.argmin())
    # save number of points within abl (z <= h)
    dd.attrs["nzabl"] = dd.z.where(dd.z <= dd.h, drop=True).size
    # grab ustar0 and calc tstar0 for normalizing in plotting
    dd["ustar0"] = dd.ustar.isel(z=0)
    dd["tstar0"] = -dd.tw_cov_tot.isel(z=0)/dd.ustar0
    # local thetastar
    dd["tstar"] = -dd.tw_cov_tot / dd.ustar
    # calculate TKE
    dd["e"] = 0.5 * (dd.u_var + dd.v_var + dd.w_var)
    # calculate Obukhov length L
    dd["L"] = -(dd.ustar0**3) * dd.theta_mean.isel(z=0) / (0.4 * 9.81 * dd.tw_cov_tot.isel(z=0))
    # calculate uh and wdir
    dd["uh"] = np.sqrt(dd.u_mean**2. + dd.v_mean**2.)
    dd["wdir"] = np.arctan2(-dd.u_mean, -dd.v_mean) * 180./np.pi
    dd["wdir"] = dd.wdir.where(dd.wdir < 0.) + 360.
    # calculate mean lapse rate between lowest grid point and z=h
    delta_T = dd.theta_mean.sel(z=dd.h, method="nearest") - dd.theta_mean[0]
    delta_z = dd.z.sel(z=dd.h, method="nearest") - dd.z[0]
    dd["dT_dz"] = delta_T / delta_z
    # calculate eddy turnover time TL
    dd["TL"] = dd.h / dd.ustar0
    dd["nTL"] = 3600. / dd.TL
    # calculate MOST dimensionless functions phim, phih
    kz = 0.4 * dd.z # kappa * z
    dd["phim"] = (kz/dd.ustar) * np.sqrt(dd.u_mean.differentiate("z", 2)**2.+\
                                         dd.v_mean.differentiate("z", 2)**2.)
    dd["phih"] = (kz/dd.tstar) * dd.theta_mean.differentiate("z", 2)
    # MOST stability parameter z/L
    dd["zL"] = dd.z / dd.L
    if SBL:
        # calculate TKE-based sbl depth
        dd["he"] = dd.z.where(dd.e <= 0.05*dd.e[0], drop=True)[0]
        # calculate h/L as global stability parameter
        dd["hL"] = dd.h / dd.L
        # create string for labels from hL
        dd.attrs["label3"] = f"$h/L = {{{dd.hL.values:3.2f}}}$"
        # calculate Richardson numbers
        # sqrt((du_dz**2) + (dv_dz**2))
        dd["du_dz"] = np.sqrt(dd.u_mean.differentiate("z", 2)**2. +\
                              dd.v_mean.differentiate("z", 2)**2.)
        # Rig = N^2 / S^2
        dd["N2"] = dd.theta_mean.differentiate("z", 2) * 9.81 / dd.theta_mean.isel(z=0)
        # flag negative values of N^2
        dd.N2[dd.N2 < 0.] = np.nan
        dd["Rig"] = dd.N2 / dd.du_dz / dd.du_dz
        # Rif = beta * w'theta' / (u'w' du/dz + v'w' dv/dz)
        dd["Rif"] = (9.81/dd.theta_mean.isel(z=0)) * dd.tw_cov_tot /\
                    (dd.uw_cov_tot*dd.u_mean.differentiate("z", 2) +\
                     dd.vw_cov_tot*dd.v_mean.differentiate("z", 2))
        # # bulk Richardson number Rib based on values at top of sbl and sfc
        # dz = dd.z[dd.nzsbl] - dd.z[0]
        # dTdz = (dd.theta_mean[dd.nzsbl] - dd.theta_mean[0]) / dz
        # dUdz = (dd.u_mean[dd.nzsbl] - dd.u_mean[0]) / dz
        # dVdz = (dd.v_mean[dd.nzsbl] - dd.v_mean[0]) / dz
        # dd.attrs["Rib"] = (dTdz * 9.81 / dd.theta_mean[0]) / (dUdz**2. + dVdz**2.)
        # calc Ozmidov scale real quick
        dd["Lo"] = np.sqrt(-dd.dissip_mean / (dd.N2 ** (3./2.)))
        # calculate Kolmogorov microscale: eta = (nu**3 / dissip) ** 0.25
        dd["eta"] = ((1.14e-5)**3. / (-dd.dissip_mean)) ** 0.25
        # calculate MOST dimensionless dissipation rate: kappa*z*epsilon/ustar^3
        dd["phie"] = -1*dd.dissip_mean*kz / (dd.ustar**3.)
        # calculate gradient scales from Sorbjan 2017, Greene et al. 2022
        l0 = 19.22 # m
        l1 = 1./(dd.Rig**(3./2.)).where(dd.z <= dd.h, drop=True)
        kz = 0.4 * dd.z.where(dd.z <= dd.h, drop=True)
        dd["Ls"] = kz / (1 + (kz/l0) + (kz/l1))
        dd["Us"] = dd.Ls * np.sqrt(dd.N2)
        dd["Ts"] = dd.Ls * dd.theta_mean.differentiate("z", 2)
        # calculate local Obukhov length Lambda
        dd["LL"] = -(dd.ustar**3.) * dd.theta_mean / (0.4 * 9.81 * dd.tw_cov_tot)
        # calculate level of LLJ: zj
        dd["zj"] = dd.z.isel(z=dd.uh.argmax())
    # print table statistics
    if display:
        print(f"---{dd.stability}---")
        print(f"u*: {dd.ustar0.values:4.3f} m/s")
        print(f"theta*: {dd.tstar0.values:5.4f} K")
        print(f"Q*: {1000*dd.tw_cov_tot.isel(z=0).values:4.3f} K m/s")
        print(f"h: {dd.h.values:4.3f} m")
        print(f"L: {dd.L.values:4.3f} m")
        print(f"h/L: {(dd.h/dd.L).values:4.3f}")
        # print(f"Rib: {dd.Rib.values:4.3f}")
        print(f"zj/h: {(dd.z.isel(z=dd.uh.argmax())/dd.h).values:4.3f}")
        print(f"dT/dz: {1000*dd.dT_dz.values:4.1f} K/km")
        print(f"TL: {dd.TL.values:4.1f} s")
        print(f"nTL: {dd.nTL.values:4.1f}")

    return dd
# ---------------------------------------------
def load_full(dnc, t0, t1, dt, delta_t, SBL=False, stats=None):
    """Reading function for multiple instantaneous volumetric netcdf files
    Load netcdf files using xarray
    :param str dnc: abs path directory for location of netcdf files
    :param int t0: first timestep to process
    :param int t1: last timestep to process
    :param int dt: number of timesteps between files to load
    :param float delta_t: dimensional timestep in simulation (seconds)
    :param bool SBL: calculate SBL-specific parameters. defaulte=False
    :param str stats: name of statistics file. default=None

    :return dd: xarray dataset of 4d volumes
    :return s: xarray dataset of statistics file
    """
    # load individual files into one dataset
    timesteps = np.arange(t0, t1+1, dt, dtype=np.int32)
    # determine files to read from timesteps
    fall = [f"{dnc}all_{tt:07d}.nc" for tt in timesteps]
    nf = len(fall)
    # calculate array of times represented by each file
    times = np.array([i*delta_t*dt for i in range(nf)])
    # read files
    print("Loading files...")
    dd = xr.open_mfdataset(fall, combine="nested", concat_dim="time")
    dd.coords["time"] = times
    dd.time.attrs["units"] = "s"
    if stats is not None:
        # load stats file
        s = load_stats(dnc+stats, SBL=SBL)
        # calculate rotated u, v based on xy mean at each timestep
        uxy = dd.u.mean(dim=("x","y"))
        vxy = dd.v.mean(dim=("x","y"))
        angle = np.arctan2(vxy, uxy)
        dd["u_rot"] = dd.u*np.cos(angle) + dd.v*np.sin(angle)
        dd["v_rot"] =-dd.u*np.sin(angle) + dd.v*np.cos(angle)
        # return both dd and s
        return dd, s
    # just return dd if no stats
    return dd
# ---------------------------------------------
def timeseries2netcdf(dout, dnc, scales, delta_t, nz, Lz, nhr, tf, simlabel):
    """Load raw timeseries data at each level and combine into single
    netcdf file with dimensions (time, z)
    :param str dout: absolute path to directory with LES output binary files
    :param str dnc: absolute path to directory for saving output netCDF files
    :param tuple<Quantity> scales: dimensional scales from LES code\
        (uscale, Tscale)
    :param float delta_t: dimensional timestep in simulation (seconds)
    :param int nz: resolution of simulation in vertical
    :param float Lz: height of domain in m
    :param float nhr: number of hours to load, counting backwards from end
    :param int tf: final timestep in file
    :param str simlabel: unique identifier for batch of files
    """
    # grab relevent parameters
    u_scale, theta_scale = scales
    dz = Lz/nz
    # define z array
    # u- and w-nodes are staggered
    # zw = 0:Lz:nz
    # zu = dz/2:Lz-dz/2:nz-1
    # interpolate w, txz, tyz, q3 to u grid
    zw = np.linspace(0., Lz, nz)
    zu = np.linspace(dz/2., Lz+dz/2., nz)
    # determine number of hours to process from tavg
    nt = int(nhr*3600./delta_t)
    istart = tf - nt
    # define array of time in seconds
    time = np.linspace(0., nhr*3600.-delta_t, nt, dtype=np.float64)
    print(f"Loading {nt} timesteps = {nhr} hr for simulation {simlabel}")
    # define DataArrays for u, v, w, theta, txz, tyz, q3
    # shape(nt,nz)
    # u, v, theta
    u_ts, v_ts, theta_ts =\
    (xr.DataArray(np.zeros((nt, nz), dtype=np.float64),
                  dims=("t", "z"), coords=dict(t=time, z=zu)) for _ in range(3))
    # w, txz, tyz, q3
    w_ts, txz_ts, tyz_ts, q3_ts =\
    (xr.DataArray(np.zeros((nt, nz), dtype=np.float64),
                  dims=("t", "z"), coords=dict(t=time, z=zw)) for _ in range(4))
    # now loop through each file (one for each jz)
    for jz in range(nz):
        print(f"Loading timeseries data, jz={jz}")
        fu = f"{dout}u_timeseries_c{jz:03d}.out"
        u_ts[:,jz] = np.loadtxt(fu, skiprows=istart, usecols=1)
        fv = f"{dout}v_timeseries_c{jz:03d}.out"
        v_ts[:,jz] = np.loadtxt(fv, skiprows=istart, usecols=1)
        fw = f"{dout}w_timeseries_c{jz:03d}.out"
        w_ts[:,jz] = np.loadtxt(fw, skiprows=istart, usecols=1)
        ftheta = f"{dout}t_timeseries_c{jz:03d}.out"
        theta_ts[:,jz] = np.loadtxt(ftheta, skiprows=istart, usecols=1)
        ftxz = f"{dout}txz_timeseries_c{jz:03d}.out"
        txz_ts[:,jz] = np.loadtxt(ftxz, skiprows=istart, usecols=1)
        ftyz = f"{dout}tyz_timeseries_c{jz:03d}.out"
        tyz_ts[:,jz] = np.loadtxt(ftyz, skiprows=istart, usecols=1)
        fq3 = f"{dout}q3_timeseries_c{jz:03d}.out"
        q3_ts[:,jz] = np.loadtxt(fq3, skiprows=istart, usecols=1)
    # apply scales
    u_ts *= u_scale
    v_ts *= u_scale
    w_ts *= u_scale
    theta_ts *= theta_scale
    txz_ts *= (u_scale * u_scale)
    tyz_ts *= (u_scale * u_scale)
    q3_ts *= (u_scale * theta_scale)
    # define dictionary of attributes
    attrs = {"label": simlabel, "dt": delta_t, "nt": nt, "nz": nz, "total_time": nhr}
    # combine DataArrays into Dataset and save as netcdf
    # initialize empty Dataset
    ts_all = xr.Dataset(data_vars=None, coords=dict(t=time, z=zu), attrs=attrs)
    # now store
    ts_all["u"] = u_ts
    ts_all["v"] = v_ts
    ts_all["w"] = w_ts.interp(z=zu, method="linear", 
                              kwargs={"fill_value": "extrapolate"})
    ts_all["theta"] = theta_ts
    ts_all["txz"] = txz_ts.interp(z=zu, method="linear", 
                                  kwargs={"fill_value": "extrapolate"})
    ts_all["tyz"] = tyz_ts.interp(z=zu, method="linear", 
                                  kwargs={"fill_value": "extrapolate"})
    ts_all["q3"] = q3_ts.interp(z=zu, method="linear", 
                                kwargs={"fill_value": "extrapolate"})
    # save to netcdf
    fsave_ts = f"{dnc}timeseries_all_{nhr}h.nc"
    with ProgressBar():
        ts_all.to_netcdf(fsave_ts, mode="w")
        
    print(f"Finished saving timeseries for simulation {simlabel}")

    return
# ---------------------------------------------
def load_timeseries(dnc, detrend=True, tavg="1.0h"):
    """Reading function for timeseries files created from timseries2netcdf()
    Load netcdf files using xarray and calculate numerous relevant parameters
    :param str dnc: path to netcdf directory for simulation
    :param bool detrend: detrend timeseries for calculating variances, default=True
    :param str tavg: select which timeseries file to use in hours, default="1h"
    :return d: xarray dataset
    """
    # load timeseries data
    fts = f"timeseries_all_{tavg}.nc"
    d = xr.open_dataset(dnc+fts)
    # calculate means
    for v in ["u", "v", "w", "theta"]:
        d[f"{v}_mean"] = d[v].mean("t") # average in time
    # rotate coords so <v> = 0
    angle = np.arctan2(d.v_mean, d.u_mean)
    d["u_mean_rot"] = d.u_mean*np.cos(angle) + d.v_mean*np.sin(angle)
    d["v_mean_rot"] =-d.u_mean*np.sin(angle) + d.v_mean*np.cos(angle)
    # rotate instantaneous u and v
    d["u_rot"] = d.u*np.cos(angle) + d.v*np.sin(angle)
    d["v_rot"] =-d.u*np.sin(angle) + d.v*np.cos(angle)
    # calculate "inst" vars
    if detrend:
        ud = xrft.detrend(d.u, dim="t", detrend_type="linear")
        udr = xrft.detrend(d.u_rot, dim="t", detrend_type="linear")
        vd = xrft.detrend(d.v, dim="t", detrend_type="linear")
        vdr = xrft.detrend(d.v_rot, dim="t", detrend_type="linear")
        wd = xrft.detrend(d.w, dim="t", detrend_type="linear")
        td = xrft.detrend(d.theta, dim="t", detrend_type="linear")
        # store these detrended variables
        d["ud"] = ud
        d["udr"] = udr
        d["vd"] = vd
        d["vdr"] = vdr
        d["wd"] = wd
        d["td"] = td
        # now calculate vars
        d["uu"] = ud * ud
        d["uur"] = udr * udr
        d["vv"] = vd * vd
        d["vvr"] = vdr * vdr
        d["ww"] = wd * wd
        d["tt"] = td * td
        # calculate "inst" covars
        d["uw"] = (ud * wd) + d.txz
        d["vw"] = (vd * wd) + d.tyz
        d["tw"] = (td * wd) + d.q3
    else:
        d["uu"] = (d.u - d.u_mean) * (d.u - d.u_mean)
        d["uur"] = (d.u_rot - d.u_mean_rot) * (d.u_rot - d.u_mean_rot)
        d["vv"] = (d.v - d.v_mean) * (d.v - d.v_mean)
        d["vvr"] = (d.v_rot - d.v_mean_rot) * (d.v_rot - d.v_mean_rot)
        d["ww"] = (d.w - d.w_mean) * (d.w - d.w_mean)
        d["tt"] = (d.theta - d.theta_mean) * (d.theta - d.theta_mean)
        # calculate "inst" covars
        d["uw"] = (d.u - d.u_mean) * (d.w - d.w_mean) + d.txz
        d["vw"] = (d.v - d.v_mean) * (d.w - d.w_mean) + d.tyz
        d["tw"] = (d.theta - d.theta_mean) * (d.w - d.w_mean) + d.q3
    
    return d
# ---------------------------------------------
@njit
def interp_uas(dat, z_LES, z_UAS):
    """Interpolate LES virtual tower timeseries data in the vertical to match
    ascent rate of emulated UAS to create timeseries of ascent data
    :param float dat: 2d array of the field to interpolate, shape(nt, nz)
    :param float z_LES: 1d array of LES grid point heights
    :param float z_UAS: 1d array of new UAS z from ascent & sampling rates
    Outputs 2d array of interpolated field, shape(nt, len(z_UAS))
    """
    nt = np.shape(dat)[0]
    nz = len(z_UAS)
    dat_interp = np.zeros((nt, nz), dtype=np.float64)
    for i in range(nt):
        dat_interp[i,:] = np.interp(z_UAS, z_LES, dat[i,:])
    return dat_interp
# ---------------------------------------------
def UASprofile(ts, zmax=2000., err=None, ascent_rate=3.0, time_average=3.0, time_start=0.0):
    """Emulate a vertical profile from a rotary-wing UAS sampling through a
    simulated ABL with chosen constant ascent rate and time averaging
    :param xr.Dataset ts: timeseries data from virtual tower created by
        timeseries2netcdf()
    :param float zmax: maximum height within domain to consider in m, 
        default=2000.
    :param xr.Dataset err: profile of errors to accompany emulated
        measurements, default=None
    :param float ascent_rate: ascent rate of UAS in m/s, default=3.0
    :param float time_average: averaging time bins in s, default=3.0
    :param float time_start: when to initialize ascending profile
        in s, default=0.0
    Returns new xarray Dataset with emulated profile
    """
    # First, calculate array of theoretical altitudes based on the base time
    # vector and ascent_rate while keeping account for time_start
    zuas = ascent_rate * ts.t.values
    # find the index in ts.time that corresponds to time_start
    istart = int(time_start / ts.dt)
    # set zuas[:istart] = 0 and then subtract everything after that
    zuas[:istart] = 0
    zuas[istart:] -= zuas[istart]
    # now only grab indices where 1 m <= zuas <= zmax
    iuse = np.where((zuas >= 1.) & (zuas <= zmax))[0]
    zuas = zuas[iuse]
    # calculate dz_uas from ascent_rate and time_average
    dz_uas = ascent_rate * time_average
    # loop over keys and interpolate
    interp_keys = ["u", "v", "theta"]
    d_interp = {} # define empty dictionary for looping
    for key in interp_keys:
        print(f"Interpolating {key}...")
        d_interp[key] = interp_uas(ts[key].isel(t=iuse).values,
                                   ts.z.values, zuas)

    # grab data from interpolated arrays to create simulated raw UAS profiles
    # define xarray dataset to eventually store all
    uas_raw = xr.Dataset(data_vars=None, coords=dict(z=zuas))
    # begin looping through keys
    for key in interp_keys:
        # define empty list
        duas = []
        # loop over altitudes/times
        for i in range(len(iuse)):
            duas.append(d_interp[key][i,i])
        # assign to uas_raw
        uas_raw[key] = xr.DataArray(data=np.array(duas), coords=dict(z=zuas))
    
    # emulate post-processing and average over altitude bins
    # can do this super easily with xarray groupby_bins
    # want bins to be at the midpoint between dz_uas grid
    zbin = np.arange(dz_uas/2, zmax-dz_uas/2, dz_uas)
    # group by altitude bins and calculate mean in one line
    uas_mean = uas_raw.groupby_bins("z", zbin).mean("z", skipna=True)
    # fix z coordinates: swap z_bins out for dz_uas grid
    znew = np.arange(dz_uas, zmax-dz_uas, dz_uas)
    # create new coordinate "z" from znew that is based on z_bins, then swap and drop
    uas_mean = uas_mean.assign_coords({"z": ("z_bins", znew)}).swap_dims({"z_bins": "z"})
    # # only save data for z <= h
    # h = err.z.max()
    # uas_mean = uas_mean.where(uas_mean.z <= h, drop=True)
    # calculate wspd, wdir from uas_mean profile
    uas_mean["wspd"] = (uas_mean.u**2. + uas_mean.v**2.) ** 0.5
    wdir = np.arctan2(-uas_mean.u, -uas_mean.v) * 180./np.pi
    wdir[wdir < 0.] += 360.
    uas_mean["wdir"] = wdir
    #
    # interpolate errors for everything in uas_mean
    #
    if err is not None:
        uas_mean["wspd_err"] = err.uh.interp(z=uas_mean.z)
        uas_mean["wdir_err"] = err.alpha.interp(z=uas_mean.z)
        uas_mean["theta_err"] = err.theta.interp(z=uas_mean.z)

    return uas_mean
# ---------------------------------------------
def ec_tow(ts, h, time_average=1800.0, time_start=0.0):
    """Emulate a tower extending throughout ABL with EC system at each vertical
    gridpoint and calculate variances and covariances
    :param xr.Dataset ts: dataset with virtual tower data to construct UAS prof
    :param float h: ABL depth in m
    :param float time_average: time range in s to avg timeseries; default=1800
    :param float time_start: when to initialize averaging; default=0.0
    :param bool quicklook: flag to make quicklook of raw vs averaged profiles
    Outputs new xarray Dataset with emulated vars and covars
    """
    # check if time_average is an array or single value and convert to iterable
    if np.shape(time_average) == ():
        time_average = np.array([time_average])
    else:
        time_average = np.array(time_average)
    # initialize empty dataset to hold everything
    ec_ = xr.Dataset(data_vars=None, coords=dict(z=ts.z, Tsample_ec=time_average))
    # loop through variable names to initialize empty DataArrays
    for v in ["uw_cov_tot","vw_cov_tot", "tw_cov_tot", "ustar2", 
              "u_var", "v_var", "w_var", "theta_var",
              "u_var_rot", "v_var_rot", "e"]:
        ec_[v] = xr.DataArray(data=np.zeros((len(ts.z), len(time_average)), 
                                            dtype=np.float64),
                              coords=dict(z=ts.z, Tsample_ec=time_average))
    # loop over time_average to calculate ec stats
    for jt, iT in enumerate(time_average):
        # first find the index in df.t that corresponds to time_start
        istart = int(time_start / ts.dt)
        # determine how many indices to use from time_average
        nuse = int(iT / ts.dt)
        # create array of indices to use
        iuse = np.linspace(istart, istart+nuse-1, nuse, dtype=np.int32)
        # begin calculating statistics
        # use the detrended stats from load_timeseries
        # u'w'
        ec_["uw_cov_tot"][:,jt] = ts.uw.isel(t=iuse).mean("t")
        # v'w'
        ec_["vw_cov_tot"][:,jt] = ts.vw.isel(t=iuse).mean("t")
        # theta'w'
        ec_["tw_cov_tot"][:,jt] = ts.tw.isel(t=iuse).mean("t")
        # ustar^2 = sqrt(u'w'^2 + v'w'^2)
        ec_["ustar2"][:,jt] = ((ec_.uw_cov_tot[:,jt]**2.) + (ec_.vw_cov_tot[:,jt]**2.)) ** 0.5
        # variances
        ec_["u_var"][:,jt] = ts.uu.isel(t=iuse).mean("t")
        ec_["u_var_rot"][:,jt] = ts.uur.isel(t=iuse).mean("t")
        ec_["v_var"][:,jt] = ts.vv.isel(t=iuse).mean("t")
        ec_["v_var_rot"][:,jt] = ts.vvr.isel(t=iuse).mean("t")
        ec_["w_var"][:,jt] = ts.ww.isel(t=iuse).mean("t")
        ec_["theta_var"][:,jt] = ts.tt.isel(t=iuse).mean("t")
        # calculate TKE
        ec_["e"][:,jt] = 0.5 * (ec_.u_var.isel(Tsample_ec=jt) +\
                                ec_.v_var.isel(Tsample_ec=jt) +\
                                ec_.w_var.isel(Tsample_ec=jt))
    
    # only return ec where z <= h
    return ec_.where(ec_.z <= h, drop=True)