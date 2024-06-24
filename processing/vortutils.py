# ------------------------------------------------
# Name: vortutils.py
# Author: Robby Frost
# University of Oklahoma
# Created: 21 June 2024
# Purpose: Functions for calculating vorticity 
# statistics from large eddy simulations of the
# convective boundary layer
# ------------------------------------------------

import xarray as xr
import numpy as np
import sys
sys.path.append("/home/rfrost/LES-utils/")
from spec import autocorr_2d
from tranutils import ls_rot
from dask.diagnostics import ProgressBar

# ---------------------------------
# Settings to run functions
# ---------------------------------
# name of simulation
sim_label = "full_step_15"
# director for netCDF simulation output files
dnc = f"/home/rfrost/simulations/nc/{sim_label}/"
# first timestep
t0 = 576000
# final timestep
t1 = 1152000
# number of timesteps between files
dt = 1000
# dimensional timestep (seconds)
delta_t = 0.05

# flag to run calc_vorticity
calc_vort = False
# flag to run calc_vort_2d_autocorr
calc_vort_ac = False
# flag to run calc_vort_ls
calc_vort_ls = True

# ---------------------------------
# Calculate the vorticity terms
# ---------------------------------
def calc_vorticity(dnc, t0, t1, dt, delta_t):
    """Calculate 4d vorticity statistics on netCDF LES output

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
    print("Beginning vorticity calculations")
    # create empty dataset that will hold everything
    dd_stat = xr.Dataset()
    # dd_ac = xr.Dataset()
    # dd_ls = xr.Dataset()

    # calculate horizontal vorticity
    dd_stat["zeta1"] = dd.w.differentiate(coord="y") - dd.v.differentiate(coord="z")
    dd_stat["zeta2"] = dd.w.differentiate(coord="x") - dd.u.differentiate(coord="z")
    # calculate vertical vorticity
    dd_stat["zeta3"] = dd.v.differentiate(coord="x") - dd.u.differentiate(coord="y")
    print("Finished vorticity components.")

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
    print("Finished vorticity spatial averaging.")

    # # 2d autocorrelation
    # R_ds = autocorr_2d("naw", dd_stat, ["zeta1", "zeta2", "zeta3"], timeavg=False, output=False)
    # dd_ac["zeta1_autocorr2d"] = R_ds["zeta1"]
    # dd_ac["zeta2_autocorr2d"] = R_ds["zeta3"]
    # dd_ac["zeta3_autocorr2d"] = R_ds["zeta3"]
    # print("Finished vorticity 2d autocorrelation.")

    # # integral length scales
    # Lw_ds_zeta1 = ls_rot(R_ds["zeta1"], t0, t1, dt, read_in=False, output=False)
    # dd_ls["ls_zeta1_rolls"] = Lw_ds_zeta1["rolls"]
    # dd_ls["ls_zeta1_normal"] = Lw_ds_zeta1["normal"]
    # Lw_ds_zeta2 = ls_rot(R_ds["zeta2"], t0, t1, dt, read_in=False, output=False)
    # dd_ls["ls_zeta2_rolls"] = Lw_ds_zeta2["rolls"]
    # dd_ls["ls_zeta2_normal"] = Lw_ds_zeta2["normal"]
    # Lw_ds_zeta3 = ls_rot(R_ds["zeta3"], t0, t1, dt, read_in=False, output=False)
    # dd_ls["ls_zeta3_rolls"] = Lw_ds_zeta3["rolls"]
    # dd_ls["ls_zeta3_normal"] = Lw_ds_zeta3["normal"]
    # print("Finished vorticity length scales.")

    # Add attributes
    dd_stat.attrs = dd.attrs
    # dd_stat.attrs, dd_ac.attrs, dd_ls.attrs = [dd.attrs for _ in range(3)]
    # dd_stat.attrs["delta"], dd_ac.attrs["delta"], dd_ls.attrs["delta"] = [(dd.dx * dd.dy * dd.dz) ** (1./3.) for _ in range(3)]

    # save output
    fs_stat = f"{dnc}{t0}_{t1}_vort.nc"
    # fs_ac = f"{dnc}{t0}_{t1}_vort_autocorr.nc"
    # fs_ls = f"{dnc}{t0}_{t1}_vort_ls.nc"
    with ProgressBar():
        # save vorticity stats
        print(f"Saving file: {fs_stat}")
        dd_stat.to_netcdf(fs_stat, mode="w")
        # # save vorticity autocorrelation
        # print(f"Saving file: {fs_ac}")
        # dd_ac.to_netcdf(fs_ac, mode="w")
        # # save vorticity length scales
        # print(f"Saving file: {fs_ls}")
        # dd_ls.to_netcdf(fs_ls, mode="w")

    print("Finished!")
    return

# ---------------------------------
# Calculate vorticity 2d autocorr
# ---------------------------------
def calc_vort_2d_autocorr(dnc, t0, t1):
    """Calculate 2d autocorrelation of vorticity using output from calc_vorticity

    :param str dnc: absolute path to directory for reading/saving output netCDF files
    :param int t0: first timestep for stats to be computed
    :param int t1: final timestep for stats to be computed
    """

    # read in vorticity statistics
    dd_stat = xr.open_dataset(f"{dnc}{t0}_{t1}_vort.nc")
    # dataset for autocorr data
    dd_ac = xr.Dataset()

    # 2d autocorrelation
    R_ds = autocorr_2d("naw", dd_stat, ["zeta1", "zeta2", "zeta3"], timeavg=False, output=False)
    dd_ac["zeta1_autocorr2d"] = R_ds["zeta1"]
    dd_ac["zeta2_autocorr2d"] = R_ds["zeta3"]
    dd_ac["zeta3_autocorr2d"] = R_ds["zeta3"]
    print("Finished vorticity 2d autocorrelation.")

    # save output
    fs_ac = f"{dnc}{t0}_{t1}_vort_autocorr.nc"
    with ProgressBar():
        # save vorticity autocorrelation
        print(f"Saving file: {fs_ac}")
        dd_ac.to_netcdf(fs_ac, mode="w")

    print("Finished!")
    return

# ---------------------------------
# Calculate vorticity length scales
# ---------------------------------
def calc_vort_length_scales(dnc, t0, t1):
    """Calculate integral length scales of vorticity using output from calc_vort_autocorr

    :param str dnc: absolute path to directory for reading/saving output netCDF files
    :param int t0: first timestep for stats to be computed
    :param int t1: final timestep for stats to be computed
    """

    # read in vorticity autocorr data
    R_ds = xr.open_dataset(f"{dnc}{t0}_{t1}_vort_autocorr.nc")
    # dataset for length scale data
    dd_ls = xr.Dataset()

    # integral length scales
    Lw_ds_zeta1 = ls_rot(R_ds["zeta1_autocorr2d"], t0, t1, dt, read_in=False, output=False)
    dd_ls["ls_zeta1_rolls"] = Lw_ds_zeta1["rolls"]
    dd_ls["ls_zeta1_normal"] = Lw_ds_zeta1["normal"]
    Lw_ds_zeta2 = ls_rot(R_ds["zeta2_autocorr2d"], t0, t1, dt, read_in=False, output=False)
    dd_ls["ls_zeta2_rolls"] = Lw_ds_zeta2["rolls"]
    dd_ls["ls_zeta2_normal"] = Lw_ds_zeta2["normal"]
    Lw_ds_zeta3 = ls_rot(R_ds["zeta3_autocorr2d"], t0, t1, dt, read_in=False, output=False)
    dd_ls["ls_zeta3_rolls"] = Lw_ds_zeta3["rolls"]
    dd_ls["ls_zeta3_normal"] = Lw_ds_zeta3["normal"]
    print("Finished vorticity length scales.")

    # save output
    fs_ls = f"{dnc}{t0}_{t1}_vort_ls.nc"
    with ProgressBar():
        # save vorticity length scales
        print(f"Saving file: {fs_ls}")
        dd_ls.to_netcdf(fs_ls, mode="w")

    print("Finished!")
    return

# ---------------------------------
# Run desired scripts
# ---------------------------------
if calc_vort:
    calc_vorticity(dnc, t0, t1, dt, delta_t)

if calc_vort_ac:
    calc_vort_2d_autocorr(dnc,t0,t1)

if calc_vort_ls:
    calc_vort_length_scales(dnc,t0,t1)