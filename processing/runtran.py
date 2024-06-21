# ------------------------------------------------
# Name: runtran.py
# Author: Robert M. Frost
# University of Oklahoma
# Created: 28 April 2023
# Purpose: Script for running functions in 
# tranutils.py
# ------------------------------------------------
import sys
sys.path.append("/home/rfrost/LES-utils/")

import xarray as xr
from spec import autocorr_2d
from LESutils import sim2netcdf, load_full, nc_rotate
from tranutils import calc_stats_tran, polar_grid, roll_factor, length_scales, ls_rot, calc_vorticity
# ---------------------------------
# Parameters for script
# --------------------------------- 

# label for simulation files
simlabel = "full_step_15"
# directory for raw simulation output
dout = f"/home/rfrost/simulations/nc/{simlabel}"
# dout = f"/share/yeti/simulations/cbl/{simlabel}/output/"
# directory for netCDF files to be read then saved
dnc = f"/home/rfrost/simulations/nc/{simlabel}/"
# simulation resolution
nx = 160
ny = 160
nz = 160
# domain size in meters
Lx = 12000
Ly = 12000
Lz = 2000
# scales from LES code (uscale, tscale)
scales = [0.4, 300]
# start and end timesteps to be read in
t0 = 576000
t1 = 1152000
# output frequency in timesteps
dt = 1000
# temporal resolution in seconds
delta_t = 0.05
# dimensionless height
heights = [0.10, 0.25, 0.50]

# flag to run sim2netcdf
netcdf = False
# rotate or not
rotate = False
# flag to run calc_stats_tran
stats = False
# flag to calculate vorticity fields
vort = True
# flag for zi calculation method (0=heat flux method, 1=theta gradient method)
zi_mode = 1
# flag to calculate 2D autocorrelation or not
autocorr = False
# flag to convert to polar coordiantes
polar = False
# flag to calculate roll factor
roll = False
# flag to calculate rotated integral length scales
length_rot = False
# flag to calculate integral length scales
length = False

# number of angular and radial bins (angular bins must be odd number)
ntbin = 21
nrbin = 20

# ---------------------------------
# Calling functions to run
# --------------------------------- 

# sim2netcdf
if netcdf:
    print("Begin sim2netcdf...")
    sim2netcdf(dout, dnc, (nx,ny,nz), (Lx,Ly,Lz), scales, t0, t1, dt, 
               use_dissip=False, use_q=False, simlabel=simlabel)
    print("Finished sim2netcdf! \n")

# rotate coords along wind angle
if rotate:
    print("Begin nc_rotate...")
    nc_rotate(dnc, t0, t1, dt)
    print("Finished nc_rotate! \n")

# calc_stats_tran
if stats:
    print("Begin calc_stats_tran...")
    calc_stats_tran(dnc, t0, t1, dt, delta_t, zi_mode)
    print("Finished calc_stats_tran! \n")

# vorticity calculation
if vort:
    print("Begin calc_vorticity...")
    calc_vorticity(dnc, t0, t1, dt, delta_t)
    print("Finished calc_vorticity! \n")

# autocorr_2d
if autocorr:
    print("Begin autocorr_2d...")
    if rotate:
        # load in volumetric output
        df = load_full(dnc, t0, t1, dt, delta_t, rotate=True)
    else:
        df = load_full(dnc, t0, t1, dt, delta_t)
    # list of variables to have autocorrelation calculated over
    var = ["w"]
    # compute 2D autocorrelation
    autocorr_2d(dnc, df, var, timeavg=False)
    print("Finished autocorr_2d! \n")

# polar_grid
if polar:
    print("Begin polar_grid...")
    # load in volumetric output
    df = load_full(dnc, t0, t1, dt, delta_t)
    # rotate coordinates of autocorr
    polar_grid(df, dnc, heights, Lx, ntbin, nrbin)
    print("Finished polar_grid (finally)! \n")

# roll_factor
if roll:
    print("Begin roll factor...")
# load in volumetric output
    df = load_full(dnc, t0, t1, dt, delta_t)
    # compute roll factor
    roll_factor(dnc, df, heights)
    print("Finished roll_factor! \n")

# # length_scales
# if length:
#     print("Begin length_scales...")
#     length_scales(dnc, t0, t1, dt, height, avg_method, avg_time)
#     print("Finished lenght_scales! \n")

# ls_rot
if length_rot:
    print("Begin ls_rot...")
    ls_rot(dnc, t0, t1, dt)
    print("Finished ls_rot! \n")