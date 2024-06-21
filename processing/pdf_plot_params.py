# ------------------------------------------------
# Name: pdf_plot_params.py
# Author: Robby Frost
# University of Oklahoma
# Created: 21 June 2024
# Purpose: Calculate parameters needed to plot
# probability density functions of CBL vorticity
# ------------------------------------------------
# imports
import xarray as xr
import numpy as np

# settings
dnc = f"/home/rfrost/simulations/nc/"
# list of simulations to analyze
sims = ["full_step_6", "full_step_9", "full_step_12", "full_step_15"]
# start and end timesteps
t0 = 576000
t1 = 1152000
# start and end in hours
t0hr = t0 * 0.05 / 3600
t1hr = t1 * 0.05 / 3600
# height index
h_idx = 0

# code begins
vort_all = []
heights = []

# loop over sims
for sim in sims:
    # vorticity stats
    vort = xr.open_dataset(f"{dnc}{sim}/{t0}_{t1}_vort.nc")
    # convert time to hours
    vort["time"] = vort.time / 3600 + t0hr
    vort_all.append(vort)
    heights.append(vort.z[h_idx].values)

def compute_pdf(data, bins=50):
    """Compute the probability density function for the given data."""
    data = data[~np.isnan(data)]  # Remove NaN values
    counts, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_width = bin_edges[1] - bin_edges[0]
    prob = counts * bin_width
    return prob, bin_edges

# Initialize lists to store the results for each dataset
bin_width_stat, bin_width_highq0, bin_width_lowq0 = [], [], []
bin_edges_stat, bin_edges_highq0, bin_edges_lowq0 = [], [], []
prob_stat, prob_highq0, prob_lowq0 = [], [], []

for i, v in enumerate(vort_all):
    # Extract and flatten the data
    data_stat = v.zeta3[:144,:,:,h_idx].mean(dim="time").values.flatten()
    data_highq0 = v.zeta3[144:360,:,:,h_idx].mean(dim="time").values.flatten()
    data_lowq0 = v.zeta3[360:,:,:,h_idx].mean(dim="time").values.flatten()

    # Compute PDFs
    prob_stat_i, bin_edges_stat_i = compute_pdf(data_stat)
    prob_highq0_i, bin_edges_highq0_i = compute_pdf(data_highq0)
    prob_lowq0_i, bin_edges_lowq0_i = compute_pdf(data_lowq0)

    # Append results to lists
    prob_stat.append(prob_stat_i)
    bin_edges_stat.append(bin_edges_stat_i)
    prob_highq0.append(prob_highq0_i)
    bin_edges_highq0.append(bin_edges_highq0_i)
    prob_lowq0.append(prob_lowq0_i)
    bin_edges_lowq0.append(bin_edges_lowq0_i)

    # Compute and store bin widths
    bin_width_stat.append(bin_edges_stat_i[1] - bin_edges_stat_i[0])
    bin_width_highq0.append(bin_edges_highq0_i[1] - bin_edges_highq0_i[0])
    bin_width_lowq0.append(bin_edges_lowq0_i[1] - bin_edges_lowq0_i[0])

prob_stat = np.array(prob_stat)
prob_highq0 = np.array(prob_highq0)
prob_lowq0 = np.array(prob_lowq0)
bin_edges_stat = np.array(bin_edges_stat)
bin_edges_highq0 = np.array(bin_edges_highq0)
bin_edges_lowq0 = np.array(bin_edges_lowq0)
bin_width_stat = np.array(bin_width_stat)
bin_width_highq0 = np.array(bin_width_highq0)
bin_width_lowq0 = np.array(bin_width_lowq0)

# Create xarray Dataset
ds = xr.Dataset(
    {
        "prob_stationary": (["sim", "bin"], prob_stat),
        "prob_high_q0": (["sim", "bin"], prob_highq0),
        "prob_low_q0": (["sim", "bin"], prob_lowq0),
        "bin_edges_stationary": (["sim", "edge"], bin_edges_stat),
        "bin_edges_high_q0": (["sim", "edge"], bin_edges_highq0),
        "bin_edges_low_q0": (["sim", "edge"], bin_edges_lowq0),
        "bin_width_stationary": (["sim"], bin_width_stat),
        "bin_width_high_q0": (["sim"], bin_width_highq0),
        "bin_width_low_q0": (["sim"], bin_width_lowq0),
    },
    coords={
        "sim": sims,
        "bin": np.arange(prob_stat.shape[1]),
        "edge": np.arange(bin_edges_stat.shape[1]),
        "height": ("sim", heights),  # Add height as a coordinate
    },
)

# Save Dataset to NetCDF file
fsave = f"{dnc}{t0}_{t1}_vort_pdf_plot_params_{int(vort_all[0].z[h_idx])}m.nc"
ds.to_netcdf(fsave)