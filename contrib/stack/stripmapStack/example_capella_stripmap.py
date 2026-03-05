# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Capella Stripmap InSAR with ISCE2's stripmapStack
#
# End-to-end workflow for processing Capella Space stripmap SLC GeoTIFFs
# through ISCE2's stripmapStack processor to generate interferograms.
#
# ## Prerequisites
# - ISCE2 installed with Capella sensor support
# - Capella SLC GeoTIFFs (stripmap mode)
# - A DEM covering the area of interest (e.g. Copernicus DEM or 3DEP)
# - stripmapStack scripts on PATH (or run from source tree)

# %% [markdown]
# ## 1. Setup

# %%
import os
import glob
import subprocess
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ### Configure paths
#
# Update these paths to point to your data.

# %%
# Input Capella SLC GeoTIFFs
slc_tiff_dir = "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/capella/mexico_city/cropped_slcs"

# DEM file (GeoTIFF - will be converted to ISCE format)
dem_tiff = "/Volumes/WD_BLACK_SN7100_4TB/Documents/Learning/capella/mexico_city/dem_3dep.tif"

# Working directory for ISCE2 processing
work_dir = os.path.expanduser("~/repos/isce2/data_notebook")
os.makedirs(work_dir, exist_ok=True)

# Unpacked SLC output directory
slc_dir = os.path.join(work_dir, "SLC")
os.makedirs(slc_dir, exist_ok=True)

# Path to stripmapStack scripts (if not on PATH, set this to source tree)
script_dir = os.path.expanduser("~/repos/isce2/contrib/stack/stripmapStack")

# %%
# List available SLC TIFFs
tiffs = sorted(glob.glob(os.path.join(slc_tiff_dir, "CAPELLA*.tif")))
print(f"Found {len(tiffs)} Capella SLC TIFFs:")
for t in tiffs:
    print(f"  {os.path.basename(t)}")

# %% [markdown]
# ## 2. Prepare DEM
#
# Convert the GeoTIFF DEM to ISCE2 format using `dem_to_isce.py` or GDAL.
# The DEM must have `.xml` and `.vrt` sidecar files.

# %%
dem_isce = os.path.join(work_dir, "dem.wgs84")

if not os.path.exists(dem_isce + ".xml"):
    subprocess.run(
        ["gdal_translate", "-of", "ENVI", dem_tiff, dem_isce],
        check=True
    )
    # Create ISCE XML/VRT metadata
    subprocess.run(
        ["fixImageXml.py", "-f", "-i", dem_isce],
        check=True
    )
    print("DEM converted to ISCE format")
else:
    print("DEM already exists in ISCE format")

# %% [markdown]
# ## 3. Organize SLCs by date and unpack
#
# `prepSlcCapella.py` reads each TIFF's metadata to extract the acquisition
# date, moves files into date folders, and generates an unpacking run file.
#
# `unpackFrame_Capella.py` converts each GeoTIFF to ISCE2's binary SLC format
# and extracts orbit/timing metadata into a shelve file.

# %%
# Step 1: Organize TIFFs into date folders and generate unpack commands
os.chdir(work_dir)
subprocess.run(
    [
        "python", os.path.join(script_dir, "prepSlcCapella.py"),
        "-i", slc_tiff_dir,
        "-o", slc_dir,
    ],
    check=True
)

# %%
# Step 2: Run the unpack commands
run_file = os.path.join(work_dir, "run_unPackCapella")
if os.path.exists(run_file):
    with open(run_file) as f:
        cmds = f.read().strip().split("\n")

    print(f"Unpacking {len(cmds)} SLCs...")
    t0 = time.time()
    for cmd in cmds:
        print(f"  {cmd}")
        subprocess.run(cmd.split(), check=True)
    print(f"Unpacking done in {time.time() - t0:.1f}s")

# %%
# Verify unpacked SLCs
dates = sorted([d for d in os.listdir(slc_dir) if os.path.isdir(os.path.join(slc_dir, d))])
print(f"\nUnpacked {len(dates)} dates: {dates}")

# %% [markdown]
# ## 4. Generate stripmapStack run files
#
# `stackStripMap.py` creates the coregistration processing steps (run files).
# We use `--nofocus` since Capella data is already focused to SLC.

# %%
os.chdir(work_dir)
subprocess.run(
    [
        "python", os.path.join(script_dir, "stackStripMap.py"),
        "-s", slc_dir,
        "-d", dem_isce,
        "-w", work_dir,
        "-a", "1",
        "-r", "1",
        "--nofocus",
        "-W", "slc",
    ],
    check=True
)

# %%
# List generated run files
run_files = sorted(glob.glob(os.path.join(work_dir, "run_files", "run_*")))
print("Generated run files:")
for rf in run_files:
    print(f"  {os.path.basename(rf)}")

# %% [markdown]
# ## 5. Execute coregistration steps
#
# Run each step sequentially. Steps:
# 1. Reference geometry (topo)
# 2. Secondary focus/split (no-op for pre-focused SLCs)
# 3. Geo2rdr + coarse resampling
# 4. Refine secondary timing (ampcor) - **slowest step**
# 5. Invert misregistration
# 6. Fine resampling
# 7. Grid baseline
#
# **Note on step 4 (refineSecondaryTiming):** The default 60x60 ampcor grid
# can be very slow. For small test scenes, this is fine. For large scenes,
# you can reduce the grid by editing the config files to add `--nwa 10 --nwd 10`.

# %%
os.chdir(work_dir)
total_t0 = time.time()

for rf in run_files:
    step_name = os.path.basename(rf)
    print(f"\n{'='*60}")
    print(f"Running: {step_name}")
    print(f"{'='*60}")

    with open(rf) as f:
        cmds = f.read().strip().split("\n")

    t0 = time.time()
    for cmd in cmds:
        if cmd.strip():
            subprocess.run(cmd, shell=True, check=True)
    elapsed = time.time() - t0
    print(f"  {step_name} completed in {elapsed:.1f}s")

total_elapsed = time.time() - total_t0
print(f"\nAll steps completed in {total_elapsed:.1f}s")

# %% [markdown]
# ## 6. Generate interferograms
#
# Create interferograms for consecutive date pairs using `crossmul.py`.
#
# **Multi-looking ratio**: Capella stripmap has range pixel size ~0.617m and
# azimuth spacing ~1.2m. For approximately square pixels, use a 1:2 ratio
# (az:range looks). Example: 5 azimuth x 9 range gives ~5.8m x 5.6m pixels.

# %%
# Find all coregistered SLC dates
merged_dir = os.path.join(work_dir, "merged")
coreg_dir = os.path.join(merged_dir, "SLC")

# Reference date
ref_date = dates[0]
ref_slc = os.path.join(coreg_dir, ref_date, ref_date + ".slc")

all_dates = sorted(os.listdir(coreg_dir))
print(f"Coregistered dates: {all_dates}")

# %%
# Generate interferograms for consecutive pairs
igram_dir = os.path.join(work_dir, "interferograms")
os.makedirs(igram_dir, exist_ok=True)

pairs = list(zip(all_dates[:-1], all_dates[1:]))
print(f"Generating {len(pairs)} consecutive interferograms")

for date1, date2 in pairs:
    pair_dir = os.path.join(igram_dir, f"{date1}_{date2}")
    os.makedirs(pair_dir, exist_ok=True)

    slc1 = os.path.join(coreg_dir, date1, date1 + ".slc")
    slc2 = os.path.join(coreg_dir, date2, date2 + ".slc")

    out_prefix = os.path.join(pair_dir, f"{date1}_{date2}")

    cmd = [
        "python", os.path.join(script_dir, "crossmul.py"),
        "-m", slc1,
        "-s", slc2,
        "-o", out_prefix,
        "-a", "5",
        "-r", "9",
    ]
    print(f"  {date1} x {date2}")
    subprocess.run(cmd, check=True)

print("Interferogram generation complete")

# %% [markdown]
# ## 7. Visualize results
#
# Show phase and coherence for each interferogram pair.

# %%
def load_igram(pair_dir, pair_name):
    """Load interferogram and amplitude from crossmul output."""
    int_file = os.path.join(pair_dir, pair_name + ".int")
    amp_file = os.path.join(pair_dir, pair_name + ".amp")

    # Get dimensions from XML
    import xml.etree.ElementTree as ET
    xml_file = int_file + ".xml"
    tree = ET.parse(xml_file)
    root = tree.getroot()

    width = None
    length = None
    for prop in root.iter("property"):
        name = prop.get("name")
        if name == "width":
            width = int(prop.find("value").text)
        elif name == "length":
            length = int(prop.find("value").text)
    assert width and length, f"Could not read dimensions from {xml_file}"

    igram = np.fromfile(int_file, dtype=np.complex64).reshape(length, width)
    amp = np.fromfile(amp_file, dtype=np.float32).reshape(length, width * 2)

    return igram, amp, width, length


# %%
fig, axes = plt.subplots(len(pairs), 2, figsize=(14, 4 * len(pairs)))
if len(pairs) == 1:
    axes = axes[np.newaxis, :]

for idx, (date1, date2) in enumerate(pairs):
    pair_name = f"{date1}_{date2}"
    pair_dir = os.path.join(igram_dir, pair_name)

    igram, amp, width, length = load_igram(pair_dir, pair_name)

    # Phase
    phase = np.angle(igram)
    axes[idx, 0].imshow(phase, cmap="hsv", vmin=-np.pi, vmax=np.pi, aspect="auto")
    axes[idx, 0].set_title(f"Phase: {date1} - {date2}")
    axes[idx, 0].set_ylabel("Azimuth")

    # Coherence (|E[s1*s2]| / sqrt(E[|s1|^2] * E[|s2|^2]))
    # Simple magnitude-based estimate
    coherence = np.abs(igram) / (amp[:, :width] * amp[:, width:] + 1e-10)
    coherence = np.clip(coherence, 0, 1)
    im = axes[idx, 1].imshow(coherence, cmap="gray", vmin=0, vmax=1, aspect="auto")
    mean_coh = np.nanmean(coherence[coherence > 0])
    axes[idx, 1].set_title(f"Coherence: {date1} - {date2} (mean={mean_coh:.3f})")

plt.tight_layout()
plt.savefig(os.path.join(work_dir, "interferograms_overview.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Saved overview figure to interferograms_overview.png")

# %% [markdown]
# ## 8. Compare with sarlet (optional)
#
# If sarlet/isce3 results exist for the same pairs, load and compare coherence.

# %%
sarlet_dir = os.path.expanduser("~/repos/sarlet")
sarlet_output = os.path.join(sarlet_dir, "network_output2_mc")

if os.path.exists(sarlet_output):
    print("Found sarlet output, comparing coherence...")
    # sarlet interferograms are named by date pair
    for date1, date2 in pairs[:3]:  # Compare first 3 pairs
        sarlet_coh_file = glob.glob(
            os.path.join(sarlet_output, f"*{date1}*{date2}*", "coherence.tif")
        )
        if sarlet_coh_file:
            from osgeo import gdal
            ds = gdal.Open(sarlet_coh_file[0])
            sarlet_coh = ds.GetRasterBand(1).ReadAsArray()
            ds = None
            print(f"  {date1}-{date2}: sarlet mean coherence = {np.nanmean(sarlet_coh):.3f}")
else:
    print("No sarlet output found, skipping comparison")

# %% [markdown]
# ## Notes
#
# - **Coherence**: Expect 0.35-0.55 for 3-day Capella stripmap pairs over urban areas
# - **Performance**: For a 1000x1000 crop with 8 dates, total processing is ~2-3 minutes
# - **refineSecondaryTiming**: This step runs ampcor for sub-pixel coregistration.
#   The default grid is scaled to image size (max 10x10 windows). For very large scenes,
#   processing time scales with the number of ampcor windows.
# - **Multi-looking**: The example uses 5 azimuth x 9 range looks (~5.8m x 5.6m pixels).
#   Capella stripmap has azimuth spacing ~1.2m and range ~0.617m, so use ~1:2 (az:range)
#   ratio for approximately square pixels. More looks = better coherence estimation
#   but lower resolution.
