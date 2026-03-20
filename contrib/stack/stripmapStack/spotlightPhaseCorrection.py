#!/usr/bin/env python3

"""
Apply phase correction to Capella spotlight SLCs for interferometry.

Capella spotlight SLCs are delivered deramped (baseband). After coregistration
to a common reference geometry, a geometric phase must be restored for
interferometric processing.

The correction phase for each pixel is:
    phi = -4*pi/lambda * (|R_ant - target| - |R_ant - R_target|)

where:
    R_ant    = reference antenna position (ECEF, from metadata)
    R_target = scene reference target position (ECEF, from metadata)
    target   = each pixel's ECEF position (computed from lon/lat/hgt geometry)

Each SLC (reference and all secondaries) gets its own correction using its
own antenna/target positions from the shelve data.
"""

import argparse
import os
import shelve

import numpy as np

import isce
import isceobj
from isceobj.Planet.AstronomicalHandbook import Const


def createParser():
    parser = argparse.ArgumentParser(
        description='Apply spotlight phase correction to coregistered Capella SLCs'
    )
    parser.add_argument(
        '-s', '--slc', type=str, required=True,
        help='Path to SLC file (without .xml extension)'
    )
    parser.add_argument(
        '-g', '--geom', type=str, required=True,
        help='Geometry directory containing lat.rdr, lon.rdr, hgt.rdr'
    )
    parser.add_argument(
        '-d', '--shelve', type=str, required=True,
        help='Path to shelve data file containing frame with spotlight metadata'
    )
    parser.add_argument(
        '-o', '--output', type=str, default=None,
        help='Output corrected SLC path (default: overwrite input)'
    )
    parser.add_argument(
        '--block-size', type=int, default=512,
        help='Number of lines to process per block (default: 512)'
    )
    return parser


def cmdLineParse(iargs=None):
    return createParser().parse_args(args=iargs)


def llh_to_ecef(lon_deg, lat_deg, height):
    """
    Convert geodetic lon/lat/height to ECEF XYZ coordinates (WGS84).

    Parameters
    ----------
    lon_deg : ndarray, longitude in degrees
    lat_deg : ndarray, latitude in degrees
    height  : ndarray, height above ellipsoid in meters

    Returns
    -------
    x, y, z : ndarray, ECEF coordinates in meters
    """
    # WGS84 parameters
    a = 6378137.0  # semi-major axis
    f = 1.0 / 298.257223563  # flattening
    e2 = 2 * f - f * f  # eccentricity squared

    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    N = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)

    x = (N + height) * cos_lat * cos_lon
    y = (N + height) * cos_lat * sin_lon
    z = (N * (1.0 - e2) + height) * sin_lat

    return x, y, z


def compute_correction_phase(lon, lat, hgt, ref_antenna_pos, ref_target_pos, wavelength):
    """
    Compute the phase correction for spotlight re-ramping.

    phi = -4*pi/lambda * (|R_ant - pixel_ecef| - |R_ant - R_target|)

    Parameters
    ----------
    lon, lat, hgt : ndarray (nlines, nsamples) - pixel coordinates
    ref_antenna_pos : list [x, y, z] ECEF meters
    ref_target_pos  : list [x, y, z] ECEF meters
    wavelength : float, meters

    Returns
    -------
    phase : ndarray (nlines, nsamples), radians
    """
    # Pixel ECEF positions
    px, py, pz = llh_to_ecef(lon, lat, hgt)

    # Reference antenna position
    ax, ay, az = ref_antenna_pos

    # Distance from antenna to each pixel
    dx = px - ax
    dy = py - ay
    dz = pz - az
    dist_ant_pixel = np.sqrt(dx * dx + dy * dy + dz * dz)

    # Distance from antenna to reference target (scalar)
    tx, ty, tz = ref_target_pos
    dist_ant_target = np.sqrt(
        (ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2
    )

    phase = (-4.0 * np.pi / wavelength) * (dist_ant_pixel - dist_ant_target)
    return phase


def main(iargs=None):
    inps = cmdLineParse(iargs)

    # Load frame from shelve to get spotlight metadata
    with shelve.open(inps.shelve, flag='r') as db:
        frame = db['frame']

    ref_ant = frame.spotlightReferenceAntennaPosition
    ref_tgt = frame.spotlightReferenceTargetPosition
    wavelength = frame.getInstrument().getRadarWavelength()

    print(f'Wavelength: {wavelength:.6f} m')
    print(f'Reference antenna position: {ref_ant}')
    print(f'Reference target position: {ref_tgt}')

    # Load SLC image dimensions (prefer XML, fall back to VRT via GDAL)
    xml_path = inps.slc + '.xml'
    vrt_path = inps.slc + '.vrt'
    if os.path.exists(xml_path):
        slc_img = isceobj.createImage()
        slc_img.load(xml_path)
        width = slc_img.width
        length = slc_img.length
    else:
        from osgeo import gdal
        ds = gdal.Open(vrt_path, gdal.GA_ReadOnly)
        assert ds is not None, f'Cannot open {vrt_path}'
        width = ds.RasterXSize
        length = ds.RasterYSize
        ds = None

    print(f'SLC dimensions: {length} x {width}')

    # Load geometry (DOUBLE precision)
    lat_data = np.fromfile(
        os.path.join(inps.geom, 'lat.rdr'), dtype=np.float64
    ).reshape(length, width)
    lon_data = np.fromfile(
        os.path.join(inps.geom, 'lon.rdr'), dtype=np.float64
    ).reshape(length, width)
    hgt_data = np.fromfile(
        os.path.join(inps.geom, 'hgt.rdr'), dtype=np.float64
    ).reshape(length, width)

    # Load SLC data
    slc_data = np.fromfile(inps.slc, dtype=np.complex64).reshape(length, width)

    output_path = inps.output if inps.output else inps.slc

    # Process in blocks for memory efficiency
    block_size = inps.block_size
    for i0 in range(0, length, block_size):
        i1 = min(i0 + block_size, length)

        phase = compute_correction_phase(
            lon_data[i0:i1],
            lat_data[i0:i1],
            hgt_data[i0:i1],
            ref_ant, ref_tgt, wavelength
        )

        # Apply phase correction: multiply by exp(-j*phi)
        slc_data[i0:i1] *= np.exp(-1j * phase).astype(np.complex64)

        if (i0 // block_size) % 10 == 0:
            print(f'  Processed lines {i0}-{i1} / {length}')

    # Write corrected SLC
    slc_data.tofile(output_path)
    print(f'Wrote corrected SLC to: {output_path}')

    # Update XML if writing to a new file
    if output_path != inps.slc:
        out_img = isceobj.createSlcImage()
        out_img.setByteOrder('l')
        out_img.setFilename(output_path)
        out_img.setAccessMode('read')
        out_img.setWidth(width)
        out_img.setLength(length)
        out_img.renderHdr()


if __name__ == '__main__':
    main()
