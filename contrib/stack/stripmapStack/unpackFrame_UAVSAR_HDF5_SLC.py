#!/usr/bin/env python3

import isce  # noqa
from isceobj.Sensor import createSensor
import shelve
import argparse
import os


def cmdLineParse():
    """
    Command line parser.
    """

    parser = argparse.ArgumentParser(
        description="Unpack UAVSAR SLC data and store metadata in pickle file."
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="h5_file",
        required=True,
        help="Input UAVSAR HDF5 file",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="slc_dir",
        required=True,
        help="Output SLC directory",
    )
    parser.add_argument(
        "-p",
        "--polarization",
        dest="polarization",
        default="VV",
        help="SLC polarization (default=%(default)s ) ",
    )
    parser.add_argument(
        "-f",
        "--frequency",
        default="A",
        choices=("A", "B"),
        help="NISAR frequency choices (choices = %(choices)s , default=%(default)s )",
    )
    return parser.parse_args()


def unpack(h5_file, slc_dir, frequency="A", polarization="VV"):
    """
    Unpack HDF5 to binary SLC file.
    """

    obj = createSensor("UAVSAR_HDF5_SLC")
    obj.configure()
    obj.hdf5 = h5_file
    obj.frequency = "frequency" + frequency
    obj.polarization = polarization

    if not os.path.isdir(slc_dir):
        os.mkdir(slc_dir)

    # obj.parse()
    date = os.path.basename(slc_dir)
    obj.output = os.path.join(slc_dir, date + ".slc")

    obj.extractImage()
    obj.frame.getImage().renderHdr()

    obj.extractDoppler()

    pickName = os.path.join(slc_dir, "data")
    with shelve.open(pickName) as db:
        db["frame"] = obj.frame


if __name__ == "__main__":
    """
    Main driver.
    """

    inps = cmdLineParse()
    inps.slc_dir.rstrip("/")
    inps.h5_file.rstrip("/")

    unpack(inps.h5_file, inps.slc_dir, inps.frequency, inps.polarization)
