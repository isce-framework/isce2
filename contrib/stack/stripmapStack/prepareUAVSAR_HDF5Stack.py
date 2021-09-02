#!/usr/bin/env python3

import os
import glob
import argparse

import isce  # noqa
import isceobj
import subprocess
import shelve


def get_cli_args():
    """
    Create command line parser.
    """

    parser = argparse.ArgumentParser(description="Prepare UAVSAR HDF5 SLC Stack files.")
    parser.add_argument(
        "-i",
        "--input-dir",
        dest="input_dir",
        required=True,
        help="Input UAVSAR HDF5 file",
    )
    parser.add_argument(
        "-o",
        "--output",
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


def write_xml(shelveFile, slcFile):
    with shelve.open(shelveFile, flag="r") as db:
        frame = db["frame"]

    length = frame.numberOfLines
    width = frame.numberOfSamples
    print(width, length)

    slc = isceobj.createSlcImage()
    slc.setWidth(width)
    slc.setLength(length)
    slc.filename = slcFile
    slc.setAccessMode("write")
    slc.renderHdr()
    slc.renderVRT()


def get_date(file):
    yyyymmdd = "20" + file.split("_")[4]
    return yyyymmdd


def main():
    """
    The main driver.
    """

    inps = get_cli_args()

    outputDir = os.path.abspath(inps.output)

    #######################################
    slc_files = glob.glob(os.path.join(inps.input_dir, "*.h5"))

    for h5_file in slc_files:
        imgDate = get_date(h5_file)
        print(imgDate)
        print(h5_file)
        imgDir = os.path.join(outputDir, imgDate)
        os.makedirs(imgDir, exist_ok=True)

        cmd = (
            "unpackFrame_UAVSAR_HDF5_SLC.py -i "
            + h5_file
            + " -p "
            + inps.polarization
            + " -f "
            + inps.frequency
            + " -o "
            + imgDir
        )
        print(cmd)
        subprocess.check_call(cmd, shell=True)

        slcFile = os.path.join(imgDir, imgDate + ".slc")

        # Now extract the correct pol SLC from the HDF5 file
        subdataset = "/science/LSAR/SLC/swaths"
        subdataset += "/frequency{}/{}".format(inps.frequency, inps.polarization)
        cmd = 'gdal_translate -of ISCE HDF5:"{fname}":"/{sds}" {out}'.format(
            fname=h5_file, sds=subdataset, out=slcFile
        )

        print(cmd)
        subprocess.check_call(cmd, shell=True)

        shelveFile = os.path.join(imgDir, "data")
        write_xml(shelveFile, slcFile)


if __name__ == "__main__":
    main()
