"""Regression tests for the TOPS Stack ESD response."""
from types import SimpleNamespace

import numpy as np
import pytest
import Stack
import estimateAzimuthMisreg as estimate
import overlap_withDEM as overlap


@pytest.mark.parametrize("stack_reference", [None, 0, 1])
def test_original_four_blocks_match_carrier_phase_derivatives(
    tmp_path, stack_reference
):
    from datetime import datetime, timedelta

    start = datetime(2025, 1, 1)
    shape, dt = (3, 4), 0.002
    sources = []
    paths = []
    for index in range(4):
        doppler = 30.0 + index * 7
        sources.append(
            SimpleNamespace(
                sensingStart=start + timedelta(seconds=(index % 2) * 1360 * dt),
                sensingMid=start,
                firstValidLine=5,
                numberOfLines=1504,
                startingRange=800000,
                rangePixelSize=2.3,
                azimuthTimeInterval=dt,
                azimuthSteeringRate=0.028,
                radarWavelength=0.056,
                orbit=SimpleNamespace(
                    interpolateOrbit=lambda *a, **kw: SimpleNamespace(
                        getVelocity=lambda: [7000, 0, 0]
                    )
                ),
                doppler=lambda r, d=doppler: d + (np.asarray(r) - 800000) * 0.001,
                azimuthFMRate=lambda r: -2500 + np.asarray(r) * 0.0001,
            )
        )
        for direction, value in [("az", 5 if index % 2 else 1365), ("rg", 20)]:
            path = tmp_path / ("%s_%s.off" % (index, direction))
            if index // 2 != stack_reference:
                np.full(shape, value, dtype=np.float32).tofile(path)
            paths.append(str(path))
    crop = SimpleNamespace(
        sensingStart=start + timedelta(seconds=1365 * dt),
        startingRange=800046,
        numberOfLines=3,
        numberOfSamples=4,
        azimuthTimeInterval=dt,
    )
    shifts = [-1.0, 2.0]
    actual = overlap.overlapSpectralSeparation(
        crop,
        crop,
        *sources,
        *paths,
        referenceShift=shifts[0],
        secondaryShift=shifts[1],
        referenceIsStack=stack_reference == 0,
        secondaryIsStack=stack_reference == 1,
    )
    numeric = []
    for i, source in enumerate(sources):
        y = np.arange(3)[:, None] + (5 if i % 2 else 1365)
        r = 800000 + (np.arange(4)[None, :] + 20) * 2.3
        dop, fm = source.doppler(r), source.azimuthFMRate(r)
        steering = 2 * 7000 * 0.028 / 0.056
        rate = steering / (1 - steering / fm)
        eta_ref = source.doppler(800000) / source.azimuthFMRate(800000) - dop / fm

        def phase(row):
            return (
                np.pi * rate * ((row - 752 + shifts[i // 2]) * dt - eta_ref) ** 2
                + 2 * np.pi * dop * row * dt
            )

        h = 1e-4
        numeric.append((phase(y + h) - phase(y - h)) / (2 * h))
    np.testing.assert_allclose(
        actual, (numeric[0] - numeric[1] + numeric[2] - numeric[3]) / 2, rtol=1e-7
    )


def test_generated_config_reaches_existing_parsers(tmp_path):
    from SentinelWrapper import ConfigParser

    filename = tmp_path / "config"
    cfg = Stack.config(str(filename))
    cfg.configure(
        SimpleNamespace(
            work_dir=str(tmp_path),
            referenceDir=str(tmp_path / "first"),
            secondaryDir=str(tmp_path / "second"),
            interferogramDir=str(tmp_path / "ifg"),
            overlapDir=str(tmp_path / "ESD"),
            misregFile=str(tmp_path / "misreg.txt"),
            esdCoherenceThreshold="0.85",
        )
    )
    cfg.overlap_withDEM("[Function-1]")
    cfg.azimuthMisreg("[Function-2]")
    cfg.finalize()
    parser = ConfigParser(str(filename), [], {})
    parser.readConfig()
    first = overlap.cmdLineParse(parser.funcParams["Function-1"])
    second = estimate.cmdLineParse(parser.funcParams["Function-2"])
    assert first.stack_reference_dir == str(tmp_path / "reference")
    assert second.esdCoherenceThreshold == 0.85
    assert (first.esdAzimuthLooks, first.esdRangeLooks) == (
        second.esdAzimuthLooks,
        second.esdRangeLooks,
    )
