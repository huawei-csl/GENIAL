#!/usr/bin/env python3
"""Calibrate switching activity to power using linear regression."""

import csv
import sys
import numpy as np


def load_data(path):
    swact = []
    power = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            swact.append(float(row["swact"]))
            power.append(float(row["power"]))
    return np.array(swact), np.array(power)


def calibrate(swact, power):
    a = np.vstack([swact, np.ones(len(swact))]).T
    coeffs, _, _, _ = np.linalg.lstsq(a, power, rcond=None)
    return coeffs[0], coeffs[1]


def main():
    if len(sys.argv) != 2:
        print("Usage: swact_calibration.py <data.csv>")
        sys.exit(1)
    swact, power = load_data(sys.argv[1])
    a, b = calibrate(swact, power)
    print("a = {:.6f}".format(a))
    print("b = {:.6f}".format(b))


if __name__ == "__main__":
    main()
