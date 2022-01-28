#!/usr/bin/env python3

import argparse
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path, help="input file")
    parser.add_argument('output', type=Path, help="output file")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    asfloats = [list(map(float, s.strip().split())) for s in open(args.input)]
    maxlen = max(list(map(len, asfloats)))
    chembl_enc = np.zeros((len(asfloats), maxlen), dtype=np.int16)
    for i, x in enumerate(asfloats):
        chembl_enc[i,:len(x)] = x
    np.save(args.output, chembl_enc)
    return


if __name__ == "__main__":
    main()
    
