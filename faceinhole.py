#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 mjirik <mjirik@hp-mjirik>
#
# Distributed under terms of the MIT license.

"""

"""

import logging
logger = logging.getLogger(__name__)
import argparse
import numpy as np
import cv2
from PIL import Image
import scipy
import scipy.misc


class FaceInHole():
    def run(self):
        cap = cv2.VideoCapture(0)
        imurl = 'images/plakat.jpg'

        # im = Image.open('images/mona.png')
        scipy.misc.imread(imurl)
        # im = Image.open(imurl)
        # fgbg = cv2.createBackgroundSubtractorMOG()
        fgbg = cv2.BackgroundSubtractorMOG()

        while(1):
            ret, frame = cap.read()

            fgmask = fgbg.apply(frame)
            # cv2_im = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
            # pil_im = Image.fromarray(cv2_im)

            cv2.imshow('frame',fgmask)
            # cv2.imshow('frame',frame)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

def immerge(im1, im2, mask1, mask2):
    # scipy.misc.imresize
    if len(im1.shape) != len(mask1.shape):
        if len(mask1.shape) == 2:
            masknew = np.array(im1.shape, dtype=mask1.dtype)
            masknew[:, :, 0] = mask1
            masknew[:, :, 1] = mask1
            masknew[:, :, 2] = mask1

    if len(im2.shape) != len(mask2.shape):
        if len(mask2.shape) == 2:
            masknew = np.array(im1.shape, dtype=mask2.dtype)
            masknew[:, :, 0] = mask2
            masknew[:, :, 1] = mask2
            masknew[:, :, 2] = mask2

    return im1 * mask1 + im2 * mask2

def loop():
    pass

def main():
    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    # create file handler which logs even debug messages
    # fh = logging.FileHandler('log.txt')
    # fh.setLevel(logging.DEBUG)
    # formatter = logging.Formatter(
    #     '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)
    # logger.debug('start')

    # input parser
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    # parser.add_argument(
    #     '-i', '--inputfile',
    #     default=None,
    #     # required=True,
    #     help='input file'
    # )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Debug mode')
    args = parser.parse_args()

    if args.debug:
        ch.setLevel(logging.DEBUG)

    fih = FaceInHole()
    fih.run()


if __name__ == "__main__":
    main()
