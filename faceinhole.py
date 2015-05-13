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
        imscene = scipy.misc.imread(imurl)[:, :, ::-1]
        # im = Image.open(imurl)
        # fgbg = cv2.createBackgroundSubtractorMOG()
        fgbg = cv2.BackgroundSubtractorMOG()

        while(1):
            ret, frame = cap.read()
            npframe = np.asarray(frame)
            print npframe.shape
            imscene = fill_to_shape(imscene, npframe.shape)
            fgmask = fgbg.apply(frame)
            newframe = immerge(npframe, imscene, fgmask, 255-fgmask)

            # vis2 = cv.CreateMat(h, w, cv.CV_32FC3)
            # vis0 = cv.fromarray(vis)
            # cv.CvtColor(vis0, vis2, cv.CV_GRAY2BGR)
            # cv2.imshow('frame',fgmask)
            cv2.imshow('frame', newframe)
            # cv2.imshow('frame',frame)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

def fill_to_shape(im, shape):
    """
    fill image to requested shape
    """
    # retim = np.zeros(shape)
    # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT

    
    return scipy.misc.imresize(im, shape)


def immerge(im1, im2, mask1, mask2):
    # scipy.misc.imresize
    print im1.shape
    print im2.shape
    print mask1.shape
    print mask2.shape
    print np.max(mask1)
    print np.min(mask1)
    print np.max(mask2)
    print np.min(mask2)
    # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT
    if len(im1.shape) != len(mask1.shape):
        if len(mask1.shape) == 2:
            masknew = np.zeros(im1.shape, dtype=mask1.dtype)
            masknew[:, :, 0] = mask1
            masknew[:, :, 1] = mask1
            masknew[:, :, 2] = mask1
            mask1 = masknew

    if len(im2.shape) != len(mask2.shape):
        if len(mask2.shape) == 2:
            masknew = np.zeros(im2.shape, dtype=mask2.dtype)
            masknew[:, :, 0] = mask2
            masknew[:, :, 1] = mask2
            masknew[:, :, 2] = mask2
            mask2 = masknew


    return (im1 * mask1 * 1.0/255) + (im2 * mask2 * 1.0/255)

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
