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
import pygame
import pygame.locals
import yaml
import time

import inputbox


class FaceInHole():
    def __init__(self, conf_file='config.yml'):
        """TODO: Docstring for __init__.
        :returns: TODO

        """
# config
        stream = open(conf_file, 'r')
        self.config = yaml.load(stream)

        pygame.init()

        self.screen = pygame.display.set_mode((640,480))         # vytvoření okna s nastavením jeho velikosti
        pygame.display.set_caption("Example")               # nastavení titulku okna
        
        self.background = pygame.Surface(self.screen.get_size())      # vytvoření vrstvy pozadí
        self.background = self.background.convert()                   # převod vrstvy do vhodného formátu
        self.background.fill((0,0,255))                 
        self.clock = pygame.time.Clock()                         # časování
        self.keepGoing = True 
        self.photo_number = 1
        self.camera_zoom = 1.0
        self.camera_offset = [0, 0]

    def snapshot(self):
        filename = '{0:010x}'.format(int(time.time() * 256))[:10] + '.jpg'
        pygame.image.save(self.screen, filename)
        email = inputbox.ask2(self.screen, "email")
        if email is not None:
            self.send_mail(filename, email)

    def send_mail(self, filename, email):
        print "Sending email to: " + email

    def __prepare_scene(self, photo_number):
        self.photo_number = photo_number
        info_scene = self.config['images'][photo_number]

        info_fore = info_scene['foreground']
        info_back = info_scene['background']

        self.imforeground = self.__read_surf(info_fore)
        self.imbackground = self.__read_surf(info_back)
        self.camera_zoom = info_scene['camera_zoom']
        self.camera_offset = info_scene['camera_offset']

    def __read_surf(self, info):

        if info is None or info == 'None':
            return None
        
        surface = pygame.image.load(info['impath'])
        # pygame.transform.scale(surface)
        surface = pygame.transform.rotozoom(surface, 0, info['zoom'])
        return surface



    def __mask_processing(self, fgmask): 
        # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT

        kernel = np.ones((5,5), np.uint8)
        # fgmask = cv2.erode(fgmask, kernel)
        # fgmask = cv2.GaussianBlur(fgmask, (21, 21), 7)
        return fgmask

    def run(self):
        cap = cv2.VideoCapture(0)
        imurl = 'images/plakat_full.jpg'
        imurl2 = 'images/D6-12_small.png'
        imscene = scipy.misc.imread(imurl)# [:, :, ::-1]
        imscene2 = pygame.image.load(imurl2)

        self.__prepare_scene(1)
        # im = Image.open(imurl)
        # fgbg = cv2.createBackgroundSubtractorMOG()
        fgbg = BackgroundSegmentation()

        while(self.keepGoing):
            ret, frame = cap.read()
            npframe = np.asarray(frame)[:, :, ::-1]
            imscene = fill_to_shape(imscene, npframe.shape)
            fgmask = fgbg.apply(frame)
            fgmask = self.__mask_processing(fgmask)

            # print npframe.shape
            # print 'mask'
            # print np.max(fgmask)
            # print np.min(fgmask)
            # newframe = immerge(npframe, imscene, fgmask, 255-fgmask)

            # # cv2.imshow('frame',fgmask)
            # cv2.imshow('frame', newframe)
            # # cv2.imshow('frame', imscene)
            # # cv2.imshow('frame',frame)
            # k = cv2.waitKey(30) & 0xff
            # if k == 27:
            #     break
        
            # scipy.misc.imresize(new_frame, )
            # backp = pygame.surfarray.pixels2d(self.background)

            # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT
            # newframe = np.rot90(newframe, 1)
            # sf_newframe = makesurf(newframe)

# novy s alphou
            npframer = np.rot90(npframe, 1)
            fgmaskr = np.rot90(fgmask, 1)
            sf_mid = make_surf_with_alpha(npframer, fgmaskr)
            sf_mid = pygame.transform.rotozoom(sf_mid, 0, self.camera_zoom)
            
            if self.imbackground is not None:
                self.screen.blit(self.imbackground, (0,0))                  # přidání pozadí k vykreslení na pozici 0, 0
            # self.screen.blit(sf_newframe, (0,0))                  # přidání pozadí k vykreslení na pozici 0, 0
            self.screen.blit(sf_mid, self.camera_offset)                  # přidání pozadí k vykreslení na pozici 0, 0
            if self.imforeground is not None:
                self.screen.blit(self.imforeground, (0,0))                  # přidání pozadí k vykreslení na pozici 0, 0
            # self.screen.blit(imscene2, (0,0))                  # přidání pozadí k vykreslení na pozici 0, 0
            # self.screen.blit(self.background, (0,0))                  # přidání pozadí k vykreslení na pozici 0, 0
            # self.screen.blit(text, textRect)                     # přidání textu k vykreslení na střed
            pygame.display.flip()        

            self.clock.tick(5)                                  # omezení maximálního počtu snímků za sekundu

            for event in pygame.event.get():
                # any other key event input
                if event.type == pygame.locals.QUIT:
                    done = True        
                elif event.type == pygame.locals.KEYDOWN:
                    if event.key == pygame.locals.K_ESCAPE:
                        self.keepGoing = False
                    elif event.key == pygame.locals.K_1:
                        print "hi world mode"


                    # if event.key == pygame.K_ESCAPE:
                    #     self.keepGoing = False                       # ukončení hlavní smyčky
                    elif event.key == pygame.locals.K_SPACE:
                        self.snapshot()
                    elif event.key == pygame.locals.K_KP0:
                        self.__prepare_scene(0)
                    elif event.key == pygame.locals.K_KP1:
                        self.__prepare_scene(1)
                    elif event.key == pygame.locals.K_KP2:
                        self.__prepare_scene(2)
                    elif event.key == pygame.locals.K_KP3:
                        self.__prepare_scene(3)
                    elif event.key == pygame.locals.K_KP4:
                        self.__prepare_scene(4)

        cap.release()
        cv2.destroyAllWindows()

class BackgroundSegmentation():
    def __init__(self):
        self.fgbg = cv2.BackgroundSubtractorMOG2()

    def apply(self, frame):
        seg = self.fgbg.apply(frame)

        return seg


def fill_to_shape(im, shape):
    """
    fill image to requested shape
    """
    # retim = np.zeros(shape)
    # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT

    
    return scipy.misc.imresize(im, shape)


def immerge(im1, im2, mask1, mask2):
    # scipy.misc.imresize
    if len(im1.shape) != len(mask1.shape):
        print "mask is binary"
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


    retim = (im1 * (mask1 / 255.0)) + (im2 * (mask2 / 255.0))
    print 'retim'
    print np.max(retim)
    print np.min(retim)
    # print retim.dtype
    return retim.astype(np.uint8)

def loop():
    pass

def make_surf_with_alpha(pixels, fmask):
    (width, height, colours) = pixels.shape
    surf = pygame.display.set_mode((width, height))
    surf = pygame.Surface((width, height), flags=pygame.SRCALPHA)
    pygame.surfarray.blit_array(surf, pixels)
    # surf = makesurf(frame)
    # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT
    #
    alpha = pygame.surfarray.pixels_alpha(surf)
    alpha[...] = fmask
    del alpha

    return surf

def makesurf(pixels):
    try:
        surf = pygame.surfarray.make_surface(pixels)
    except IndexError:
        (width, height, colours) = pixels.shape
        surf = pygame.display.set_mode((width, height))
        pygame.surfarray.blit_array(surf, pixels)
    return surf
    

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
