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
# import PIL
from PIL import Image
import scipy
import scipy.misc
import pygame
import pygame.locals
import yaml
import time
import cv
from expocomp import AutomaticExposureCompensation

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

        self.screen = pygame.display.set_mode(self.config['resolution'])         # vytvoření okna s nastavením jeho velikosti
        pygame.display.set_caption("Example")               # nastavení titulku okna
        
        self.background = pygame.Surface(self.screen.get_size())      # vytvoření vrstvy pozadí
        self.background = self.background.convert()                   # převod vrstvy do vhodného formátu
        self.background.fill((0,0,255))                 
        self.clock = pygame.time.Clock()                         # časování
        self.keepGoing = True 
        self.photo_number = 1
        self.camera_zoom = 1.0
        self.camera_offset = [0, 0]
        self.cap = cv2.VideoCapture(self.config['camera_source'])
        # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT
        # self.cap.set( cv.CV_CAP_PROP_EXPOSURE, 5)
        # print self.cap.get(cv.CV_CAP_PROP_CONTRAST)
        # print self.cap.get(cv.CV_CAP_PROP_BRIGHTNESS)
        #
        # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT
        self.cap.set(cv.CV_CAP_PROP_FRAME_WIDTH, 1024)
        self.cap.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 768)
        # cv.SetCaptureProperty(self.cap, cv.CV_CAP_PROP_EXPOSURE, 5)
        ret, frame = self.cap.read()
        self.aec = AutomaticExposureCompensation()
        self.aec.set_ref_image(frame)
        self.aec.set_area(-1, -1, -40, -40)

        self.camera_rgb2xyz = (
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0
            )
        try:
            self.watermark, self.watermark_offset = self.__read_surf(self.config['watermark'])
            # self.watermark_offset = self.config['watermark_offset']
            # pygame.image.load(self.config['watermark_file'])
        except:
            self.watermark = None

            # 0.412453, 0.357580, 0.180423, 0,
            # 0.212671, 0.715160, 0.072169, 0,
            # 0.019334, 0.119193, 0.950227, 0 )
        self.debugmode = 'N'

    def snapshot(self):
        filename = '{0:010x}'.format(int(time.time() * 256))[:10] + '.jpg'
        # self.screen = pygame.transform.flip(self.screen, True, False)
        # if self.watermark is not None:
        #     self.screen.blit(self.watermark, (0,0))                  # přidání pozadí k vykreslení na pozici 0, 0
        # pygame.display.flip()        
        pygame.image.save(self.screen, filename)
        email = inputbox.ask2(self.screen, "email")
        if email is not None:
            self.send_mail(filename, email)


    def send_mail(self, filename, email):
        print "Sending email to: " + email
        import send_photo_by_mail as spbm
        spbm.sendmail(
                filename, 
                addrs_to=email,
                addrs_cc=self.config['mail_cc'],
                addrs_from=self.config['mail_from'],
                preamble=self.config['mail_preamble'],
                subject=self.config['mail_subject'],

                )

    def __prepare_scene(self, photo_number):
        self.photo_number = photo_number
        info_scene = self.config['images'][photo_number]

        info_fore = info_scene['foreground']
        info_back = info_scene['background']

        self.imforeground, self.imforeground_offset = self.__read_surf(info_fore)
        # self.imforeground_offset = info_fore['offset']
        self.imbackground, self.imbackground_offset = self.__read_surf(info_back)
        # self.imbackground_offset = info_back['offset']
        self.camera_zoom = info_scene['camera_zoom']
        self.camera_offset = info_scene['camera_offset']
        self.camera_rgb2xyz = info_scene['camera_rgb2xyz']

    def __read_surf(self, info):

        if info is None or info == 'None':
            return None, None
        
        surface = pygame.image.load(info['impath'])
        if self.config['flip']:
            surface = pygame.transform.flip(surface, True, False)
        # pygame.transform.scale(surface)
        surface = pygame.transform.rotozoom(surface, 0, info['zoom'])
        return surface, info['offset']


    def __camera_image_processing(self, npframe):
# TODO
# http://effbot.org/zone/pil-sepia.htm
# http://stackoverflow.com/questions/3114925/pil-convert-rgb-image-to-a-specific-8-bit-palette

        pilim = Image.fromarray(npframe)
        # sepia_filter = np.array(
        #         [[.393, .769, .189],
        #         [.349, .686, .168],
        #         [.272, .534, .131]])
        # self.rgb2xyz = (
        #     0.412453, 0.357580, 0.180423, 0,
        #     0.212671, 0.715160, 0.072169, 0,
        #     0.019334, 0.119193, 0.950227, 0 )
        out = pilim.convert("RGB", self.camera_rgb2xyz)
        new_npframe = np.array(out)

        return new_npframe

    def __mask_processing(self, fgmask): 
        # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT

        # remove shadows
        cr = self.config['cut_right']
        cl = self.config['cut_left']
        fgmask[:, :cr] = 0 
        fgmask[:, -cl:] = 0 
        fgmask[fgmask < 128] = 0
        ks = self.config['erosion_kernel_size']
        kernel = np.ones((ks, ks), np.uint8)
        bs = self.config['blur_kernel_size']
        bsi = self.config['blur_sigma']
        fgmask = cv2.erode(fgmask, kernel)
        fgmask = cv2.GaussianBlur(fgmask, (bs, bs), bsi)

        return fgmask

    def run(self):

        self.__prepare_scene(1)
        self.fgbg = BackgroundSegmentation()

        while(self.keepGoing):
            self.tick()

        self.cap.release()
        cv2.destroyAllWindows()

    def tick(self):
        """One tick in run().

        """
        ret, frame = self.cap.read()
        npframe = np.asarray(frame)[:, :, ::-1]
        # imscene = fill_to_shape(imscene, npframe.shape)
        framec = self.aec.compensate(frame)
        print 'pred ', np.max(frame[:,:,0])
        print 'po   ', np.max(framec[:,:,0])
        fgmask_raw = self.fgbg.apply(framec)
        fgmask = self.__mask_processing(fgmask_raw)
        npframe = self.__camera_image_processing(npframe)


# novy s alphou
        npframer = np.rot90(npframe, 1)
        fgmaskr = np.rot90(fgmask, 1)
        sf_mid = make_surf_with_alpha(npframer, fgmaskr)
        sf_mid = pygame.transform.rotozoom(sf_mid, 0, self.camera_zoom)
        # pygame.display.set_mode(self.config['resolution'])
        
        if self.imbackground is not None:
            self.screen.blit(self.imbackground, self.imbackground_offset)                  # přidání pozadí k vykreslení na pozici 0, 0
        # self.screen.blit(sf_newframe, (0,0))                  # přidání pozadí k vykreslení na pozici 0, 0
        self.screen.blit(sf_mid, self.camera_offset)                  # přidání pozadí k vykreslení na pozici 0, 0
        if self.imforeground is not None:
            self.screen.blit(self.imforeground, self.imforeground_offset)                  # přidání pozadí k vykreslení na pozici 0, 0
        # self.screen.blit(imscene2, (0,0))                  # přidání pozadí k vykreslení na pozici 0, 0
        # self.screen.blit(self.background, (0,0))                  # přidání pozadí k vykreslení na pozici 0, 0
        # self.screen.blit(text, textRect)                     # přidání textu k vykreslení na střed

        if self.watermark is not None:
            self.screen.blit(self.watermark, self.watermark_offset)                  # přidání pozadí k vykreslení na pozici 0, 0
        if self.debugmode is 'D':
            self.screen.blit(makesurf(np.rot90(fgmask_raw, 1)), (0, 0))
        if self.debugmode is 'C':
            self.screen.blit(makesurf(np.rot90(npframe, 1)), (0, 0))
        if self.debugmode is 'CC':
            dframe = np.asarray(framec)[:, :, ::-1]
            self.screen.blit(makesurf(np.rot90(dframe, 1)), (0, 0))

        pygame.display.flip()        

        self.clock.tick(5)                                  # omezení maximálního počtu snímků za sekundu
        self.event_processing()

    def event_processing(self):
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
                elif event.key == pygame.locals.K_KP5:
                    self.__prepare_scene(5)
                elif event.key == pygame.locals.K_KP6:
                    self.__prepare_scene(6)
                elif event.key == pygame.locals.K_KP7:
                    self.__prepare_scene(7)
                elif event.key == pygame.locals.K_KP8:
                    self.__prepare_scene(8)
                elif event.key == pygame.locals.K_KP9:
                    self.__prepare_scene(9)
                elif event.key == pygame.locals.K_i:
                    print self.cap.get(cv.CV_CAP_PROP_MODE)
                    print self.cap.get(cv.CV_CAP_PROP_BRIGHTNESS)
                    print self.cap.get(cv.CV_CAP_PROP_CONTRAST)
                    print self.cap.get(cv.CV_CAP_PROP_SATURATION)
                    print self.cap.get(cv.CV_CAP_PROP_GAIN)
                    import ipdb; ipdb.set_trace() #  noqa BREAKPOINT

                elif event.key == pygame.locals.K_d:
                        self.debugmode = 'D' 
                elif event.key == pygame.locals.K_f:
                        self.debugmode = 'C' 
                elif event.key == pygame.locals.K_g:
                        self.debugmode = 'CC' 
                elif event.key == pygame.locals.K_n:
                        self.debugmode = 'N' 
                    # self.__prepare_scene(5)

class BackgroundSegmentation():
    def __init__(self):
        self.fgbg = cv2.BackgroundSubtractorMOG2(
                history=10,
                varThreshold=4.0,
                bShadowDetection=True)

        # self.fgbg = cv2.BackgroundSubtractorMOG2()
        # self.fgbg = cv2.BackgroundSubtractorMOG()

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
    if len(pixels.shape) == 2:
        px = np.zeros([pixels.shape[0], pixels.shape[1], 3], dtype=pixels.dtype)
        px[:,:,0] = pixels
        px[:,:,1] = pixels
        px[:,:,2] = pixels
        pixels = px
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
