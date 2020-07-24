import errno
import os
import time
import sys
import random

import numpy as np
import cv2 as cv
from math import floor, ceil
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageChops  # Pillow

random.seed(1234)
CAPTCHA_FOLDER = "C:/Users/neggi/code/syrnia/captcha images/unmodified cv"
DEFAULT_UNFILTERED = "C:/Users/neggi/code/syrnia/captcha images/unmodified cv/1002 0.png"
DEFAULT_FILTERED = "C:/Users/neggi/code/syrnia/captcha images/denoised/0/1009 0.png"
BLANK = "C:/Users/neggi/code/syrnia/captcha images/other/blank.png"


def load_pil(filename=DEFAULT_UNFILTERED):
    im = Image.open(filename)
    return im.convert('RGB')


def load_cv(file_path=DEFAULT_UNFILTERED):
    im = cv.imread(file_path)
    return im


def load_cv_filtered(file_path=DEFAULT_FILTERED):
    im = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
    return im


def difference(im):
    blank = load_cv(BLANK)[..., 2]
    blank = blank - im[..., 2]
    blank[blank > 200] = 0
    return blank


def denoise_thick(filename=DEFAULT_UNFILTERED, number=None):
    im = load_cv(filename)
    diff = difference(im)
    blurred = cv.medianBlur(diff, 3)
    _, blurred = cv.threshold(blurred, blurred.max()/6, 255, cv.THRESH_BINARY)

    k1 = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    k2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4,4))
    k3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2,2))
    k4 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))

    # mask = cv.morphologyEx(diff, cv.MORPH_CLOSE, k1, borderType=cv.BORDER_CONSTANT, borderValue=0)
    # mask = cv.morphologyEx(mask, cv.MORPH_OPEN, (3,3), borderType=cv.BORDER_CONSTANT, borderValue=0)

    # points = cv.findNonZero(nuclei)
    # points = np.ma.array(points.reshape(points.shape[0], -1))
    #
    # p2 = np.zeros(points.shape[0])
    # for i in range(points.shape[0]):
    #     distance = points - points[i]
    #     closest = np.sort(np.hypot(distance.T[0], distance.T[1]))[:8]  # no sqrt: diff.T[0]**2 + diff.T[1]**2
    #     p2[i] = closest.mean()  # TODO: standardise p2, or try geometric mean? Different distance function?
    #
    # pmax = p2.max()
    # new_im = np.zeros((im.shape[0], im.shape[1]))
    # new_im[points.T[1], points.T[0]] = pmax - p2
    #
    # _, close_points = cv.threshold(new_im, pmax*.6, pmax, cv.THRESH_BINARY)
    #
    # dilated = cv.dilate(close_points, cv.getStructuringElement(cv.MORPH_ELLIPSE, (6, 6)))

    _, dilate_diff = cv.threshold(diff, diff.max()*.5, 255, cv.THRESH_BINARY)
    dilate_diff = cv.dilate(dilate_diff, k1)
    _, threshed = cv.threshold(im[:,:,2], im[:,:,2].max()*.4, 255, cv.THRESH_BINARY_INV)  # B/W version of captcha
    black = dilate_diff * threshed  # 4 numbers and background text, no noise

    black2 = cv.erode(black, (3,3))

    # closed = cv.morphologyEx(black, cv.MORPH_CLOSE, k3, borderType=cv.BORDER_CONSTANT, borderValue=0)
    # closed = cv.morphologyEx(closed, cv.MORPH_OPEN, k2)
    opened = cv.morphologyEx(black, cv.MORPH_OPEN, k2)
    opened = cv.morphologyEx(opened, cv.MORPH_CLOSE, k3)
    result = opened * black2
    o2 = cv.erode(black, k4, borderType=cv.BORDER_CONSTANT, borderValue=0)

    # print(filename, opened.mean())
    # print(result.mean())
    if number is not None:
        filename = number
    # cv.imshow(f'{filename} nuclei', black*255)
    # cv.imshow(f'{filename} diff', opened*255)
    # cv.imshow(f'{filename} final', opened*black*255)
    # cv.imshow(f'{filename} close', result*255)
    cv.imshow(f'{filename} mask', o2*255)
    # return result


def denoise_thin(filename=DEFAULT_UNFILTERED, number=None):
    im = load_cv(filename)
    diff = difference(im)
    blurred = cv.medianBlur(diff, 3)
    _, blurred = cv.threshold(blurred, blurred.max()/6, 255, cv.THRESH_BINARY)
    # k1 = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # mask = cv.morphologyEx(diff, cv.MORPH_CLOSE, k1, borderType=cv.BORDER_CONSTANT, borderValue=0)
    # mask = cv.morphologyEx(mask, cv.MORPH_OPEN, (3,3), borderType=cv.BORDER_CONSTANT, borderValue=0)

    _, nuclei = cv.threshold(diff, diff.max()*.9, 255, cv.THRESH_BINARY)
    points = cv.findNonZero(nuclei)
    points = np.ma.array(points.reshape(points.shape[0], -1))

    p2 = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        distance = points - points[i]
        closest = np.sort(np.hypot(distance.T[0], distance.T[1]))[:8]  # no sqrt: diff.T[0]**2 + diff.T[1]**2
        p2[i] = closest.mean()  # TODO: standardise p2, or try geometric mean? Different distance function?

    pmax = p2.max()
    new_im = np.zeros((im.shape[0], im.shape[1]))
    new_im[points.T[1], points.T[0]] = pmax - p2

    _, close_points = cv.threshold(new_im, pmax*.6, pmax, cv.THRESH_BINARY)

    dilated = cv.dilate(close_points, cv.getStructuringElement(cv.MORPH_ELLIPSE, (6, 6)))

    # _, dilate_diff = cv.threshold(diff, diff.max()*.5, 255, cv.THRESH_BINARY)
    # dilate_diff = cv.dilate(dilate_diff, k1)
    # _, threshed = cv.threshold(im[:,:,2], im[:,:,2].max()*.4, 255, cv.THRESH_BINARY_INV)
    # black = dilate_diff * threshed
    # k2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4,4))
    # k3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    # closed = cv.morphologyEx(black, cv.MORPH_CLOSE, k2)
    # closed = cv.morphologyEx(closed, cv.MORPH_OPEN, k2)
    # c2 = cv.morphologyEx(black, cv.MORPH_OPEN, k3)
    # opened = cv.morphologyEx(black, cv.MORPH_OPEN, k2)
    # # opened = cv.dilate(opened, k3) * black
    # med = cv.medianBlur(black, 5)

    # print(filename, opened.mean())
    if number is not None:
        filename = number
    # cv.imshow(f'{filename} nuclei', med*255)
    # cv.imshow(f'{filename} diff', black*255)
    # cv.imshow(f'{filename} mask', closed*255)
    # cv.imshow(f'{filename} close', opened*255)
    # cv.imshow(f'{filename} final', mask)

    # return opened


def preview():
    list_of_files = [file for file in os.listdir(CAPTCHA_FOLDER) if file[-4:] in (".png", ".PNG")]
    picks = random.choices(list_of_files, k=10)
    for file in picks:
        filename = os.path.join(CAPTCHA_FOLDER, file)
        im = cv.imread(filename)
        a = denoise_thick(filename)
        cv.imshow(f'{file} orig', im[:,:,2])
        cv.imshow(f'{file} a', a)
    cv.waitKey()


def main():
    list_of_files = [file for file in os.listdir(CAPTCHA_FOLDER) if file[-4:] in (".png", ".PNG")]
    picks = random.choices(list_of_files, k=3)
    for n, file in enumerate(picks):
        filename = os.path.join(CAPTCHA_FOLDER, file)
        im = cv.imread(filename)
        cv.imshow(f'{n} orig', im[:,:,2])
        denoise_thick(filename, n)
    cv.imshow('default', load_cv()[:,:,2])
    denoise_thick()

    cv.waitKey()

    # Need to: find a way to remove outlier pixels better (square distance?)
    # Then: dilate nucleus pixels, mask with dilated diff img
    # bada bing bada boom: image denoised
    # then split up into the 4 numbers. Hopefully build off the code already written previously


if __name__ == "__main__":
    main()
