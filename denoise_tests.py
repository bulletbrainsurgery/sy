import errno
import os
import time
import sys
import random

import numpy as np
import cv2 as cv
from math import floor, ceil
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageChops  # Pillow
from scipy import ndimage

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

# Original denoise algorithm:


def difference(im):
    blank = load_cv(BLANK)[..., 2]
    blank = blank - im[..., 2]
    blank[blank > 200] = 0
    return blank


def show_edges(pic):
    p1 = pic.filter(ImageFilter.FIND_EDGES)
    # p1.show(title = "Edges shown")
    return p1


def contrast(pic):
    # pic = ImageEnhance.Sharpness(pic)
    # pic = pic.enhance(2)  # This is the line that does stuff, previous line just sets it up
    pic = ImageEnhance.Contrast(pic)
    pic = pic.enhance(3)  # This is the line that does stuff, previous line just sets it up
    # pic = pic.convert('1')  # Contrast only works with non-b/w images
    # p1.show(title = "Contrasted")
    return pic


def cluster(pic, mask_size=20):
    p1 = np.array(ImageOps.invert(pic.convert('L')))

    mask = p1 > p1.mean()
    label_im, nb_labels = ndimage.label(mask)
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    mask_size = sizes < mask_size
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0
    p2 = Image.fromarray(label_im * 255)
    result = ImageOps.invert(p2.convert('L'))
    # result.show(title = "Cluster filter")
    return result


def rank_filter(pic, level=7, size=3):
    if type(level) == float:
        level = int(level * (size**2))
    p1 = pic.filter(ImageFilter.RankFilter(size, level))
    # p1.show(title = "Rank filter applied")
    return p1


def denoise(pic):
    p1 = show_edges(pic)
    p2 = contrast(p1)
    p3 = cluster(p2, mask_size=80)
    p4 = rank_filter(p3, level=7)  # level 6 works better with thin fonts but makes thick fonts worse
    p5 = cluster(p4, mask_size=15)
    return p5


def make_blank():
    blank_filename = "C:/Users/neggi/code/syrnia/captcha images/other/blank.png"

    list_of_files = [file for file in os.listdir(CAPTCHA_FOLDER) if file[-4:] in (".png", ".PNG")]
    picks = list_of_files[::73]

    if os.path.isfile(blank_filename):
        blank = load_cv(blank_filename)
    else:
        print("No blank pic yet, making now")
        blank = load_cv(os.path.join(CAPTCHA_FOLDER, picks[0]))
    for file in picks:
        filename = os.path.join(CAPTCHA_FOLDER, file)
        im = load_cv(filename)
        blank = np.maximum(blank, im)
    # Image.fromarray(blank).show()
    # blank = cv.cvtColor(blank, cv.COLOR_BGR2RGB)
    cv.imwrite(blank_filename, blank)


def d2(filename=DEFAULT_UNFILTERED):
    im = load_cv(filename)  # "C:/Users/neggi/code/syrnia/captcha images/unmodified cv/1002 i.png"

    im = difference(im)

    # d2 = rank_filter(diff, level=7)
    # d2 = np.array(d2)
    # d2 = (d2 - d2.min())/(d2.max()-d2.min())*255
    # d3 = np.uint8(d2)
    # _, d3 = cv.threshold(d3, 110, 255, 0)
    # # dst = cv.GaussianBlur(diff3, (5, 5), 2, borderType=cv.BORDER_REFLECT_101)
    # dst = cv.dilate(d3, (3, 3), borderType=cv.BORDER_REFLECT_101) / 255.
    # # dst = cv.bitwise_not(dst)
    # cv.imshow('blurred', dst)
    # cv.waitKey(0)

    im2 = rank_filter(Image.fromarray(im), level=4)
    _, im2 = cv.threshold(np.array(im2), 25, 255, cv.THRESH_BINARY)

    cont, _ = cv.findContours(im2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    im2 = np.uint8(im2)
    cv.drawContours(im2, cont, -1, 200)
    im3 = cv.dilate(im, (3, 3), borderType=cv.BORDER_REFLECT_101)
    im3 = cv.erode(im3, cv.getStructuringElement(cv.MORPH_RECT, (3,3)), borderType=cv.BORDER_REFLECT_101)
    _, im3 = cv.threshold(im3, 50, 255, cv.THRESH_BINARY)
    # im2.show()
    # im3 = rank_filter(im2, level=3)
    # im3.show()
    # im4 = cv.Canny(im2, 250, 255, 0)

    # _, im4 = cv.threshold(np.uint8(im2), 100, 255, cv.THRESH_BINARY)
    cv.imshow('original', im)
    cv.imshow('bin open', im2)
    cv.imshow('rank filter', im3)
    # cv.imshow('canny', im4)
    cv.waitKey(0)
    return im


def d3(filename=DEFAULT_UNFILTERED):
    a = load_cv(filename)
    a = difference(a)
    cv.threshold(a, 20, 255, cv.THRESH_BINARY, a)
    b = cv.medianBlur(a, 3)
    # dilated = cv.dilate(b, (3, 3), borderType=cv.BORDER_REFLECT_101)
    eroded = cv.erode(a, cv.getStructuringElement(cv.MORPH_CROSS, (5, 4)), borderType=cv.BORDER_REFLECT_101)
    mask = np.ones((a.shape[0]+2, a.shape[1]+2))
    mask[1:-1, 1:-1] = np.invert(b)

    new_img = np.zeros((80, 150), dtype=np.uint8)
    points, _ = cv.findContours(eroded, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    points = list(map(lambda x: x[0][0], points))
    points = np.vstack(points)
    for point in points:
        _, new_img, _, _ = cv.floodFill(new_img, np.uint8(mask), tuple(point), 255)
    return new_img


def d4(filename=DEFAULT_FILTERED):
    a = load_cv(filename)
    a = difference(a)
    cv.threshold(a, 20, 255, cv.THRESH_BINARY, a)
    b = cv.medianBlur(a, 3)
    # dilated = cv.dilate(b, (3, 3), borderType=cv.BORDER_REFLECT_101)
    eroded = cv.erode(a, cv.getStructuringElement(cv.MORPH_CROSS, (4, 4)), borderType=cv.BORDER_REFLECT_101)
    mask = np.ones((a.shape[0]+2, a.shape[1]+2))
    mask[1:-1, 1:-1] = np.invert(b)

    new_img = np.zeros((80, 150), dtype=np.uint8)
    points, _ = cv.findContours(eroded, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    points = list(map(lambda x: x[0][0], points))
    points = np.vstack(points)
    for point in points:
        _, new_img, _, _ = cv.floodFill(new_img, np.uint8(mask), tuple(point), 255)
    return new_img


def draw_contours(pic):
    im = cv.Canny(cv.bitwise_not(pic), 250, 255, 0)
    # cv.imshow('d4', d4)
    contours, _ = cv.findContours(im, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(im, contours, -1, 127)
    cv.imshow('contours', im)
    cv.waitKey(0)


def denoise_all():
    list_of_files = [file for file in os.listdir(CAPTCHA_FOLDER) if file[-4:] in (".png", ".PNG")]
    denoised_path = "C:/Users/neggi/code/syrnia/captcha images/denoised cv"
    try:
        os.mkdir(denoised_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    current_prefix = 0
    for file in list_of_files:
        if current_prefix != file[0]:
            current_prefix = file[0]
            print(current_prefix, time.time())
        denoised_img = d3(os.path.join(CAPTCHA_FOLDER, file))
        cv.imwrite(os.path.join(denoised_path, file), denoised_img)


def denoise_compare(func_a=d3, func_b=d4):
    list_of_files = [file for file in os.listdir(CAPTCHA_FOLDER) if file[-4:] in (".png", ".PNG")]
    picks = random.choices(list_of_files, k=10)
    for file in picks:
        filename = os.path.join(CAPTCHA_FOLDER, file)
        a = func_a(filename)
        b = func_b(filename)
        cv.imshow(f'{file} a', a)
        cv.imshow(f'{file} b', b)
    cv.waitKey()


def test():
    # a = denoise_old("C:/Users/neggi/code/syrnia/captcha images/d.png")
    kernel = np.ones((3,3),np.uint8)
    im = load_cv()
    o = np.copy(im)[:,:,2]
    a = np.copy(im)[:,:,2]
    _, o = cv.threshold(o, 85, 255, cv.THRESH_BINARY_INV)
    thing = cv.morphologyEx(o, cv.MORPH_OPEN, kernel)
    thing2 = cv.erode(thing, cv.getStructuringElement(cv.MORPH_CROSS, (2,2)))

    im = difference(im)
    _, im = cv.threshold(im, 75, 255, cv.THRESH_BINARY)
    im = np.array(rank_filter(Image.fromarray(im), level=6))
    v_d1 = cv.dilate(o, cv.getStructuringElement(cv.MORPH_RECT, (1, 30)))
    v_e1 = cv.erode(v_d1, cv.getStructuringElement(cv.MORPH_RECT, (1, 30)))

    cont, _ = cv.findContours(thing2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cont = list(x for x in cont if cv.contourArea(x) > 15)
    asdf = np.zeros((80, 150))
    cv.drawContours(asdf, cont, -1, 200)

    cv.imshow('a', im)
    cv.imshow('t', thing)
    cv.imshow('o', o)
    cv.imshow('b', asdf)
    cv.imshow('c', a)
    cv.waitKey()

    # Looks cool: maybe do voronoi stuff in future
    # pretty = cv.distanceTransform(cv.bitwise_not(diff), cv.DIST_L2, cv.DIST_MASK_PRECISE)
    # print(pretty)
    # cv.imshow('test', pretty*3/pretty.max())
    # a = rank_filter(Image.fromarray(diff), 22, 5)


def denoise_best():
    im = load_cv()
    pic = difference(im)
    _, nuclei = cv.threshold(pic, pic.max()*.9, 255, cv.THRESH_BINARY)

    points = cv.findNonZero(nuclei)
    points = np.ma.array(points.reshape(points.shape[0], -1))

    p2 = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        diff = points - points[i]
        closest = np.sort(np.hypot(diff.T[0], diff.T[1]))[:8]  # no sqrt: diff.T[0]**2 + diff.T[1]**2
        p2[i] = closest.mean()  # TODO: standardise p2, or try geometric mean? Different distance function?

    pmax = p2.max()
    new_im = np.zeros((im.shape[0], im.shape[1]))
    new_im[points.T[1], points.T[0]] = pmax - p2

    _, b = cv.threshold(new_im, pmax*.6, pmax, cv.THRESH_BINARY)

    blurred = cv.medianBlur(pic, 3)
    _, blurred = cv.threshold(blurred, blurred.max()/5, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    mask = cv.morphologyEx(pic, cv.MORPH_CLOSE, kernel, borderType=cv.BORDER_CONSTANT, borderValue=0)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, (3,3), borderType=cv.BORDER_CONSTANT, borderValue=0)

    dilated = cv.dilate(b, cv.getStructuringElement(cv.MORPH_ELLIPSE, (6, 6)))

    # cv.imshow('c', dilated)
    # cv.imshow('pic', pic)
    # cv.imshow('mask', mask)
    # cv.imshow('final', mask * dilated)
    # cv.waitKey()
    return mask * dilated


def main():
    pass


if __name__ == "__main__":
    main()
