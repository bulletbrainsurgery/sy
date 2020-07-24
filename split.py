import numpy as np
import cv2 as cv
import sys
import os
import random
from math import floor, ceil
from PIL import Image
import time

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


def find_borders(im):
    # Default pics are black on white, function requires white on black so invert
    contours, hierarchy = cv.findContours(np.invert(im), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # RETR_TREE also gives contours of internal shapes (holes inside 04689)

    if len(hierarchy) != 1:
        print("Exception: hierarchy looks different. Uh oh")
        # TODO: add error
        return None, im
    hierarchy = hierarchy[0]

    reshaped = []
    for n, c in enumerate(contours):  # Currently c has X,1,Y shape
        if hierarchy[n][3] == -1:  # Keep if hierarchy shows no parents; remove internal holes
            # if cv.contourArea(c) >= 10:  # Remove super small stuff
            d = c.reshape(c.shape[0], c.shape[2])  # Reshape to X,Y
            reshaped.append(d)  # Make new list, modifying same list doesn't work somehow

    if len(reshaped) < 4:
        return None  # TODO: add error

    borders = list(map(cv.boundingRect, reshaped))  # map is cool :)
    # print(f"number of regions: {len(borders)}")

    modified = True
    while modified:
        modified = False
        for n, i in enumerate(borders):
            if i[2] > 30:  # if x size > 30 then split left-right
                modified = True
                b = borders.pop(n)
                borders.append((b[0], b[1], b[2] // 2, b[3]))
                borders.append((b[0] + b[2] // 2, b[1], b[2] // 2, b[3]))
                break
            if i[3] > 35:
                modified = True
                b = borders.pop(n)
                borders.append((b[0], b[1], b[2], b[3] // 2))
                borders.append((b[0], b[1] + b[3] // 2, b[2], b[3] // 2))
                break

    for c in borders:
        cv.rectangle(im, c, color=1)
    cv.imshow('', im)
    cv.waitKey()
    # cv.destroyAllWindows()
    return borders


def rect_size(b):
    return b[2] * b[3]


def filter_borders(borders, im):
    if borders is None:
        return  # TODO: raise error

    modified = True
    while modified:
        modified = False
        borders.sort(key=rect_size, reverse=True)
        for n, b in enumerate(borders[:4]):
            b_x = b[0] + b[2]
            b_y = b[1] + b[3]

            for m, c in enumerate(borders):  # TODO: order by closest X coordinate?
                c_x = c[0] + c[2]
                c_y = c[1] + c[3]

                # Constrain c size between 60 and 200 to avoid all sorts of issues
                if 60 < rect_size(c)[0] < 200:

                    # If a corner of c is close enough to b then add them together
                    if (b[0] - 2 < c[0] < (b[0] + b_x) // 2 or (b[0] + b_x) // 2 < c_x < b_x + 2) and \
                            (b[1] < c[1] < b_y + 5 or b[1] - 5 < c_y < b_y):
                        modified = True

                        # Remove old borders, add the new one
                        # print("removing old border:", b)
                        borders.pop(n)
                        # print("removing old border:", c)
                        borders.pop(m - 1)  # removed one so the index is one lower

                        # print("adding new border")
                        x_start = min(b[0], c[0])
                        x_end = max(b_x, c_x)
                        y_start = min(b[1], c[1])
                        y_end = max(b_y, c_y)
                        borders.append(x_start, y_start, x_end - x_start, y_end - y_start)
                        break
            if modified:
                # print("modified one thing, breaking loop")
                break
        if not modified:
            break

    borders = borders[:4]  # Take the largest 4 regions
    # TODO: delete if border is smaller than a certain size
    # >20 tall (ones) or area>100 (overlapping numbers are smaller than 200 probably)

    borders.sort(key=lambda x: x[0])  # Sort by left side of rect

    numbers = []

    for b in borders:
        # area = rect_size(b)
        # print(b)  # Location
        # print(f"{b[2]}x{b[3]}={area}")
        numbers.append(im[b[1]:b[1] + b[3], b[0]:b[0] + b[2]])
        # cv.rectangle(im, tuple(b[0]), tuple(b[1]), 1)
    # for n in numbers:
    # cv.imshow('',n)
    # cv.waitKey(0)

    return numbers


def save_digits(filename):
    base_dir = os.path.abspath(os.getcwd())
    file_path = os.path.join(base_dir, filename)

    im = load_cv_filtered(file_path)
    borders = find_borders(im)
    pictures = filter_borders(borders, im)

    if len(pictures) != 4:
        print("Incorrect amount of borders found: ", len(pictures))
        print("Filename:", filename)
        return False

    numbers = tuple(int(i) for i in filename[:4])  # tuple of the 4 numbers
    # print(numbers)

    for n, i in enumerate(numbers):
        # Pad picture to 22x30
        dims = (22, 30)
        dim_x, dim_y = dims

        p1 = pictures[n]
        y, x = p1.shape
        p2 = p1[max(y - dim_y, 0):, max(x - dim_x, 0):]  # cut the first few pixels to get it down to 22x30

        # Pad the picture with a random amount on either side
        y, x = p2.shape
        y_pad = dim_y - y
        x_pad = dim_x - x
        y_rand = random.randint(0, y_pad)
        x_rand = random.randint(0, x_pad)
        p3 = np.pad(p2, [(y_rand, y_pad - y_rand), (x_rand, x_pad - x_rand)], mode='constant', constant_values=255)

        newdir = os.path.join(base_dir, str(i))
        os.chdir(newdir)
        no_of_files = len([file for file in os.listdir() if os.path.isfile(file)])
        p4 = Image.fromarray(p3)
        p4 = p4.convert("1")
        p4.save(os.path.join(newdir, f'{no_of_files}.png'))
        os.chdir(base_dir)
    return True


def show_digits(im, orig_img=None):
    borders = find_borders(im)
    output_numbers = filter_borders(borders, im)

    # cv.imshow('Original picture',loaded_pic)
    # cv.waitKey(0)

    x_pad = im.shape[1] // 4
    y_pad = im.shape[0]
    concat_numbers = np.zeros((y_pad, 1))
    for i in output_numbers:
        y, x = i.shape  # colour channels: not used
        j = np.pad(i, [(floor((y_pad - y) / 2), ceil(((y_pad - y) / 2))),
                       (floor((x_pad - x) / 2), ceil(((x_pad - x) / 2)))],
                   mode='constant')  # Defaults to padding with 0s
        # cv.imshow("",j)
        # cv.waitKey(0)
        concat_numbers = np.hstack((concat_numbers, j))
    final = np.pad(concat_numbers, [(0, 0), (0, 1)], mode='constant', constant_values=127)
    if orig_img is not None:
        three = np.zeros(3)
        final = final[..., None] + three[None, None, :]
        final = np.vstack((orig_img, final))
    else:
        final = np.vstack((im, final))
    cv.imshow("", final)
    cv.waitKey(0)


def split_all():
    start = time.time()
    print("start time:", start)
    # Where the denoised captcha images are
    os.chdir(CAPTCHA_FOLDER)
    allfiles = os.listdir(os.getcwd())

    for i in range(10):  # Make the folders for each picture
        if not os.path.exists(str(i)):
            os.mkdir(str(i))

    imlist = [filename for filename in allfiles if os.path.isfile(filename) and filename[-4:] in [".png", ".PNG"]]

    successes = 0
    failures = 0

    for i in imlist:
        if save_digits(filename=i):
            successes += 1
        else:
            failures += 1
    end = time.time()
    print("end time:", end)
    print("took (seconds): ", end - start)
    print("number of files:", len(imlist))
    print("number of successes:", successes)
    print("number of failures:", failures)


def main():
    im = load_cv(input("filename? > "))
    show_digits(im)


if __name__ == "__main__":
    main()
