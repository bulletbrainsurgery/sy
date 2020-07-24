import os
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance  # this is pillow
from scipy import ndimage
import time

def getpic(pic = None, letter = None):
    if pic is None:
        pic = Image.open(letter+".png").convert('RGB')
    return pic

def denoise(pic = None, filename = None, letter = None):
    if pic == None:
        if filename == None:
            filename = '%s.png'%letter
        pic = Image.open(filename).convert('RGB')
        # pic = pic.crop((0,0,150,55))
    r1 = showedges(pic)
    r2 = mycontrast(r1)
    r3 = cluster(r2, mask_s = 80)
    r4 = medfilt(r3, level = 7) # level 6 works better with thin fonts but makes thick fonts worse
    r5 = cluster(r4, mask_s = 15)
    return(r5)
    # os.chdir('C:\\Users\\David\\test\\syrnia\\captchas\\processed')
    # r5.save(filename)
    # os.chdir('C:\\Users\\David\\test\\syrnia\\captchas')

def showedges(pic = None, letter = None):
    pic = getpic(pic, letter)
    picEdges = pic.filter(ImageFilter.FIND_EDGES)
    # picEdges.show(title = "Edges shown")
    return picEdges

def mycontrast(pic = None, letter = None):
    pic = getpic(pic, letter)
    a = ImageEnhance.Contrast(pic)
    a = a.enhance(10) # Not sure if it accomplishes much
    pic = pic.convert('1')  # Contrast only works with rgb images
    # a.show(title = "Contrasted")
    return a

def cluster(pic = None, letter = None, mask_s = 20):
    pic = getpic(pic, letter)
    p1 = np.array(ImageOps.invert(pic.convert('L')))
    mask = p1 > p1.mean()
    label_im, nb_labels = ndimage.label(mask)
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    mask_size = sizes < mask_s
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0
    picture1 = Image.fromarray(label_im*255)
    result = ImageOps.invert(picture1.convert('L'))
    # result.show(title = "Cluster filter")
    return result

def medfilt(pic = None, letter = None, level = 7):
    pic = getpic(pic, letter)
    pic1 = pic.filter(ImageFilter.RankFilter(3, level))
    # pic1.show(title = "Rank filter applied")
    return pic1

def convert_all():
    start = time.time()
    os.chdir('C:\\Users\\neggi\\code\\syrnia\\captcha\\captchas')
    # allfiles = os.listdir(os.getcwd())
    # imlist = [filename for filename in allfiles if filename[-4:] in [".png",".PNG"]]
#     for i in imlist:
#         file = denoise(filename = i)
#         # save file somewhere
    end = time.time()
    print(end-start)

def main():
    # denoise(filename = "a.png")
    pic = Image.open("c.png")
    p1 = medfilt(pic, level = 4)

if __name__ == "__main__":
    main()