import cv2 as cv
import numpy as np

im = cv.imread("7.png")
im2 = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

im_neg = cv.bitwise_not(im2)
contours, hierarchy = cv.findContours(im_neg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

hierarchy = hierarchy[0]

# cv.drawContours(im,contours,-1, (0,255,0), 1)

# # cv.imshow('',im)
# # cv.waitKey(0)


# print(hierarchy)

# print(contours[0])
# # for i in contours:
#     # print(i)
#     # print(cv.contourArea(i))

reshaped = []
for n, c in enumerate(contours):  # Currently c has X,1,Y shape
    if hierarchy[n][3] == -1:  # Keep if hierarchy shows no parents: remove internal holes
        if cv.contourArea(c) >= 10:  # Remove super small stuff
            d = c.reshape(c.shape[0],c.shape[2])  # Reshape to X,Y
            reshaped.append(d)  # Make new list, modifying same list doesn't work somehow


borders = []
for c in reshaped:
    min_x, min_y = c[0]
    max_x, max_y = c[0]
    for point in c:
        if min_x > point[0]:
            min_x = point[0]
        elif max_x < point[0]:
            max_x = point[0]
        if min_y > point[1]:
            min_y = point[1]
        elif max_y < point[1]:
            max_y = point[1]
    borders.append(np.array([[min_x,min_y],[max_x,max_y]]))
# print(f"number of regions: {len(borders)}")

im3 = im
for b in borders:
    cv.rectangle(im3,tuple(b[0]),tuple(b[1]),50)
cv.imshow('',im3)
cv.waitKey(0)

def rect_size(b):
    min_x, min_y = b[0]
    max_x, max_y = b[1]

    x = max_x - min_x + 1
    y = max_y - min_y + 1

    return x*y,x,y

to_split = []
for n, i in enumerate(borders):

    min_x, min_y = i[0]
    max_x, max_y = i[1]

    # if x > 33 then split vertically
    if max_x - min_x > 32:
        to_split.append(n)

for n in to_split:
    b = borders.pop(n)
    x_0, y_0 = b[0]
    x_2, y_1 = b[1]
    x_1 = (x_2+x_0)//2  # Getting the middle coord
    borders.append(np.array([[x_0,y_0],[x_1,y_1]]))  # Left half
    borders.append(np.array([[x_1,y_0],[x_2,y_1]]))  # Right half

im3 = im
for b in borders:
    cv.rectangle(im3,tuple(b[0]),tuple(b[1]),50)
cv.imshow('',im3)
cv.waitKey(0)

modified = True
while modified:
    borders.sort(key = rect_size, reverse = True)  # Tuples sort by comparing 1st things, then 2nd if 1st are same etc.
    modified = False
    border_pop = []
    for n, b in enumerate(borders[:4]):
        b_x_0, b_y_0 = b[0]
        b_x_1, b_y_1 = b[1]

        for m, c in enumerate(borders):  # TODO: order by closest X coordinate?
            c_x_0, c_y_0 = c[0]
            c_x_1, c_y_1 = c[1]

            # Constrain c size between 60 and 200 to avoid all sorts of issues
            if 60 < rect_size(c)[0] < 200:

                # If a corner of c is close enough to b then add them together
                if (b_x_0 - 2 < c_x_0 < (b_x_1+b_x_0)//2 or (b_x_1+b_x_0)//2 < c_x_1 < b_x_1 + 2) and \
                   (b_y_0 - 5 < c_y_1 < b_y_1 or b_y_0 < c_y_0 < b_y_1 + 5):
                    modified = True

                    # Remove old borders, add the new one
                    # print("removing old border:", b)
                    borders.pop(n)
                    # print("removing old border:", c)
                    borders.pop(m-1)

                    # print("adding new border")
                    borders.append(np.array(
                        [[min(b_x_0,c_x_0),min(b_y_0,c_y_0)],
                        [max(b_x_1,c_x_1),max(b_y_1,c_y_1)]]))
                    break
        if modified:
            # print("modified one thing, breaking loop")
            break

im3 = im
for b in borders:
    cv.rectangle(im3,tuple(b[0]),tuple(b[1]),50)
cv.imshow('',im3)
cv.waitKey(0)
