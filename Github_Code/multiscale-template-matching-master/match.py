# USAGE
# python match.py --template cod_logo.png --images images

import glob

import cv2
# import the necessary packages
import numpy as np

import imutils

# load the image image, convert it to grayscale, and detect edges
template = cv2.imread('mario_coin.png')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]
cv2.imshow("Template", template)

# loop over the images to find the template in
for imagePath in glob.glob('mario.jpg'):
    # load the image, convert it to grayscale, and initialize the
    # bookkeeping variable to keep track of the matched region
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None

    # loop over the scales of the image
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])

        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        edged = cv2.Canny(resized, 50, 200)
        res = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        w, h = resized.shape[::-1]
        threshold = 0.99
        loc = np.where(res >= threshold)
        print(len(list(zip(*loc[::-1]))))
        for pt in zip(*loc[::-1]):
            cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
    cv2.imshow('res.png', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
