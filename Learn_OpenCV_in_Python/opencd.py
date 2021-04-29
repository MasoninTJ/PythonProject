import MTM

print("MTM version: ", MTM.__version__)

from MTM import matchTemplates, drawBoxesOnRGB

import cv2
import matplotlib.pyplot as plt

image = cv2.imread('image/mario.jpg', cv2.IMREAD_GRAYSCALE)

small_image = cv2.imread('image/mario_coin.png', cv2.IMREAD_GRAYSCALE)

# 1st format the template into a list of tuple (label, templateImage)
listTemplate = [('small', small_image)]

# Then call the function matchTemplates (here a single template)
Hits = matchTemplates(listTemplate, image, score_threshold=0.8, method=cv2.TM_CCOEFF_NORMED, maxOverlap=0)
print("Found {} hits".format(len(Hits.index)))
print(Hits)
Overlay = drawBoxesOnRGB(image, Hits, showLabel=False)
plt.imshow(Overlay)

plt.show()
