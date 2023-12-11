# -*- coding: utf-8 -*-
"""
Created on 2023-11-02 (Thu) 17:43:56

reference:
https://github.com/vqdang/hover_net/tree/67e2ce5e3f1a64a2ece77ad1c24233653a9e0901/misc

@author: I.Azuma
"""
import numpy as np
import cv2
import colorsys
import random

def random_colors(N, bright=True):
    """Generate random colors.
    
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def overlay_viz(image,inst_dict,type_colour=None):
    if type_colour is None:
        type_colour = {
                0: ("nolabe", (0, 0, 0)),  # no label
                1: ("neopla", (255, 0, 0)),  # neoplastic
                2: ("inflam", (0, 255, 0)),  # inflamm
                3: ("connec", (0, 0, 255)),  # connective
                4: ("necros", (255, 255, 0)),  # dead
                5: ("no-neo", (255, 165, 0)),  # non-neoplastic epithelial
            }

    overlay = np.copy((image))
    inst_rng_colors = random_colors(len(inst_dict))
    inst_rng_colors = np.array(inst_rng_colors) * 255
    inst_rng_colors = inst_rng_colors.astype(np.uint8)

    for idx, [inst_id, inst_info] in enumerate(inst_dict.items()):
        inst_contour = inst_info["contour"]
        final_contour = tuple([np.array([t]) for t in inst_contour]) # sensitive format
        """ like this format.
        (array([[135, 986]]),
        array([[134, 987]]),
        array([[140, 988]]),
        array([[139, 988]]),
        array([[138, 987]]),
        array([[136, 987]]))
        """
        if "type" in inst_info and type_colour is not None:
            inst_colour = type_colour[inst_info["type"]][1]
        else:
            inst_colour = (inst_rng_colors[idx]).tolist()
        output = cv2.drawContours(overlay, final_contour, -1, inst_colour, 3)
    return output
    #cv2.imwrite('./output.png', output)