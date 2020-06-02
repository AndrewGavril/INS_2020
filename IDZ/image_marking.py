import numpy as np
from PIL import Image, ImageDraw


def get_borders(mask, w, h):
    points = []
    for i in range(0, h):
        for j in range(0, w):
            if round(mask[i, j]) == 1:
                isBorder = False
                isBorder = j == 0
                isBorder |= j-1 >= 0 and round(mask[i][j-1]) == 0
                isBorder |= i == 0
                isBorder |= i-1 >= 0 and round(mask[i-1][j]) == 0
                isBorder |= j == w-1
                isBorder |= j+1 < w and round(mask[i][j+1]) == 0
                isBorder |= i == h-1
                isBorder |= i+1 < h and round(mask[i+1][j]) == 0
                if isBorder:
                    points.append((j, i)) 
    return points


def draw_masks(image_arr, masks, save_as):
    im = None
    im = Image.fromarray(image_arr)
    im = im.convert('RGB')
    (w, h) = im.size
    draw = ImageDraw.Draw(im)
    fill = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]
    for i in range(0, 4):
        points = get_borders(masks[:, :, i], w, h)
        draw.point(points, fill=fill[i])
    im.save(save_as, "PNG")
    del draw
    im.close()
