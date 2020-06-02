import pandas as pd
import numpy as np
import os
from PIL import Image
from math import floor

def get_mask(enc_pixels, w, h):
    points = []
    mask = np.zeros((h, w))
    for i in range(0, len(enc_pixels)//2):
        start = enc_pixels[2*i]
        run_length = enc_pixels[2*i+1]
        x = (start-1) // h
        y = (start-1) % h
        for j in range(0, run_length):
            mask[y, x] = 1
            x += (y + 1) // h
            y = (y + 1) % h
    return mask
            
def get_labels(df, w, h, filename):
    data = df.loc[df['ImageId'] == filename]
    masks = np.zeros((h, w, 4))
    for i in range(1, 5):
        if data.values.shape[0] != 0:
            enc_pixels = data.loc[data['ClassId'] == i].values
            if enc_pixels.shape[0] != 0:
                enc_pixels = enc_pixels[0][2]
                enc_pixels = np.array(list(map(int, enc_pixels.split())))
                masks[:, :, i-1] = get_mask(enc_pixels, w, h)
    return masks
    

def convert(df, directory):
    names = list(map(lambda x: directory + x, os.listdir(directory)))
    filenames = os.listdir(directory)

    total = len(names)
    im = Image.open(names[0])
    w, h = im.size
    im.close()
    imgs = np.zeros((total, h, w), dtype=np.uint8)
    masks = np.zeros((total, h, w, 4), dtype=np.uint8)


    for i in range(0, total):
        im = Image.open(names[i])
        im_conv = im.convert('L')
        img = np.array(im_conv, dtype=np.uint8)
        imgs[i] = img
        masks[i] = get_labels(df, w, h, filenames[i])
        if i % 100 == 0:
            print('{}/{}'.format(i, total))
    np.save('imgs.npy', imgs)
    np.save('masks.npy', masks)

    
def main():
    df = pd.read_csv('new_train.csv', sep=',')
    convert(df, 'dataset\\new_train_images\\')

if __name__ == '__main__':
    main()
