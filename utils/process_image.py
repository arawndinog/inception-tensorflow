import cv2
import numpy as np

def reshape_img_batch_to_size(img_batch, size):
    # Nov 1, 2019: resize an array of img to specified size
    resized_img_list = []
    for i in range(len(img_batch)):
        resized_img = cv2.resize(img_batch[i], size, None)
        resized_img = normalize_color(resized_img)
        resized_img = np.expand_dims(resized_img, 2)
        # convert to rgb
        resized_img = np.repeat(resized_img, 3, 2)
        resized_img_list.append(resized_img)
    return resized_img_list

def normalize_color(img, out_uint8=True):
    img = img.astype(np.float32)
    max_color = np.amax(img)
    min_color = np.amin(img)
    color_diff = max((max_color - min_color),1)
    color_scale = max(255*int(out_uint8),1)
    result_img = ((img-min_color)/color_diff)*color_scale
    if out_uint8:
        result_img = result_img.astype(np.uint8)
    return result_img