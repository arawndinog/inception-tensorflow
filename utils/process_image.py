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

def aggressive_crop(img):
    img_shape = img.shape
    min_dim = min(img_shape)
    result_img_list = []
    img_size_list = [256,288,320,352]
    for img_size in img_size_list:
        img_resized = cv2.resize(img, None, fx=img_size/min_dim, fy=img_size/min_dim, interpolation = cv2.INTER_CUBIC)
        if img_shape[0] <= img_shape[1]:
            # img is landscape
            img_crop1 = img_resized[:,:img_size]
            img_crop2 = img_resized[:,img_resized.shape[1]//2-img_size//2:img_resized.shape[1]//2-img_size//2+img_size]
            img_crop3 = img_resized[:,-img_size:]
        else:
            # img is portrait
            img_crop1 = img_resized[:img_size, :]
            img_crop2 = img_resized[img_resized.shape[0]//2-img_size//2:img_resized.shape[0]//2-img_size//2+img_size, :]
            img_crop3 = img_resized[-img_size:, :]

        img_crop_list = [img_crop1, img_crop2, img_crop3]
        
        for img_crop in img_crop_list:
            img_minicrop_1 = img_crop[:224, :224]
            img_minicrop_2 = img_crop[-224:, :224]
            img_minicrop_3 = img_crop[:224, -224:]
            img_minicrop_4 = img_crop[-224:, -224:]
            img_minicrop_5 = img_crop[img_crop.shape[0]//2-224//2:img_crop.shape[0]//2-224//2+224,img_crop.shape[1]//2-224//2:img_crop.shape[1]//2-224//2+224]
            img_crop_resized = cv2.resize(img_crop, (224,224), interpolation = cv2.INTER_CUBIC)
            img_minicrop_1_flipped = cv2.flip(img_minicrop_1, 1)
            img_minicrop_2_flipped = cv2.flip(img_minicrop_2, 1)
            img_minicrop_3_flipped = cv2.flip(img_minicrop_3, 1)
            img_minicrop_4_flipped = cv2.flip(img_minicrop_4, 1)
            img_minicrop_5_flipped = cv2.flip(img_minicrop_5, 1)
            img_crop_resized_flipped = cv2.flip(img_crop_resized, 1)
            result_img_list.extend([img_minicrop_1, img_minicrop_2, img_minicrop_3, img_minicrop_4, img_minicrop_5, img_crop_resized,
                img_minicrop_1_flipped, img_minicrop_2_flipped, img_minicrop_3_flipped, img_minicrop_4_flipped, img_minicrop_5_flipped, img_crop_resized_flipped])

    assert len(result_img_list) == 144
    return result_img_list