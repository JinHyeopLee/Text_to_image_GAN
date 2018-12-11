import numpy as np
import scipy.misc


def append_nparr(arr1, arr2, axis=0):
    if arr1 is None:
        arr1 = arr2
    else:
        arr1 = np.append(arr1, arr2, axis=axis)

    return arr1


def save_images(images, image_path):
    images = (images + 1) / 2

    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]

    img = np.zeros((h * 8, w * 8, c))
    for idx, image in enumerate(images):
        i = idx % 8
        j = idx // 8
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    img = np.squeeze(img)

    return scipy.misc.imsave(image_path, img)


def save_text(text, text_path):
    text = np.squeeze(text)
    text = text.tolist()

    with open(text_path, "w") as f:
        for line in text:
            line = line.decode("utf-8")
            f.write(line)
            f.write("\n")