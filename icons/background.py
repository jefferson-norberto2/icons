import cv2
import numpy as np

def black_percentual(image, black_thresh=10):
    if image.ndim == 3:
        image = image.mean(axis=2)

    return np.mean(image <= black_thresh)

def merge_image_in_image_with_black_background(image, background, alpha=1.0, result_file_path='result.png'):

    image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)

    image_height, image_width = image.shape[:2]
    background = cv2.resize(background, (image_width, image_height))

    image_float = image.astype(float) / 255.0
    background_float = background.astype(float) / 255.0

    # Screen Blending Mode
    # Formula: 1 - (1 - A) * (1 - B)
    result_float = 1.0 - (1.0 - background_float) * (1.0 - image_float)

    image_result = (result_float * 255).astype(np.uint8)

    cv2.imwrite(result_file_path, image_result)
    print(f"Imagem salva com suavidade em: {result_file_path}")