import cv2

import numpy as np


def boxes_intersect(box, boxes, buffer=20):
    """
    Check intersection between one box and multiple boxes using NumPy.

    :param box: shape (4,) -> (x, y, w, h)
    :param boxes: shape (N, 4)
    :param buffer: tolerance in pixels
    :return: boolean mask of shape (N,)
    """
    x, y, w, h = box

    return (
        (x - buffer < boxes[:, 0] + boxes[:, 2]) &
        (x + w + buffer > boxes[:, 0]) &
        (y - buffer < boxes[:, 1] + boxes[:, 3]) &
        (y + h + buffer > boxes[:, 1])
    )


def merge_boxes(box, boxes):
    """
    Merge one box with multiple boxes into a single bounding box.

    :param box: shape (4,)
    :param boxes: shape (N, 4)
    :return: merged box (4,)
    """
    all_boxes = np.vstack([box, boxes])

    x_min = np.min(all_boxes[:, 0])
    y_min = np.min(all_boxes[:, 1])

    x_max = np.max(all_boxes[:, 0] + all_boxes[:, 2])
    y_max = np.max(all_boxes[:, 1] + all_boxes[:, 3])

    return np.array([
        x_min,
        y_min,
        x_max - x_min,
        y_max - y_min
    ])


def merge_bounding_boxes(boxes, buffer=20):
    """
    Merge overlapping or nearby bounding boxes using NumPy.

    :param boxes: list or array of bounding boxes (x, y, w, h)
    :param buffer: tolerance in pixels
    :return: list of merged bounding boxes
    """
    if len(boxes) == 0:
        return []

    boxes = np.asarray(boxes, dtype=np.float32)

    changed = True
    while changed:
        changed = False
        new_boxes = []
        used = np.zeros(len(boxes), dtype=bool)

        for i in range(len(boxes)):
            if used[i]:
                continue

            current = boxes[i]

            mask = boxes_intersect(current, boxes, buffer)
            mask[i] = False  # ignore self
            mask &= ~used    # ignore already merged boxes

            if np.any(mask):
                current = merge_boxes(current, boxes[mask])
                used[mask] = True
                changed = True

            used[i] = True
            new_boxes.append(current)

        boxes = np.vstack(new_boxes)

    return boxes.astype(int).tolist()

def read_image(image_path):
    """
    Reads an image from the given path.

    :param image_path: Path to the input image
    :return: Loaded image
    """
    return cv2.imread(image_path)


def get_clickable_areas(image, threshold=100):
    """
    This function takes an image path and a threshold value as input,
    processes the image to find clickable areas (non-white regions),
    and returns a list of bounding boxes for these areas.

    :param image_path: Path to the input image
    :param threshold: Threshold value to determine clickable areas
    :return: List of bounding boxes for clickable areas
    """
    # Load the image
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    # Apply a binary threshold to get a binary image
    _, binary_image = cv2.threshold(image_gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours of the clickable areas
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding boxes for each contour
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    # Merge overlapping bounding boxes
    bounding_boxes = merge_bounding_boxes(bounding_boxes)
    
    return bounding_boxes

def draw_bounding_boxes(image, bounding_boxes, output_path):
    """
    This function draws bounding boxes on the image and saves the result.

    :param image: the input image
    :param bounding_boxes: List of bounding boxes to draw
    :param output_path: Path to save the output image
    """
    
    # Draw each bounding box on the image
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Save the output image
    cv2.imwrite(output_path, image)

def save_bounding_boxes_to_file(bounding_boxes, image, file_path):
    """
    This function saves the bounding boxes to a text file.

    :param bounding_boxes: List of bounding boxes to save
    :param file_path: Path to the output text file
    """
    width, height = image.shape[1], image.shape[0]
    with open(file_path, 'w') as f:
        for box in bounding_boxes:
            # Take centroid and normalize
            x_center = (box[0] + box[2] / 2) / width
            y_center = (box[1] + box[3] / 2) / height
            w_norm = box[2] / width
            h_norm = box[3] / height
            f.write(f"0 {x_center} {y_center} {w_norm} {h_norm}\n")

def show_boxes(image, bounding_boxes):
    """
    This function displays the image with bounding boxes.

    :param image: the input image
    :param bounding_boxes: List of bounding boxes to draw
    """
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Adjust window size to image size
    cv2.namedWindow('Image with Bounding Boxes', cv2.WINDOW_NORMAL)
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def xcyc_wh_normalized_to_xminymin_xmaxymax_pixels(image, file_path):
    """
    This function reads bounding boxes from a text file, draws them on the image,
    and saves the result.

    :param image: the input image
    :param file_path: Path to the input text file with bounding boxes
    :param output_path: Path to save the output image
    """
    height, width = image.shape[0], image.shape[1]
    bounding_boxes = []

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, x_center, y_center, w_norm, h_norm = map(float, parts)
            w = int(w_norm * width)
            h = int(h_norm * height)
            x = int(x_center * width - w / 2)
            y = int(y_center * height - h / 2)
            bounding_boxes.append((x, y, w, h))

    return bounding_boxes
