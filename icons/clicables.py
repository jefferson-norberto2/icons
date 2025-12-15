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


def get_clickable_areas(image_path, threshold=100):
    """
    This function takes an image path and a threshold value as input,
    processes the image to find clickable areas (non-white regions),
    and returns a list of bounding boxes for these areas.

    :param image_path: Path to the input image
    :param threshold: Threshold value to determine clickable areas
    :return: List of bounding boxes for clickable areas
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply a binary threshold to get a binary image
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours of the clickable areas
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding boxes for each contour
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    # Merge overlapping bounding boxes
    bounding_boxes = merge_bounding_boxes(bounding_boxes)
    
    return bounding_boxes

def draw_bounding_boxes(image_path, bounding_boxes, output_path):
    """
    This function draws bounding boxes on the image and saves the result.

    :param image_path: Path to the input image
    :param bounding_boxes: List of bounding boxes to draw
    :param output_path: Path to save the output image
    """
    # Load the original image
    image = cv2.imread(image_path)
    
    # Draw each bounding box on the image
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Save the output image
    cv2.imwrite(output_path, image)