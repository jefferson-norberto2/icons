# Example usage
from icons.clicables import draw_bounding_boxes, get_clickable_areas


if __name__ == "__main__":
    image_path = r"icons\assets\2.jpg"
    clickable_areas = get_clickable_areas(image_path)
    draw_bounding_boxes(image_path, clickable_areas, f'{image_path}_boxes_v2.jpg')