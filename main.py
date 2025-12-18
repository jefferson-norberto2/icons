# Example usage
from icons.clicables import draw_bounding_boxes, get_clickable_areas, read_image, save_bounding_boxes_to_file
from icons.background import merge_image_in_image_with_black_background, black_percentual


if __name__ == "__main__":
    image_path = r"icons\assets\1.jpg"
    background_path = r"icons\assets\background.png"
    image = read_image(image_path)
    background = read_image(background_path)
    percentual = black_percentual(image, black_thresh=10)
    print(f"Percentual: {percentual*100:.2f}%")

    merge_image_in_image_with_black_background(image, background, 2.0, f'{image_path}_suave.png')
    clickable_areas = get_clickable_areas(image)
    base_name = image_path.split('.')
    extension = '.txt'
    base_name[-1] = extension
    file_path = ''.join(base_name)
    save_bounding_boxes_to_file(clickable_areas, image, file_path)
