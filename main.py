# Example usage
from icons.clicables import draw_bounding_boxes, get_clickable_areas, read_image, save_bounding_boxes_to_file, draw_boxes_by_file
from icons.background import merge_image_in_image_with_black_background, black_percentual


if __name__ == "__main__":
    image_path = r"icons\assets\menu_camera.jpg"
    background_path = r"icons\assets\background.png"
    image = read_image(image_path)
    background = read_image(background_path)
    percentual = black_percentual(image, black_thresh=10)
    print(f"Percentual: {percentual*100:.2f}%")

    # merge_image_in_image_with_black_background(image, background, 2.0, f'{image_path}_suave.png')
    # clickable_areas = get_clickable_areas(image)
    # draw_bounding_boxes(image, clickable_areas, f'{image_path}_boxes_v2.jpg')
    # save_bounding_boxes_to_file(clickable_areas, image, f'{image_path}_boxes_v2.txt')
    # draw_boxes_by_file(image, f'{image_path}_boxes_v2.txt', f'{image_path}_from_file_v2.jpg')
    # print("Clickable areas processed and saved.")