import cv2
import numpy as np
import os

def load_and_preprocess_image(image):
    # Carregar a imagem e converter para escala de cinza
    if image is None:
        raise ValueError(f"Imagem não encontrada no caminho: {image}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Aplicar detecção de bordas para destacar o contorno
    edges = cv2.Canny(image, 0, 20)
    return edges

def get_contours(mask, min_contour_area=1000) -> list:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

    return filtered_contours

def get_bounding_boxes(contours, min_bbox_size=300) -> list:
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= min_bbox_size or h >= min_bbox_size:
            boxes.append([x, y, w, h])

    return boxes

def drawn_bboxes_in_image(image, bboxes):
    img = image.copy()

    for bbox in bboxes:
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 1)
    
    return img

def mask_hsv(image, h_min=0, h_max=179, s_min=0, s_max=40, v_min=0, v_max=130):
    temp_image = image.copy()

    # Converter a imagem para HSV
    hsv_img = cv2.cvtColor(temp_image, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([h_min, s_min, v_min])
    upper_hsv = np.array([h_max, s_max, v_max])

    # Aplicar o filtro
    mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)

    kernel = np.ones((3,3),np.uint8)
    mask = cv2.erode(mask, kernel,iterations = 2)
    # kernel = np.ones((3,3),np.uint8)/
    # mask = cv2.dilate(mask, kernel, iterations = 1)

    return mask

def resize_by_scale(image, scale=1.0):
    img = image.copy()
    h, w, _ = img.shape
    h = int(h * scale)
    w = int(w * scale)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

    return img

def get_image(path_image: str, scale = 1.0):
    ret = False
    if os.path.isfile(path_image) and path_image.split('.')[-1].lower() == 'jpg':
        image = cv2.imread(path_image)
        image = resize_by_scale(image, scale)
        ret = True
    else:
        image = None

    return ret, image

def show_transformations(image: np.ndarray, image_name='name.jpg', contours: list = None, bboxes: list=None):
    if image is None:
        raise Exception('Image can\'t be None')

    img = image.copy()
    if contours is not None:
        cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
    
    if bboxes is not None:
        img = drawn_bboxes_in_image(img, bboxes)
    
    cv2.imshow(image_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def icon_match(image, icons_path, threshold=0.6):
    img = image.copy()
    icons = os.listdir(icons_path)

    max_values = []
    max_locations = []
    icon_dimensions = []

    for icon in icons:
        # Carrega a imagem do ícone
        icon_image = cv2.imread(f'{icons_path}/{icon}')
        icon_height, icon_width = icon_image.shape[:2]
        
        # Realiza o template matching
        result = cv2.matchTemplate(img, icon_image, cv2.TM_CCOEFF_NORMED)
        
        # Encontra a posição com a maior correspondência
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        # Define o threshold (ajuste conforme necessário)
        
        if max_val >= threshold:
            threshold = round(max_val, 1)
            max_values.append(max_val)
            max_locations.append(max_loc)
            icon_dimensions.append((icon_width, icon_height))  # Armazena as dimensões do ícone

    # Verifica se encontrou correspondências válidas
    if max_values:
        # Encontra o índice do maior valor de correspondência
        best_match_index = max_values.index(max(max_values))

        print('maior valor', max_values[best_match_index])
        
        # Pega a posição e as dimensões do ícone com maior correspondência
        best_match_loc = max_locations[best_match_index]
        best_icon_width, best_icon_height = icon_dimensions[best_match_index]
        
        # Desenha o retângulo ao redor do ícone encontrado
        top_left = best_match_loc
        bottom_right = (top_left[0] + best_icon_width, top_left[1] + best_icon_height)
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
    
    return img, threshold


if __name__ == '__main__':
    
    cap = cv2.VideoCapture(r'icons\assets\record2.mp4')
    ret, frame = cap.read()
    threshold = 0.6

    while ret:
        # frame = resize_by_scale(frame, scale=0.3)
        frame = cv2.resize(frame, (400, 850))

        frame, threshold = icon_match(frame, r'icons\assets\rois\instagram', threshold)

        cv2.imshow('Vídeo', frame)

        # Aguarda 25 ms e verifica se a tecla 'q' foi pressionada para sair
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        ret, frame = cap.read()

    # _, image = get_image(r'icons\assets\screen3.jpg', 0.3)

    # image, _ = icon_match(image, r'icons\assets\rois\facebook')

    # cv2.imshow('image', image)
    # cv2.waitKey(0)