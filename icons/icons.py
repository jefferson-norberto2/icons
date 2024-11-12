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
    print(w, h)
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

def compare_images(image1, image2):
    # Pré-processar as imagens e encontrar contornos principais
    edges1 = load_and_preprocess_image(image1)
    edges2 = load_and_preprocess_image(image2)

    contour1 = get_contours(edges1, 0)
    contour2 = get_contours(edges2, 0)

    # Comparar os contornos usando cv2.matchShapes
    similarity_score = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0.0)

    # Definir um limiar para considerar as imagens semelhantes
    similarity_threshold = 0.1  # Ajuste esse valor conforme necessário
    are_similar = similarity_score < similarity_threshold

    return are_similar, similarity_score

def compare_l2_norm(img1, img2):
    # Redimensiona as imagens para que sejam do mesmo tamanho
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    # Calcula a diferença absoluta entre as imagens
    difference = cv2.absdiff(img1, img2)
    
    # Calcula a norma L2 (distância Euclidiana)
    norm_diff = np.linalg.norm(difference)
    return norm_diff  # Valores mais baixos indicam maior similaridade

if __name__ == '__main__':
    
    # cap = cv2.VideoCapture(r'icons\assets\record.mp4')

    # ret, frame = cap.read()

    # while ret:
    #     frame = resize_by_scale(frame, scale=0.7)
    #     # mask = mask_hsv(frame)

    #     # cv2.bitwise_not(mask, mask)

    #     edges = load_and_preprocess_image(frame)

    #     cv2.dilate(edges, (9,9), edges, iterations=12)

    #     contours = get_contours(edges, 500)

    #     bboxes = get_bounding_boxes(contours, min_bbox_size=0)

    #     frame = drawn_bboxes_in_image(frame, bboxes)

    #     cv2.imshow('Vídeo', frame)

    #     # Aguarda 25 ms e verifica se a tecla 'q' foi pressionada para sair
    #     if cv2.waitKey(25) & 0xFF == ord('q'):
    #         break

    #     ret, frame = cap.read()

    # _, image = get_image(r'icons\assets\screen2.jpg', 0.4)

    # mask = mask_hsv(image, 0, 0, 0, 0, 0, 255)
    # edges = load_and_preprocess_image(image)

    # cv2.dilate(edges, (9,9), edges, iterations=9)

    # cv2.imshow('edges', edges)
    # cv2.bitwise_not(mask, mask)

    # contours = get_contours(edges, 500)

    # boxes = get_bounding_boxes(contours, min_bbox_size=0)

    # show_transformations(image, 'name', bboxes=boxes)
    # _, image = get_image(r'icons\assets\screen2.jpg', 0.5)

    image = cv2.imread(r'icons\assets\screen3.jpg')
    image = cv2.resize(image, (390, 800))
    # Obtém as dimensões do ícone
    icon = cv2.imread(r'icons\assets\rois\face.png')
    icon_height, icon_width = icon.shape[:2]

    # Realiza o template matching para encontrar o ícone no print
    result = cv2.matchTemplate(image, icon, cv2.TM_CCOEFF_NORMED)

    # Encontra a posição com a maior correspondência
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    print(max_val)

    # Define o threshold (ajuste conforme necessário)
    threshold = 0.3
    if max_val >= threshold:
        print("Ícone encontrado com correspondência:", max_val)
        # Desenha um retângulo ao redor da correspondência encontrada
        top_left = max_loc
        bottom_right = (top_left[0] + icon_width, top_left[1] + icon_height)
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    else:
        print("Ícone não encontrado ou correspondência insuficiente")

    # Exibe o resultado com as correspondências destacadas (se houver)
    cv2.imshow('Resultado', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # icon = cv2.imread(r'icons\assets\face2.png')
    # # edge_icon = load_and_preprocess_image(icon)
    # for i, (x, y, w, h) in enumerate(boxes):
    #     roi = image[y:y+h, x:x+w]
    #     # edge_roi = load_and_preprocess_image(roi)

    #     # print(compare_images(icon, roi))
        

    #     cv2.imshow('icone', roi)
    #     cv2.waitKey(0)
