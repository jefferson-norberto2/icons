import cv2
import numpy as np

def nothing(x):
    pass

# Carregar a imagem
src = cv2.imread(r'icons\assets\screen1.jpg')

# Verifica se a imagem foi carregada corretamente
if src is None:
    print("Erro ao carregar a imagem")
    exit()

# Redimensionar a imagem para um tamanho menor (exemplo: 50% do tamanho original)
scale_percent = 30  # Redimensionar para 50% do tamanho original
width = int(src.shape[1] * scale_percent / 100)
height = int(src.shape[0] * scale_percent / 100)
dim = (width, height)

# Redimensionar a imagem original
src = cv2.resize(src, (300, 600), interpolation=cv2.INTER_AREA)

# Criar uma janela para exibir o resultado
cv2.namedWindow('result')

# Criar trackbars para ajustar os valores de Hue, Saturation e Value
cv2.createTrackbar('Hue Min', 'result', 0, 179, nothing)
cv2.createTrackbar('Hue Max', 'result', 179, 179, nothing)
cv2.createTrackbar('Saturation Min', 'result', 0, 255, nothing)
cv2.createTrackbar('Saturation Max', 'result', 40, 255, nothing)
cv2.createTrackbar('Value Min', 'result', 0, 255, nothing)
cv2.createTrackbar('Value Max', 'result', 130, 255, nothing)

while True:
    try:
        # Obter os valores atuais das barras deslizantes
        h_min = cv2.getTrackbarPos('Hue Min', 'result')
        h_max = cv2.getTrackbarPos('Hue Max', 'result')
        s_min = cv2.getTrackbarPos('Saturation Min', 'result')
        s_max = cv2.getTrackbarPos('Saturation Max', 'result')
        v_min = cv2.getTrackbarPos('Value Min', 'result')
        v_max = cv2.getTrackbarPos('Value Max', 'result')

        # Definir os limites inferiores e superiores do HSV
        lower_hsv = np.array([h_min, s_min, v_min])
        upper_hsv = np.array([h_max, s_max, v_max])

        # Converter a imagem para o espaço de cor HSV
        hsv_img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

        # Aplicar o filtro HSV
        mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
        result = cv2.bitwise_and(src, src, mask=mask)

        # Mostrar a imagem filtrada
        cv2.imshow('result', result)

        # Aguardar por 1ms, se a tecla 'ESC' for pressionada, sair do loop
        if cv2.waitKey(1) & 0xFF == 27:
            break
    except:
        break

# Fechar todas as janelas
cv2.destroyAllWindows()