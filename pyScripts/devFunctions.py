# Numpy
import numpy as np

# Matplotlib
import matplotlib.pyplot as plt

# OpenCV
import cv2
import os
from pathlib import Path


# Mostra as imagens
def Imshow(img, title=""):
    img = img / 2 + 0.5 # Retirando o normalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()


def CropImage(input_path:str, output_path:str=""):
    """
        Corta as imagens de dentro pasta (input) para adquirir somente o objeto (folha) central
    """

    # Lista de sufixos dos arquivos
    suffix = [".jpeg", ".jpg", ".png", "webp"]

    if os.path.isdir(input_path):
        src = Path(input_path)

        for file in src.iterdir():
            if file.suffix in suffix:

                # Manipulação de Imagem
                img = cv2.imread((input_path + "/" if input_path[-1] != "/" else input_path) + file.name)
                blue = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                blur = cv2.GaussianBlur(blue, (5, 5), 0)

                hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)

                lower = (25, 40, 40)
                upper = (85, 255, 255)

                mask = cv2.inRange(hsv, lower, upper)

                # Limpeza de Máscara
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                # Contornos
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                largest_contour = max(contours, key=cv2.contourArea)                

                # Seleção da área e recorte
                x, y, w, h = cv2.boundingRect(largest_contour)
                crop = img[y:y+h, x:x+w]

                # Salvar o recorte
                if output_path != "" and os.path.isdir(output_path):
                    cv2.imwrite((output_path + "/" if output_path[-1] != "/" else output_path) + file.name, crop)
                else:
                    cv2.imwrite("cropped_images/" + file.name, crop)

            else:
                # Arquivo não é uma imagem
                pass

    else:
        print("Error: Diretório não existente")