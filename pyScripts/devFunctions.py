import numpy as np
import matplotlib.pyplot as plt

# Função para mostrar as imagens
def imshow(img):
    img = img / 2 + 0.5 # Retirando o normalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()