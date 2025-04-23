# Flask
from flask import request, Flask, jsonify
from flask_cors import CORS

# PIL
from PIL import Image

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Numpy
import numpy as np

# CV2
import cv2


app = Flask(__name__)
CORS(app)

GeneralPATH = './models/General/generalAI_v2_1.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## Rede Neural ##
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Construção das hidden layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        # Flatten
        self.fc1 = nn.Linear(128 * 15 * 15, 120)
        self.bn4 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

        # Ativação GeLU
        self.gelu = nn.GELU()
    
    def forward(self, x):
        x = self.pool(F.gelu(self.bn1(self.conv1(x))))
        x = self.pool(F.gelu(self.bn2(self.conv2(x))))
        x = self.pool(F.gelu(self.bn3(self.conv3(x))))

        x = torch.flatten(x, 1)

        x = F.gelu(self.bn4(self.fc1(x)))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)

        return x

Net = NeuralNetwork()
Net.load_state_dict(torch.load(GeneralPATH, map_location=DEVICE))
Net.to(DEVICE)
Net.eval()


## Função lambda que aplica o filtro Sobel ##
sobel_transform = transforms.Lambda(
    lambda img: Image.fromarray(
        cv2.convertScaleAbs(
            np.hypot(
                cv2.Sobel(np.array(img), cv2.CV_64F, 1, 0, ksize=3),
                cv2.Sobel(np.array(img), cv2.CV_64F, 0, 1, ksize=3)
            )
        )
    )
)


## Transform ##
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(1),
    sobel_transform,
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)), 
])


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada"}), 400

    file = request.files['image']
    test_image = Image.open(file.stream).convert('RGB')

    # Pré-processar a imagem
    test_image = transform(test_image).unsqueeze(0).to(DEVICE)

    # Previsão
    with torch.no_grad():
        output = Net(test_image)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()
        predicted_index = np.argmax(probabilities)
        predicted_class = getProbability(predicted_index)

    return jsonify({'prediction': predicted_class.lower(), 'raw_prediction': probabilities.tolist()}), 200

def getProbability(predicted_index):
    class_names = ['Corn', 'Soybean', 'Wheat']
    return class_names[predicted_index]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)