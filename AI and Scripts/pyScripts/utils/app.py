from flask import request, Flask, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

app = Flask(__name__)
CORS(app)

# Carregar o modelo PyTorch
MODEL_PATH = 'SicknessMinder_V3_4_1.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Certifique-se de que a classe do modelo corresponde à arquitetura salva
class SicknessClassifier(torch.nn.Module):
    def __init__(self):
        super(SicknessClassifier, self).__init__()
        # Defina a arquitetura do modelo aqui
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(16 * 150 * 150, 3)  # Ajuste conforme necessário

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 150 * 150)  # Flatten
        x = self.fc1(x)
        return x

model = SicknessClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Transformação para pré-processar a imagem
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada"}), 400

    file = request.files['image']
    test_image = Image.open(file.stream).convert('RGB')

    # Pré-processar a imagem
    test_image = transform(test_image).unsqueeze(0).to(DEVICE)

    # Fazer a previsão
    with torch.no_grad():
        output = model(test_image)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()
        predicted_index = np.argmax(probabilities)
        predicted_class = getProbability(predicted_index)

    return jsonify({'prediction': predicted_class.lower(), 'raw_prediction': probabilities.tolist()}), 200

def getProbability(predicted_index):
    class_names = ['Cercosporiose', 'Ferrugem', 'Saudavel']
    return class_names[predicted_index]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)