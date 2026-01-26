# Flask
from flask import request, Flask, jsonify
from flask_cors import CORS

# Scripts auxiliares
from imports import loadModelWithLabels

# PIL
from PIL import Image

# Torch
import torch
from torchvision import transforms

# Numpy
import numpy as np


app = Flask(__name__)
CORS(app)

# Importando os modelos
GeneralPATH = './models/General/generalAI_v3.pth'
CornPath = './models/Corn/CornAI_v1.pth'
SoybeanPath = './models/Soybean/SoybeanAI_v1.pth'
WheatPath = './models/Wheat/WheatAI_v1.pth'

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## Rede Neural ##
Net, checkpointNet = loadModelWithLabels("./models/General/generalAI.pth")
Corn, checkpointCorn = loadModelWithLabels("./models/Corn/cornAI.pth")
Soybean, checkpointSoybean = loadModelWithLabels("./models/Soybean/soybeanAI.pth")
Wheat, checkpointWheat = loadModelWithLabels("./models/Wheat/wheatAI.pth")

## Transform Geral ## # Pode ser interessante colocar em um módulo separado
transform = transforms.Compose([
    transforms.Resize((198, 198)),
    transforms.RandomRotation(4),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop((192, 192)),

    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


## Rodando a aplicação ##
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
        generalProbabilities = torch.softmax(output, dim=1).cpu().numpy()
        predicted_index = np.argmax(generalProbabilities)        
        generalPrediction = checkpointNet["class_names"][predicted_index]

        generalConfidence = float(generalProbabilities[0][predicted_index])

        match(generalPrediction):
            case "Corn":
                test_image = Image.open(file.stream).convert('RGB')
                test_image = transform(test_image).unsqueeze(0).to(DEVICE)

                output = Corn(test_image)
                probabilities = torch.softmax(output, dim=1).cpu().numpy()
                predicted_index = np.argmax(probabilities)        
                predicted_class = checkpointCorn["class_names"][predicted_index]

                sicknessConfidence = float(probabilities[0][predicted_index])
                
            case "Soybean":
                test_image = Image.open(file.stream).convert('RGB')
                test_image = transform(test_image).unsqueeze(0).to(DEVICE)

                output = Soybean(test_image)
                probabilities = torch.softmax(output, dim=1).cpu().numpy()
                predicted_index = np.argmax(probabilities)        
                predicted_class = checkpointSoybean["class_names"][predicted_index]

                sicknessConfidence = float(probabilities[0][predicted_index])

            case "Wheat":
                test_image = Image.open(file.stream).convert('RGB')
                test_image = transform(test_image).unsqueeze(0).to(DEVICE)

                output = Wheat(test_image)
                probabilities = torch.softmax(output, dim=1).cpu().numpy()
                predicted_index = np.argmax(probabilities)        
                predicted_class = checkpointWheat["class_names"][predicted_index]

                sicknessConfidence = float(probabilities[0][predicted_index])

            case _:
                predicted_class = "doença não identificada."

    return jsonify({'plant': generalPrediction.lower(), 
                    'plantConfidence': generalConfidence,
                    'prediction': predicted_class.lower(), 
                    'predictionConfidence': sicknessConfidence}), 200



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)