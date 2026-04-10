# FastAPI
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import uvicorn

# Scripts auxiliares
from imports import loadModelWithLabels

# System
import os
import io
from PIL import Image

# Torch
import torch
from torchvision import transforms

# Numpy
import numpy as np

from dotenv import load_dotenv


# Loading env
if load_dotenv():
    print(".env loaded successfully.") # Apply logging
else:
    raise("Couldn't load the .env")


# Application
app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

print(f"Origens: {origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"], # Change for security
)

# CNN
## Importing models
## Generalist
GeneralPATH = os.getenv("GENERALIST")

## Experts
CornPath = os.getenv("CORN")
WheatPath = os.getenv("WHEAT")
SoybeanPath = os.getenv("SOY")

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## Rede Neural ##
# Net, checkpointNet = loadModelWithLabels(GeneralPATH)
# Corn, checkpointCorn = loadModelWithLabels(CornPath)
# Wheat, checkpointWheat = loadModelWithLabels(WheatPath)
# Soybean, checkpointSoybean = loadModelWithLabels(SoybeanPath)


# Transform - Apply models.Get_Transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])


# Routes
@app.get("/")
def Debug():
    return "The API is working."

## Debug




# ## Rodando a aplicação ##
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({"error": "Nenhuma imagem enviada"}), 400

#     file = request.files['image']
#     test_image = Image.open(file.stream).convert('RGB')

#     # Pré-processar a imagem
#     test_image = transform(test_image).unsqueeze(0).to(DEVICE)

#     # Previsão
#     with torch.no_grad():
#         output = Net(test_image)
#         generalProbabilities = torch.softmax(output, dim=1).cpu().numpy()
#         predicted_index = np.argmax(generalProbabilities)        
#         generalPrediction = checkpointNet["class_names"][predicted_index]

#         generalConfidence = float(generalProbabilities[0][predicted_index])

#         match(generalPrediction):
#             case "Corn":
#                 test_image = Image.open(file.stream).convert('RGB')
#                 test_image = transform(test_image).unsqueeze(0).to(DEVICE)

#                 output = Corn(test_image)
#                 probabilities = torch.softmax(output, dim=1).cpu().numpy()
#                 predicted_index = np.argmax(probabilities)        
#                 predicted_class = checkpointCorn["class_names"][predicted_index]

#                 sicknessConfidence = float(probabilities[0][predicted_index])
                
#             case "Soybean":
#                 test_image = Image.open(file.stream).convert('RGB')
#                 test_image = transform(test_image).unsqueeze(0).to(DEVICE)

#                 output = Soybean(test_image)
#                 probabilities = torch.softmax(output, dim=1).cpu().numpy()
#                 predicted_index = np.argmax(probabilities)        
#                 predicted_class = checkpointSoybean["class_names"][predicted_index]

#                 sicknessConfidence = float(probabilities[0][predicted_index])

#             case "Wheat":
#                 test_image = Image.open(file.stream).convert('RGB')
#                 test_image = transform(test_image).unsqueeze(0).to(DEVICE)

#                 output = Wheat(test_image)
#                 probabilities = torch.softmax(output, dim=1).cpu().numpy()
#                 predicted_index = np.argmax(probabilities)        
#                 predicted_class = checkpointWheat["class_names"][predicted_index]

#                 sicknessConfidence = float(probabilities[0][predicted_index])

#             case _:
#                 predicted_class = "doença não identificada."

#     return jsonify({'plant': generalPrediction.lower(), 
#                     'plantConfidence': generalConfidence,
#                     'prediction': predicted_class.lower(), 
#                     'predictionConfidence': sicknessConfidence}), 200



if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)