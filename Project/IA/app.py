"""
Para rodar a aplicação, você deve estar dentro da pasta /Project/IA
"""


# FastAPI
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import uvicorn

# Scripts auxiliares
from imports import loadModel

# System
import os
import io
from PIL import Image

# Torch
import torch
from torchvision import transforms
import torch.nn.functional as F

# Numpy
import numpy as np

from dotenv import load_dotenv


# Loading env
if load_dotenv():
    print(".env loaded successfully.") # Apply logging
    print(os.getenv("CORN"))
else:
    raise("Couldn't load the .env")


# Application
app = FastAPI()

origins = [
    "http://localhost:3000", # Atualizar CORS com .env
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

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN
## Generalist
GeneralPATH = os.getenv("GENERALIST")

## Experts
if os.path.isfile(os.getenv("CORN")):
    CORN, LOADED_CORN = loadModel(os.getenv("CORN"), device=DEVICE)
    print("CORN Model loaded successfully")

else:
    raise FileNotFoundError("couldn't identify the trained Corn model archive")

# WHEAT_PATH = os.getenv("WHEAT")
# SOYBEAN_PATH = os.getenv("SOY")


## Rede Neural ##
# GENERAL, LOADED_GENERAL = loadModelWithLabels(GeneralPATH)
# WHEAT, LOADED_WHEAT = loadModelWithLabels(WHEAT_PATH)
# SOYBEAN, LOADED_SOYBEAN = loadModelWithLabels(SOYBEAN_PATH)

# Transform - Apply models.Get_Transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

### ---- ###

# Carregando modelo de milho
"""
Por conta de usarmos um modelo treinado, temos que então importa-lo e carregar a arquitetura.
Logo após, necessitamos introduzir os pesos próprios nos nós do modelo carregado e então o modelo estará pronto para uso.
"""

# Aqui vai o carregamento de modelo

### ---- ###


# Defs
def __preprocess_image(image_file: UploadFile):
    try:
        # Tratamento Imagem
        image_data = image_file.file.read()    
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        image_tensor = transform(image).unsqueeze(0)
        return image_tensor

    except Exception as e:
        print(f"Error trying to manipulate the image: {e}")


def __corn_predict(image_tensor):

    try:
        with torch.no_grad():

            outputs = CORN(image_tensor.to(DEVICE))
            probabilities = F.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()

            confidence = probabilities[0][predicted_class_idx].item()
        
        predicted_class = LOADED_CORN["class_names"][predicted_class_idx]

        all_probabilities = {
            LOADED_CORN["class_names"][i]: float(probabilities[0][i])
            for i in range(len(LOADED_CORN["class_names"]))
        }
        
        return predicted_class, round(confidence * 100, 2), all_probabilities

    except Exception as e:
        print(f"Error trying to predict: {e}")


# Routes
## Debug
@app.get("/modelinfo")
def ModelInfo():
    return {
        "class_names": LOADED_CORN["class_names"],
        "num_classes": LOADED_CORN["num_classes"],
        "model_info": LOADED_CORN["model_info"],
    }

@app.get("/modelready")
def ModelInfo():
    return {
        "model": CORN,
    }

## Predict
@app.post("/predict")
async def Predict( file: UploadFile = File(...) ):
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only images are acceptable.")

    contents = await file.read()
    if len(contents) > 8 * 1024 * 1024:
        raise HTTPException(400, "File too large.")
    
    await file.seek(0)


    image_tensor = __preprocess_image(image_file=file)

    # Prediction
    prediction = __corn_predict(image_tensor)


    return {
        "prediction": prediction,
    }

    # return {
    #     "plant": generalPrediction.lower(), 
    #     "plantConfidence": generalConfidence,
    #     "prediction": predicted_class.lower(), 
    #     "predictionConfidence": sicknessConfidence
    #     }




if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)