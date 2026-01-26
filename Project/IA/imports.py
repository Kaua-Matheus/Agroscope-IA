# Torch
import torch

from utils.model import NeuralNetwork

from typing import NamedTuple, Optional


class Response(NamedTuple):
    Net: NeuralNetwork
    Checkpoint: Optional[dict] = None


class Colors:
    def __init__(self):
        self.RED = "\033[1;31m"
        self.GREEN = "\033[1;32m"
        self.YELLOW = "\033[1;33m"
        self.BLUE = "\033[1;34m"
        self.DEFAULT = "\033[0m"

    def INFO(self, text:str) -> str:
        return f"[{self.GREEN}INFO{self.DEFAULT}]: {text}"

    def ERROR(self, text:str) -> str:
        return f"[{self.RED}ERROR{self.DEFAULT}]: {text}"


# LoadModel
def loadModelWithLabels(model_path: str) -> Response:
    """
        Função para carregamento de modelos treinados.
        Habilitada a opção de carregar os metadados.

        params:
            model_path: Caminho do arquivo .pth do modelo
            model_class: Classe do modelo com base nn.Module

        Caso existam registros no checkpoint, o primeiro retorno da tupla será o modelo.eval() e o segundo será o checkpoint.
        Senão, somente será retornada uma tupla contendo o modelo.eval()
    """

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colors = Colors()

    try:
        checkpoint = torch.load(model_path)

        if checkpoint["model_state_dict"] != None:
            model = NeuralNetwork().to(DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(colors.INFO("Modelo c/ checkpoint carregado com sucesso."))
            return Response(model.eval(), checkpoint)
        
        else:
            model = NeuralNetwork().to(DEVICE)
            model.load_state_dict(torch.load(model_path))
            print(colors.INFO("Modelo s/ checkpoint carregado com sucesso."))
            return Response(model.eval(), None)


    except Exception as exc:
        print(colors.ERROR(exc))