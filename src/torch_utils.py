import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class OpeningsPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=6, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=5)
        )

    def forward(self, X):
        return self.layer_stack(X)
    


class FootprintGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=6, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=5)
        )

    def forward(self, X):
        return self.layer_stack(X)


class CirculationDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=6, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=5)
        )

    def forward(self, X):
        return self.layer_stack(X)
    
    
openings_predictor = OpeningsPredictor().to(device)
footprint_generator = FootprintGenerator().to(device)
circulation_detector = CirculationDetector().to(device)