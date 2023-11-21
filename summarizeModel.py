import torch
import numpy as np
from train import MODEL

VECTOR_SIZE = 4
EXPANSION_FACTOR = 8
MAX_SEQ_LENGTH = 457

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL.to(device)
model_parameters = filter(lambda p: p.requires_grad, MODEL.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Parameter Count: " + str(params))