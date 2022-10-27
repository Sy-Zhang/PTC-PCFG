
from model.model import *

model_factory = {
    'MMCPCFGs': MMCPCFGs,
    'VGCPCFGs': VGCPCFGs,
    'CPCFGs': CPCFGs,
    'Random': Random,
    'LeftBranching': LeftBranching,
    'RightBranching': RightBranching,
}

def get_model(name):
    return model_factory[name]