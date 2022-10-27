from .datasets.pentathlon_dataset import PentathlonDataset
from .datasets.howto100m import HowTo100M

dataset_factory = {
    'HowTo100M': HowTo100M,
    'MSRVTT-Pentathlon': PentathlonDataset,
    'DiDeMo-Pentathlon': PentathlonDataset,
    'YouCook2-Pentathlon': PentathlonDataset,
}

def get_dataset(name):
    return dataset_factory[name]
