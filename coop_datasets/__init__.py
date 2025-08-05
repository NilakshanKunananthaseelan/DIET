from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .fgvc import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .stanford_cars import StanfordCars
from .imagenet import ImageNet


dataset_list = {
                "oxfordpets": OxfordPets,
                "eurosat": EuroSAT,
                "ucf101": UCF101,
                "sun397": SUN397,
                "caltech101": Caltech101,
                "dtd": DescribableTextures,
                "fgvc": FGVCAircraft,
                "food101": Food101,
                "oxfordflowers": OxfordFlowers,
                "stanford_cars": StanfordCars,
                "imagenet": ImageNet,
                }

dataset_list = {
                "OxfordPets": OxfordPets,
                "EuroSAT": EuroSAT,
                "UCF101": UCF101,
                "SUN397": SUN397,
                "Caltech101": Caltech101,
                "DescribableTextures": DescribableTextures,
                "FGVCAircraft": FGVCAircraft,
                "Food101": Food101,
                "OxfordFlowers": OxfordFlowers,
                "StanfordCars": StanfordCars,
                "ImageNet": ImageNet,
                }




def build_dataset(dataset, root_path, shots, preprocess):
    if dataset == 'imagenet':
        return dataset_list[dataset](root_path, shots, preprocess)
    else:
        return dataset_list[dataset](root_path, shots)