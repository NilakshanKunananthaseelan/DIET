import os
import copy
import json
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
import geoopt
import numpy as np

import arg_parser
import utils
from utils import *
from dataset_utils import coop_dataloaders, MarkedDataset
from lora_hyp import run_lora
import clip
from unlearn_utils import (
    get_synthetic_orthogonal_prototypes,
    get_synthetic_random_prototypes,
    norm_clip,
    BusePenalty
)
from clip_utils import *

# List of supported datasets for COOP-style loaders
COOP_DATASETS = [
    'OxfordPets', 'OxfordFlowers', 'FGVCAircraft', 'DescribableTextures', 'EuroSAT',
    'StanfordCars', 'Food101', 'SUN397', 'Caltech101', 'UCF101'
]

# Custom templates for prompt engineering per dataset
CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    'cifar10': 'a photo of a {}.',
    'cifar100': 'a photo of a {}.',
    'svhn': 'a street sign of the number {}.',
    'CELEBHQ': "a photo of a face of {}.",
    'TinyImagenet': 'a photo of a {}.',
    'Imagenette': 'a photo of a {}.',
    'CUB200': 'a photo of a {}, a type of bird.',
}

class NormalizeByChannelMeanStd(nn.Module):
    """
    Module for normalizing images by channel mean and std.
    """
    def __init__(self, mean, std):
        super().__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return self.normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return f"mean={self.mean}, std={self.std}"

    def normalize_fn(self, tensor, mean, std):
        # Assumes channel dimension is at dim=1
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        return tensor.sub(mean).div(std)

def replace_loader_dataset(dataset, batch_size, seed=1, shuffle=True):
    """
    Utility to create a DataLoader with a fixed seed for reproducibility.
    """
    utils.setup_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=shuffle,
    )

def setup_coop_dataloaders(args, preprocess=None):
    """
    Prepare dataloaders for COOP-style datasets, including splitting into
    forget/retain/test sets as required for unlearning experiments.
    """
    seed = args.seed

    # Load full training and validation sets
    train_loader_full, val_loader, _, classnames, num_classes, label_to_class = coop_dataloaders(
        batch_size=args.batch_size,
        dataset=args.dataset,
        num_workers=args.workers,
        is_train=True,
        preprocess=preprocess,
        data_dir=args.data_dir
    )

    print('Loading Marked Loader')
    

    # Load marked loader for forget/retain split
    marked_loader, _, test_loader, _, _, _ = coop_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        dataset=args.dataset,
        num_workers=args.workers,
        n_classes_to_replace=1,
        class_to_replace=args.class_to_replace,
        seed=args.seed,
        only_mark=True,
        shuffle=True,
        args=args,
        shots=args.shots,
        preprocess=preprocess
    )

    # Loader for evaluation (all marked samples)
    marked_eval_loader, _, _, _, _, _ = coop_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        dataset=args.dataset,
        num_workers=args.workers,
        n_classes_to_replace=1,
        class_to_replace=args.class_to_replace,
        seed=args.seed,
        only_mark=True,
        shuffle=True,
        args=args,
        shots=-1,
        preprocess=preprocess
    )

    # Split marked dataset into forget and retain sets
    forget_dataset = []
    for x, y in tqdm(marked_loader.dataset, desc="Splitting forget and retain", total=len(marked_loader.dataset)):
        if y < 0:
            forget_dataset.append((x, y))

    forget_loader = replace_loader_dataset(
        MarkedDataset(forget_dataset, num_classes),
        batch_size=args.batch_size, seed=seed, shuffle=True
    )

    # Split marked eval dataset into forget and retain sets
    forget_eval_dataset = []
    retain_eval_dataset = []
    for x, y in tqdm(marked_eval_loader.dataset, desc="Splitting forget for eval", total=len(marked_eval_loader.dataset)):
        if y < 0:
            forget_eval_dataset.append((x, y))
        else:
            retain_eval_dataset.append((x, y))

    forget_eval_loader = replace_loader_dataset(
        MarkedDataset(forget_eval_dataset, num_classes),
        batch_size=args.batch_size, seed=seed, shuffle=False
    )
    retain_eval_loader = replace_loader_dataset(
        MarkedDataset(retain_eval_dataset, num_classes),
        batch_size=args.batch_size, seed=seed, shuffle=False
    )

    print(f"number of samples retain dataset {len(retain_eval_dataset)}")
    print(f"number of samples k-shot forget dataset {len(forget_dataset)}")
    print(f"number of samples forget dataset {len(forget_eval_dataset)}")
    print(f"number of samples test dataset {len(test_loader.dataset)}")
    print(f"number of samples train dataset {len(train_loader_full.dataset)}")
    print(f"number of classnames {len(classnames)}")

    # Return all relevant dataloaders in an ordered dict
    unlearn_data_loaders = OrderedDict(
        retain=retain_eval_loader,
        forget=forget_loader,
        test=test_loader,
        forget_eval=forget_eval_loader
    )
    return unlearn_data_loaders, classnames

def main():
    """
    Main entry point for running the unlearning experiment.
    """
    args = arg_parser.parse_args()
    print(vars(args))

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_path, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed
    args.train_seed = getattr(args, 'train_seed', seed)

    # Load CLIP model and preprocessing
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()
    logit_scale = 100

    print("Preparing dataset.")
    # Prepare dataloaders and classnames
    if args.dataset in COOP_DATASETS:
        unlearn_data_loaders, classnames = setup_coop_dataloaders(args, preprocess=preprocess)
        args.num_classes = len(classnames)
    else:
        # For non-COOP datasets, use generic setup
        train_loader_full, val_loader, test_loader, marked_loader, classnames = utils.setup_model_dataset(args, preprocess)
        args.num_classes = len(classnames)
        forget_dataset = copy.deepcopy(marked_loader.dataset)

        # Handle dataset-specific logic for splitting forget/retain
        if args.dataset == "svhn":
            try:
                marked = forget_dataset.targets < 0
            except Exception:
                marked = forget_dataset.labels < 0
            forget_dataset.data = forget_dataset.data[marked]
            try:
                forget_dataset.targets = -forget_dataset.targets[marked] - 1
            except Exception:
                forget_dataset.labels = -forget_dataset.labels[marked] - 1
            forget_loader = replace_loader_dataset(forget_dataset, batch_size=args.batch_size, seed=seed, shuffle=True)

            retain_dataset = copy.deepcopy(marked_loader.dataset)
            try:
                marked = retain_dataset.targets >= 0
            except Exception:
                marked = retain_dataset.labels >= 0
            retain_dataset.data = retain_dataset.data[marked]
            try:
                retain_dataset.targets = retain_dataset.targets[marked]
            except Exception:
                retain_dataset.labels = retain_dataset.labels[marked]
            retain_loader = replace_loader_dataset(retain_dataset, batch_size=args.batch_size, seed=seed, shuffle=True)
            assert len(forget_dataset) + len(retain_dataset) == len(train_loader_full.dataset)
        else:
            try:
                marked = forget_dataset.targets < 0
                forget_dataset.data = forget_dataset.data[marked]
                forget_dataset.targets = -forget_dataset.targets[marked] - 1
                forget_loader = replace_loader_dataset(forget_dataset, batch_size=args.batch_size, seed=seed, shuffle=True)

                retain_dataset = copy.deepcopy(marked_loader.dataset)
                marked = retain_dataset.targets >= 0
                retain_dataset.data = retain_dataset.data[marked]
                retain_dataset.targets = retain_dataset.targets[marked]
                retain_loader = replace_loader_dataset(retain_dataset, batch_size=args.batch_size, seed=seed, shuffle=True)
                assert len(forget_dataset) + len(retain_dataset) == len(train_loader_full.dataset)
            except Exception:
                marked = forget_dataset.targets < 0
                forget_dataset.imgs = forget_dataset.imgs[marked]
                forget_dataset.targets = -forget_dataset.targets[marked] - 1
                forget_loader = replace_loader_dataset(forget_dataset, batch_size=args.batch_size, seed=seed, shuffle=True)

                retain_dataset = copy.deepcopy(marked_loader.dataset)
                marked = retain_dataset.targets >= 0
                retain_dataset.imgs = retain_dataset.imgs[marked]
                retain_dataset.targets = retain_dataset.targets[marked]
                retain_loader = replace_loader_dataset(retain_dataset, batch_size=args.batch_size, seed=seed, shuffle=True)
                assert len(forget_dataset) + len(retain_dataset) == len(train_loader_full.dataset)

        # Deepcopy for evaluation loader
        orig_forget_loader = copy.deepcopy(forget_loader)

        # Select k-shot forget samples using the given seed
        num_forget_samples = args.shots
        rng = np.random.RandomState(seed)
        if hasattr(forget_dataset, 'data'):
            total_indices = np.arange(len(forget_dataset.data))
        elif hasattr(forget_dataset, 'imgs'):
            total_indices = np.arange(len(forget_dataset.imgs))
        else:
            total_indices = np.arange(len(forget_dataset))
        if len(total_indices) < num_forget_samples:
            raise ValueError(f"Forget dataset has fewer than {num_forget_samples} samples.")
        selected_indices = rng.choice(total_indices, size=num_forget_samples, replace=False)

        # Subset the forget dataset to selected indices
        if hasattr(forget_dataset, 'data'):
            forget_dataset.data = forget_dataset.data[selected_indices]
            if hasattr(forget_dataset, 'targets'):
                forget_dataset.targets = forget_dataset.targets[selected_indices]
            elif hasattr(forget_dataset, 'labels'):
                forget_dataset.labels = forget_dataset.labels[selected_indices]
        elif hasattr(forget_dataset, 'imgs'):
            forget_dataset.imgs = [forget_dataset.imgs[i] for i in selected_indices]
            if hasattr(forget_dataset, 'targets'):
                forget_dataset.targets = forget_dataset.targets[selected_indices]
            elif hasattr(forget_dataset, 'labels'):
                forget_dataset.labels = forget_dataset.labels[selected_indices]
        else:
            if hasattr(forget_dataset, 'indices'):
                forget_dataset.indices = [forget_dataset.indices[i] for i in selected_indices]

        forget_train_loader = replace_loader_dataset(
            forget_dataset, batch_size=args.batch_size, seed=seed, shuffle=True
        )
        retain_train_loader = replace_loader_dataset(
            retain_dataset, batch_size=args.batch_size, seed=seed, shuffle=True
        )

        print(f"number of retain dataset {len(retain_dataset)}")
        print(f"number of forget dataset {len(forget_dataset)}")
        unlearn_data_loaders = OrderedDict(
            retain=retain_train_loader,
            forget_eval=orig_forget_loader,
            forget=forget_train_loader,
            test=test_loader,
        )

    # Extract relevant dataloaders
    test_loader = unlearn_data_loaders['test']
    forget_loader = unlearn_data_loaders['forget']

    # Set up hyperbolic manifold for prototypes
    args.manifold = geoopt.PoincareBall(c=1)
    clip_model.cuda()

    print(len(forget_loader.dataset))
    print(args.norm_r)

    # Prepare class templates and prototype texts
    template = CUSTOM_TEMPLATES[args.dataset]
    proto_classes = [classnames[i] for i in range(len(classnames)) if i != args.class_to_replace]
    proto_texts = [template.format(classname.replace('_', ' ')) for classname in proto_classes]

    # Tokenize and encode prototype texts
    texts = clip.tokenize(proto_texts).cuda()
    proto_embeddings = clip_model.encode_text(texts).detach()
    proto_type = getattr(args, "prototype_type", "hyp")
    args.proto_type = proto_type

    # Generate prototypes according to the specified type
    if proto_type == "random":
        proto_h = get_synthetic_random_prototypes(args, 512, args.manifold)
    elif proto_type == 'eucl':
        proto = proto_embeddings
        proto = norm_clip(proto, r=args.norm_r)
        proto_h = args.manifold.expmap0(proto)
    else:
        raise ValueError(f"Unknown prototype_type: {proto_type}")

    
    loss_fn = BusePenalty(0)
    print(vars(args))

    # Run LoRA-based unlearning
    clip_model = run_lora(
        args,
        clip_model,
        logit_scale,
        classnames,
        CUSTOM_TEMPLATES[args.dataset],
        forget_loader,
        test_loader,
        loss_fn=loss_fn,
        proto_h=proto_h
    )

    evaluation_result = {}

    # Evaluate accuracy on all splits
    if "new_accuracy" not in evaluation_result:
        accuracy = {}
        for name, loader in unlearn_data_loaders.items():
            val_acc = evaluate_lora(args, clip_model, loader, classnames=classnames, template=template)
            accuracy[name] = val_acc
            print(f"**** Final {name} accuracy: {val_acc:.2f}. ****\n")
        evaluation_result["accuracy"] = accuracy
        utils.save_eval_result(args, evaluation_result)

if __name__ == "__main__":
    main()