
import copy
import glob
import os
from shutil import move
import numpy as np
import torch

from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from tqdm import tqdm
from coop_datasets.utils import DatasetWrapper,build_data_loader
from coop_datasets import build_dataset

def replace_indexes_coop_dataset(
    dataset: torch.utils.data.Dataset, indexes, seed=0, only_mark: bool = False
):
    if not only_mark:
        rng = np.random.RandomState(seed)
        new_indexes = rng.choice(
            list(set(range(len(dataset))) - set(indexes)), size=len(indexes)
        )
        dataset[indexes] = dataset[new_indexes]
        try:
            dataset.targets[indexes] = dataset.targets[new_indexes]
        except:
            dataset.labels[indexes] = dataset.labels[new_indexes]
        else:
            dataset._labels[indexes] = dataset._labels[new_indexes]
    else:
        # Notice the -1 to make class 0 work
        for i, data in enumerate(dataset):
            if i in indexes:
                data.set_label(label=-data.label - 1)

def replace_class_coop_dataset(
    dataset: torch.utils.data.Dataset,
    class_to_replace: int,
    num_indexes_to_replace: int = None,
    seed: int = 0,
    only_mark: bool = False,
):
    if class_to_replace == -1:
        try:
            indexes = np.flatnonzero(np.ones_like(dataset.targets))
        except:
            try:
                indexes = np.flatnonzero(np.ones_like(dataset.labels))
            except:
                indexes = np.flatnonzero(np.ones_like(dataset._labels))
    else:
        try:
            indexes = np.flatnonzero(np.array(dataset.targets) == class_to_replace)
        except:
            try:
                indexes = np.flatnonzero(np.array([data.label for data in dataset]) == class_to_replace)
            except:
                indexes = np.flatnonzero(np.array(dataset._labels) == class_to_replace)
   

    if num_indexes_to_replace is not None:
        assert num_indexes_to_replace <= len(
            indexes
        ), f"Want to replace {num_indexes_to_replace} indexes but only {len(indexes)} samples in dataset"
        rng = np.random.RandomState(seed)
        indexes = rng.choice(indexes, size=num_indexes_to_replace, replace=False)
    print(f"Replacing indexes {indexes}")
    replace_indexes_coop_dataset(dataset, indexes, seed, only_mark)

from torchvision import transforms as T
def coop_dataloaders(
    batch_size=128,
    data_dir=None,
    dataset=None,
    num_workers=2,
    class_to_replace: int = -1,
    n_classes_to_replace=None,
    num_indexes_to_replace=None,
    indexes_to_replace=None,
    seed: int = 1,
    only_mark: bool = False,
    shuffle=True,
    shots=-1,
    preprocess=None,
    class_to_replace_test=[],
    args=None,
    **kwargs
):
    print(f"Using seed {seed} {dataset} {data_dir}")
    print(class_to_replace, num_indexes_to_replace, n_classes_to_replace)

    interp_mode = T.InterpolationMode.BICUBIC
    to_tensor = []
    to_tensor += [T.Resize((224, 224), interpolation=interp_mode)]
    to_tensor += [T.ToTensor()]
    normalize = T.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
    )
    to_tensor += [normalize]
    train_transform = preprocess
    test_transform = preprocess
    
   
    

    
    dataset = build_dataset(dataset,data_dir,shots,None)

    train_set = dataset.train_x
    test_set = dataset.test
    valid_set = dataset.val
    classnames = dataset._classnames
    num_classes = dataset._num_classes
    label_to_class = dataset._lab2cname
   

   

    
    if class_to_replace==-1  and num_indexes_to_replace is None and n_classes_to_replace is not None:


        assert len(class_to_replace_test) == 0, "Cannot specify `class_to_replace_test` when `n_classes_to_replace` is specified"
        rng = np.random.RandomState(seed)

        sorted_classes = sorted(label_to_class.keys())  # Ensure fixed order
        unl_targets = rng.choice(sorted_classes, n_classes_to_replace, replace=False)
        

        print(f"Fixed Seed {seed}, Sorted Class List: {sorted_classes}, Selected: {unl_targets}")
        
        

        # class_to_replace = unl_targets[0] #for now only one class at a time
        class_to_replace = unl_targets[0]
        args.class_to_replace = class_to_replace
        class_to_replace_test = [class_to_replace]
        args.class_to_replace_test = class_to_replace_test

        print(f"Replacing class {class_to_replace}")
        print(f"Replacing class Test {class_to_replace_test}")

    
    
    print('######',class_to_replace)
    if class_to_replace!=-1:
        print('Replacing classes')
        
        replace_class_coop_dataset(
            train_set,
            class_to_replace,
            num_indexes_to_replace=num_indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

        if num_indexes_to_replace is None or num_indexes_to_replace == 4454:
            # test_set =[data for data in test_set if data.label == class_to_replace]
            print(f'Length of test set before: {len(test_set)}')
            _test_set = []

            if len(class_to_replace_test)==0:
                for data in test_set:
                    if data.label != class_to_replace:
                        _test_set.append(data)
            else:
                for data in test_set:
                    if data.label not in class_to_replace_test:
                        _test_set.append(data)

            test_set = _test_set
            print(f'Length of test set after: {len(test_set)}')
                    
    
    if indexes_to_replace is not None:
        replace_indexes_coop_dataset(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 8, "pin_memory": True}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        DatasetWrapper(train_set,224,transform=train_transform,is_train=True),
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        DatasetWrapper(valid_set,224,transform=test_transform,is_train=False),
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        DatasetWrapper(test_set,224,transform=test_transform,is_train=False),
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, val_loader, test_loader, classnames, num_classes, label_to_class




class MarkedDataset(Dataset):
    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if label<0:
            label = -label - 1

        return img, label