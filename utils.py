"""
    setup model and datasets
"""


import copy
import os
import random

# from advertorch.utils import NormalizeByChannelMeanStd
import shutil
import sys
import time

import numpy as np
import torch
from dataset import *
from dataset import TinyImageNet
from imagenet import prepare_data

from torchvision import transforms


__all__ = [
    "setup_model_dataset",
    "AverageMeter",
    "warmup_lr",
    "save_checkpoint",
    "setup_seed",
    "accuracy",
]


def warmup_lr(epoch, step, optimizer, one_epoch_step, args):
    overall_steps = args.warmup * one_epoch_step
    current_steps = epoch * one_epoch_step + step

    lr = args.lr * current_steps / overall_steps
    lr = min(lr, args.lr)

    for p in optimizer.param_groups:
        p["lr"] = lr


def save_checkpoint(
    state, is_SA_best, save_path, pruning, filename="checkpoint.pth.tar"
):
    filepath = os.path.join(save_path, str(pruning) + filename)
    torch.save(state, filepath)
    if is_SA_best:
        shutil.copyfile(
            filepath, os.path.join(save_path, str(pruning) + "model_SA_best.pth.tar")
        )


def load_checkpoint(device, save_path, pruning, filename="checkpoint.pth.tar"):
    filepath = os.path.join(save_path, str(pruning) + filename)
    if os.path.exists(filepath):
        print("Load checkpoint from:{}".format(filepath))
        return torch.load(filepath, device)
    print("Checkpoint not found! path:{}".format(filepath))
    return None


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dataset_convert_to_train(dataset):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    dataset.transform = train_transform
    dataset.train = False


def dataset_convert_to_test(dataset, args=None):
    if args.dataset == "TinyImagenet":
        test_transform = transforms.Compose([])
    else:
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    dataset.transform = test_transform
    dataset.train = False

from coop_datasets.utils import DatasetWrapper
def coopdataset_convert_to_test(dataset, args=None):
    if args.dataset == "TinyImagenet":
        test_transform = transforms.Compose([])
    else:
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    dataset = DatasetWrapper(dataset,224,transform=test_transform,is_train=False)
    


def setup_model_dataset(args,preprocess=None):
    rng = np.random.RandomState(args.seed)
    if args.dataset == "cifar10":
        classes = 10
        classnames = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        
        if args.class_to_replace==-1:
            args.class_to_replace = rng.randint(0, classes)
            print(f"Class to replace {args.class_to_replace}")  
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        )
        train_full_loader, val_loader, _ = cifar10_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers,preprocess=preprocess,
        )
        marked_loader, _, test_loader = cifar10_dataloaders(
            batch_size=args.batch_size,
            data_dir=args.data,
            num_workers=args.workers,
            class_to_replace=args.class_to_replace,
            num_indexes_to_replace=args.num_indexes_to_replace,
            indexes_to_replace=args.indexes_to_replace,
            seed=args.seed,
            only_mark=True,
            shuffle=True,
            no_aug=args.no_aug,
            preprocess=preprocess
        )

        if args.train_seed is None:
            args.train_seed = args.seed
        setup_seed(args.train_seed)

        

        setup_seed(args.train_seed)

        
        return train_full_loader, val_loader, test_loader, marked_loader,classnames
    elif args.dataset == "svhn":
        classes = 10
        classnames = [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
        ]
        if args.class_to_replace==-1:
            args.class_to_replace = rng.randint(0, classes)
            print(f"Class to replace {args.class_to_replace}") 
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052]
        )
        train_full_loader, val_loader, _ = svhn_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers,
            preprocess=preprocess
        )
        marked_loader, _, test_loader = svhn_dataloaders(
            batch_size=args.batch_size,
            data_dir=args.data,
            num_workers=args.workers,
            class_to_replace=args.class_to_replace,
            num_indexes_to_replace=args.num_indexes_to_replace,
            indexes_to_replace=args.indexes_to_replace,
            seed=args.seed,
            only_mark=True,
            shuffle=True,
            preprocess=preprocess
        )
        
        return train_full_loader, val_loader, test_loader, marked_loader,classnames
    elif args.dataset == "cifar100":
        classes = 100
        classnames = [
            "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
            "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
            "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
            "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
            "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
            "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
            "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
            "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
            "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
            "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
        ]
        if args.class_to_replace==-1:
            args.class_to_replace = rng.randint(0, classes)
            print(f"Class to replace {args.class_to_replace}") 
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762]
        )
        train_full_loader, val_loader, _ = cifar100_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers,
            preprocess=preprocess
        )
        marked_loader, _, test_loader = cifar100_dataloaders(
            batch_size=args.batch_size,
            data_dir=args.data,
            num_workers=args.workers,
            class_to_replace=args.class_to_replace,
            num_indexes_to_replace=args.num_indexes_to_replace,
            indexes_to_replace=args.indexes_to_replace,
            seed=args.seed,
            only_mark=True,
            shuffle=True,
            no_aug=args.no_aug,
            preprocess=preprocess
        )
        
        return  train_full_loader, val_loader, test_loader, marked_loader,classnames
    elif args.dataset == "TinyImagenet":
        classes = 200
        classnames = ['goldfish', 'European fire salamander', 'bullfrog', 'tailed frog', 'American alligator', 'boa constrictor', 'trilobite', 'scorpion', 'black widow', 'tarantula', 'centipede', 'goose', 'koala', 'jellyfish', 'brain coral', 'snail', 'slug', 'sea slug', 'American lobster', 'spiny lobster', 'black stork', 'king penguin', 'albatross', 'dugong', 'Chihuahua', 'Yorkshire terrier', 'golden retriever', 'Labrador retriever', 'German shepherd', 'standard poodle', 'tabby', 'Persian cat', 'Egyptian cat', 'cougar', 'lion', 'brown bear', 'ladybug', 'fly', 'bee', 'grasshopper', 'walking stick', 'cockroach', 'mantis', 'dragonfly', 'monarch', 'sulphur butterfly', 'sea cucumber', 'guinea pig', 'hog', 'ox', 'bison', 'bighorn', 'gazelle', 'Arabian camel', 'orangutan', 'chimpanzee', 'baboon', 'African elephant', 'lesser panda', 'abacus', 'academic gown', 'altar', 'apron', 'backpack', 'bannister', 'barbershop', 'barn', 'barrel', 'basketball', 'bathtub', 'beach wagon', 'beacon', 'beaker', 'beer bottle', 'bikini', 'binoculars', 'birdhouse', 'bow tie', 'brass', 'broom', 'bucket', 'bullet train', 'butcher shop', 'candle', 'cannon', 'cardigan', 'cash machine', 'CD player', 'chain', 'chest', 'Christmas stocking', 'cliff dwelling', 'computer keyboard', 'confectionery', 'convertible', 'crane', 'dam', 'desk', 'dining table', 'drumstick', 'dumbbell', 'flagpole', 'fountain', 'freight car', 'frying pan', 'fur coat', 'gasmask', 'go-kart', 'gondola', 'hourglass', 'iPod', 'jinrikisha', 'kimono', 'lampshade', 'lawn mower', 'lifeboat', 'limousine', 'magnetic compass', 'maypole', 'military uniform', 'miniskirt', 'moving van', 'nail', 'neck brace', 'obelisk', 'oboe', 'organ', 'parking meter', 'pay-phone', 'picket fence', 'pill bottle', 'plunger', 'pole', 'police van', 'poncho', 'pop bottle', "potter's wheel", 'projectile', 'punching bag', 'reel', 'refrigerator', 'remote control', 'rocking chair', 'rugby ball', 'sandal', 'school bus', 'scoreboard', 'sewing machine', 'snorkel', 'sock', 'sombrero', 'space heater', 'spider web', 'sports car', 'steel arch bridge', 'stopwatch', 'sunglasses', 'suspension bridge', 'swimming trunks', 'syringe', 'teapot', 'teddy', 'thatch', 'torch', 'tractor', 'triumphal arch', 'trolleybus', 'turnstile', 'umbrella', 'vestment', 'viaduct', 'volleyball', 'water jug', 'water tower', 'wok', 'wooden spoon', 'comic book', 'plate', 'guacamole', 'ice cream', 'ice lolly', 'pretzel', 'mashed potato', 'cauliflower', 'bell pepper', 'mushroom', 'orange', 'lemon', 'banana', 'pomegranate', 'meat loaf', 'pizza', 'potpie', 'espresso', 'alp', 'cliff', 'coral reef', 'lakeside', 'seashore', 'acorn']
        classnames = [c.lower() for c in classnames]
        
        
        if args.class_to_replace==-1:
            args.class_to_replace = rng.randint(0, classes)
            print(f"Class to replace {args.class_to_replace}") 
        normalization = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        train_full_loader, val_loader, test_loader = TinyImageNet(args,preprocess=preprocess).data_loaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers,

        )
        # train_full_loader, val_loader, test_loader =None, None,None
        marked_loader, _, _ = TinyImageNet(args,preprocess=preprocess).data_loaders(
            batch_size=args.batch_size,
            data_dir=args.data,
            num_workers=args.workers,
            class_to_replace=args.class_to_replace,
            num_indexes_to_replace=args.num_indexes_to_replace,
            indexes_to_replace=args.indexes_to_replace,
            seed=args.seed,
            only_mark=True,
            shuffle=True,
            
        )
        

       
        return train_full_loader, val_loader, test_loader, marked_loader,classnames

    elif args.dataset == "imagenet":
        classes = 1000
        if args.class_to_replace==-1:
            args.class_to_replace = rng.randint(0, classes)
            print(f"Class to replace {args.class_to_replace}") 
        normalization = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        train_ys = torch.load(args.train_y_file)
        val_ys = torch.load(args.val_y_file)
       
       
        if args.class_to_replace is None:
            loaders = prepare_data(dataset="imagenet", batch_size=args.batch_size)
            train_loader, val_loader = loaders["train"], loaders["val"]
            return model, train_loader, val_loader
        else:
            train_subset_indices = torch.ones_like(train_ys)
            val_subset_indices = torch.ones_like(val_ys)
            train_subset_indices[train_ys == args.class_to_replace] = 0
            val_subset_indices[val_ys == args.class_to_replace] = 0
            loaders = prepare_data(
                dataset="imagenet",
                batch_size=args.batch_size,
                train_subset_indices=train_subset_indices,
                val_subset_indices=val_subset_indices,
            )
            retain_loader = loaders["train"]
            forget_loader = loaders["fog"]
            val_loader = loaders["val"]
            return retain_loader, forget_loader, val_loader

    elif args.dataset == "cifar100_no_val":
        classes = 100
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762]
        )
        train_set_loader, val_loader, test_loader = cifar100_dataloaders_no_val(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )

    elif args.dataset == "cifar10_no_val":
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        )
        train_set_loader, val_loader, test_loader = cifar10_dataloaders_no_val(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )


    else:
        raise ValueError("Dataset not supprot yet !")
    # import pdb;pdb.set_trace()

    
    
    return  train_set_loader, val_loader, test_loader


def setup_seed(seed):
    print("setup random seed = {}".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class NormalizeByChannelMeanStd(torch.nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return self.normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return "mean={}, std={}".format(self.mean, self.std)

    def normalize_fn(self, tensor, mean, std):
        """Differentiable version of torchvision.functional.normalize"""
        # here we assume the color channel is in at dim=1
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        return tensor.sub(mean).div(std)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

   
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def run_commands(gpus, commands, call=False, dir="commands", shuffle=True, delay=0.5):
    if len(commands) == 0:
        return
    if os.path.exists(dir):
        shutil.rmtree(dir)
    if shuffle:
        random.shuffle(commands)
        random.shuffle(gpus)
    os.makedirs(dir, exist_ok=True)

    fout = open("stop_{}.sh".format(dir), "w")
    print("kill $(ps aux|grep 'bash " + dir + "'|awk '{print $2}')", file=fout)
    fout.close()

    n_gpu = len(gpus)
    for i, gpu in enumerate(gpus):
        i_commands = commands[i::n_gpu]
        if len(i_commands) == 0:
            continue
        prefix = "CUDA_VISIBLE_DEVICES={} ".format(gpu)

        sh_path = os.path.join(dir, "run{}.sh".format(i))
        fout = open(sh_path, "w")
        for com in i_commands:
            print(prefix + com, file=fout)
        fout.close()
        if call:
            os.system("bash {}&".format(sh_path))
            time.sleep(delay)


def get_loader_from_dataset(dataset, batch_size, seed=1, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=shuffle
    )


def get_unlearn_loader(marked_loader, args):
    forget_dataset = copy.deepcopy(marked_loader.dataset)
    marked = forget_dataset.targets < 0
    forget_dataset.data = forget_dataset.data[marked]
    forget_dataset.targets = -forget_dataset.targets[marked] - 1
    forget_loader = get_loader_from_dataset(
        forget_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=True
    )
    retain_dataset = copy.deepcopy(marked_loader.dataset)
    marked = retain_dataset.targets >= 0
    retain_dataset.data = retain_dataset.data[marked]
    retain_dataset.targets = retain_dataset.targets[marked]
    retain_loader = get_loader_from_dataset(
        retain_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=True
    )
    print("datasets length: ", len(forget_dataset), len(retain_dataset))
    return forget_loader, retain_loader


def get_poisoned_loader(poison_loader, unpoison_loader, test_loader, poison_func, args):
    poison_dataset = copy.deepcopy(poison_loader.dataset)
    poison_test_dataset = copy.deepcopy(test_loader.dataset)

    poison_dataset.data, poison_dataset.targets = poison_func(
        poison_dataset.data, poison_dataset.targets
    )
    poison_test_dataset.data, poison_test_dataset.targets = poison_func(
        poison_test_dataset.data, poison_test_dataset.targets
    )

    full_dataset = torch.utils.data.ConcatDataset(
        [unpoison_loader.dataset, poison_dataset]
    )

    poisoned_loader = get_loader_from_dataset(
        poison_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=False
    )
    poisoned_full_loader = get_loader_from_dataset(
        full_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=True
    )
    poisoned_test_loader = get_loader_from_dataset(
        poison_test_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=False
    )

    return poisoned_loader, unpoison_loader, poisoned_full_loader, poisoned_test_loader

import os
import json
from datetime import datetime

def setup_logging(args):
    """Setup logging directory and file for the current run."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(args.save_path,'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create a unique filename based on timestamp and method name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_name = args.unlearn.lower()
    dataset_name = getattr(args, 'dataset', 'unknown').lower()
    seed = args.seed
    filename = f"{method_name}_{dataset_name}_{seed}_{timestamp}.json"
    log_file = os.path.join(log_dir, filename)
    
    # Initialize log data structure
    args_dict = {}
    args_dict = {}
    for k, v in vars(args).items():
        if isinstance(v, (str, int, float, bool, list, dict, type(None))):
            # Convert all elements in lists to built-in types if necessary
            if isinstance(v, list):
                def convert_item(item):
                    if isinstance(item, (str, int, float, bool, dict, type(None))):
                        return item
                    elif hasattr(item, "item") and callable(item.item):
                        return item.item()
                    else:
                        return str(item)
                args_dict[k] = [convert_item(item) for item in v]
            elif hasattr(v, "item") and callable(v.item):
                args_dict[k] = v.item()
            else:
                args_dict[k] = v

    log_data = {
        "method": method_name,
        "dataset": dataset_name,
        "timestamp": timestamp,
        "args": args_dict,
        "epochs": []
    }
    
    # Save initial log data
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    return log_file

def log_metrics(log_file, epoch, metrics):
    """Log metrics for the current epoch."""
    try:
        # Read existing log data
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        
        # Add epoch metrics
        epoch_data = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        log_data["epochs"].append(epoch_data)
        
        # Write updated log data
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
            
    except Exception as e:
        print(f"Warning: Could not log metrics: {e}")


def save_eval_result(args,evaluation_result):
    # Save evaluation result as json for each seed
        save_dir = args.save_path if hasattr(args, "save_dir") else "."
        os.makedirs(save_dir, exist_ok=True)
        seed = getattr(args, "seed", "unknown")
        
        json_path = os.path.join(save_dir, f"evaluation_result_seed_{seed}.json")
        # Convert any non-serializable values to float
        serializable_result = {}
        for key,values in evaluation_result.items():
            if isinstance(values,dict):
                for k, v in values.items():
                    try:
                        serializable_result[k] = float(v)
                    except Exception:
                        serializable_result[k] = v
            else:
                try:
                        serializable_result[key] = float(values)
                except Exception:
                    serializable_result[key] = values
        
        with open(json_path, "w") as f:
            json.dump(serializable_result, f, indent=4)
        print(f"Saved evaluation result to {json_path}")


