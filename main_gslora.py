#!/usr/bin/env python3

import os
import sys
import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import argparse

import clip
from utils import (
    cifar10_dataloaders, cifar100_dataloaders, svhn_dataloaders, TinyImageNet,
    setup_seed, 
)
import utils

from clip_utils import cls_acc, clip_classifier, pre_load_features
from loralib.utils import (
    INDEX_POSITIONS_VISION, INDEX_POSITIONS_TEXT,
    save_lora, load_lora, apply_lora,
    mark_only_lora_as_trainable, get_lora_parameters
)
from dataset_utils import coop_dataloaders, MarkedDataset

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Check CUDA availability
if torch.cuda.is_available():
    print("CUDA is available.")
    num_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices: {num_devices}")
    for i in range(num_devices):
        device = torch.cuda.get_device_properties(i)
        print(f"Device {i}:")
        print(f"  Name: {device.name}")
        print(f"  Compute Capability: {device.major}.{device.minor}")
        print(f"  Total Memory: {device.total_memory / 1024**2} MB")

# Dataset/classname lists
tiny_imagenet_cls = [
    'goldfish', 'European fire salamander', 'bullfrog', 'tailed frog', 'American alligator', 'boa constrictor', 'trilobite', 'scorpion', 'black widow', 'tarantula', 'centipede', 'goose', 'koala', 'jellyfish', 'brain coral', 'snail', 'slug', 'sea slug', 'American lobster', 'spiny lobster', 'black stork', 'king penguin', 'albatross', 'dugong', 'Chihuahua', 'Yorkshire terrier', 'golden retriever', 'Labrador retriever', 'German shepherd', 'standard poodle', 'tabby', 'Persian cat', 'Egyptian cat', 'cougar', 'lion', 'brown bear', 'ladybug', 'fly', 'bee', 'grasshopper', 'walking stick', 'cockroach', 'mantis', 'dragonfly', 'monarch', 'sulphur butterfly', 'sea cucumber', 'guinea pig', 'hog', 'ox', 'bison', 'bighorn', 'gazelle', 'Arabian camel', 'orangutan', 'chimpanzee', 'baboon', 'African elephant', 'lesser panda', 'abacus', 'academic gown', 'altar', 'apron', 'backpack', 'bannister', 'barbershop', 'barn', 'barrel', 'basketball', 'bathtub', 'beach wagon', 'beacon', 'beaker', 'beer bottle', 'bikini', 'binoculars', 'birdhouse', 'bow tie', 'brass', 'broom', 'bucket', 'bullet train', 'butcher shop', 'candle', 'cannon', 'cardigan', 'cash machine', 'CD player', 'chain', 'chest', 'Christmas stocking', 'cliff dwelling', 'computer keyboard', 'confectionery', 'convertible', 'crane', 'dam', 'desk', 'dining table', 'drumstick', 'dumbbell', 'flagpole', 'fountain', 'freight car', 'frying pan', 'fur coat', 'gasmask', 'go-kart', 'gondola', 'hourglass', 'iPod', 'jinrikisha', 'kimono', 'lampshade', 'lawn mower', 'lifeboat', 'limousine', 'magnetic compass', 'maypole', 'military uniform', 'miniskirt', 'moving van', 'nail', 'neck brace', 'obelisk', 'oboe', 'organ', 'parking meter', 'pay-phone', 'picket fence', 'pill bottle', 'plunger', 'pole', 'police van', 'poncho', 'pop bottle', "potter's wheel", 'projectile', 'punching bag', 'reel', 'refrigerator', 'remote control', 'rocking chair', 'rugby ball', 'sandal', 'school bus', 'scoreboard', 'sewing machine', 'snorkel', 'sock', 'sombrero', 'space heater', 'spider web', 'sports car', 'steel arch bridge', 'stopwatch', 'sunglasses', 'suspension bridge', 'swimming trunks', 'syringe', 'teapot', 'teddy', 'thatch', 'torch', 'tractor', 'triumphal arch', 'trolleybus', 'turnstile', 'umbrella', 'vestment', 'viaduct', 'volleyball', 'water jug', 'water tower', 'wok', 'wooden spoon', 'comic book', 'plate', 'guacamole', 'ice cream', 'ice lolly', 'pretzel', 'mashed potato', 'cauliflower', 'bell pepper', 'mushroom', 'orange', 'lemon', 'banana', 'pomegranate', 'meat loaf', 'pizza', 'potpie', 'espresso', 'alp', 'cliff', 'coral reef', 'lakeside', 'seashore', 'acorn'
]
coop_datasets = [
    'OxfordPets', 'OxfordFlowers', 'FGVCAircraft', 'DescribableTextures', 'EuroSAT',
    'StanfordCars', 'Food101', 'SUN397', 'Caltech101', 'UCF101'
]

def parse_args():
    parser = argparse.ArgumentParser(description='GSLoRA Training Script')
    
    # Basic arguments
    parser.add_argument('--root_path', type=str, default="")
    parser.add_argument('--data_dir',default="")
    parser.add_argument('--dataset', type=str, default='OxfordFlowers')
    parser.add_argument('--n_iters', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=70)
    parser.add_argument('--encoder', type=str, default="vision")
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--class_to_replace', type=int, default=-1)
    parser.add_argument('--noisy_gating', action='store_true')
    parser.add_argument('--unlearn', action='store_true', default=True)
    parser.add_argument('--position', type=str, default="top3")
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--task_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--backbone', type=str, default="ViT-B/16")
    # Remove any path containing identity information
    parser.add_argument('--clip_path', type=str, default="clip_models/ViT-B-16.pt")
    parser.add_argument('--save_path', type=str, default="checkpoints/vanilla_moe/gslora_test")
    parser.add_argument('--vanilla', action='store_true', default=True)
    parser.add_argument('--similarity_mul', type=float, default=0.5)
    parser.add_argument('--single_router', action='store_true', default=True)
    parser.add_argument('--freeze_experts', action='store_true')
    parser.add_argument('--prev_task_id', type=int, default=0)
    parser.add_argument('--class_to_replace_2', type=int, default=1)
    parser.add_argument('--class_to_replace_test', type=int, nargs='+', default=[])
    parser.add_argument('--num_indexes_to_replace', type=int, default=None)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--indexes_to_replace', type=int, nargs='+', default=None)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--r', type=int, default=2)
    parser.add_argument('--alpha', type=int, default=4)
    parser.add_argument('--gs_alpha', type=float, default=0.1)
    parser.add_argument('--gs_beta', type=float, default=0.3)
    parser.add_argument('--group_type', type=str, default='block')
    parser.add_argument('--shots', type=int, default=16)
    parser.add_argument('--no_aug', action='store_true')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'],
                       help="list of attention matrices where putting a LoRA")
   
    parser.add_argument('--filename', type=str, default='lora_weights', help="file name to save the lora weights")
    # Add argument to switch using retain set
    parser.add_argument('--use_retain', action='store_true', default=False, help="Whether to use the retain set in training")
    args = parser.parse_args()
    
    # Set derived arguments
    args.train_seed = args.seed
    args.freeze_experts = False
    args.single_router = True
    
    return args

# The rest of the code remains unchanged, as it does not contain identity information.
def setup_coop_dataloaders(args,preprocess=None):

    seed = args.seed
    train_loader_full, val_loader, _,classnames,num_classes,label_to_class= coop_dataloaders(
                batch_size=args.batch_size,
                dataset =args.dataset,
                num_workers=args.workers,
                is_train=True,
                data_dir=args.data_dir,
                preprocess=preprocess)
    print('Loading Marked Loader')

    marked_loader,_,test_loader,_, _,_ = coop_dataloaders(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                dataset =args.dataset,
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
    
    marked_eval_loader,_,_,_, _,_ = coop_dataloaders(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                dataset =args.dataset,
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
        
    forget_dataset = []
    retain_dataset = []
    
    for x, y in tqdm(marked_loader.dataset, desc="Splitting forget and retain", total=len(marked_loader.dataset)):
        if y < 0:
            forget_dataset.append((x, y))
        else:
            retain_dataset.append((x, y))
        
    forget_loader = replace_loader_dataset(
            MarkedDataset(forget_dataset,num_classes),batch_size=args.batch_size, seed=seed, shuffle=True)
    
    forget_eval_dataset = []
    for x, y in tqdm(marked_eval_loader.dataset, desc="Splitting forget for eval", total=len(marked_eval_loader.dataset)):
        if y < 0:
            forget_eval_dataset.append((x, y))
       
    forget_eval_loader = replace_loader_dataset(
            MarkedDataset(forget_eval_dataset,num_classes),batch_size=args.batch_size, seed=seed, shuffle=False)

    retain_eval_loader = replace_loader_dataset(
            MarkedDataset(retain_dataset,num_classes),batch_size=args.batch_size, seed=seed, shuffle=True)

    print(f"number of samples retain dataset {len(retain_dataset)}")
    print(f"number of samples k-shot forget dataset {len(forget_dataset)}")
    print(f"number of samples forget dataset {len(forget_eval_dataset)}")
    print(f"number of samples test dataset {len(test_loader.dataset)}")
    print(f"number of samples train dataset {len(train_loader_full.dataset) }")
    print(f"number of classnames {len(classnames)}")
    unlearn_data_loaders = OrderedDict(
            retain=retain_eval_loader, forget=forget_loader, test=test_loader,
            forget_eval=forget_eval_loader
        )
    return unlearn_data_loaders, classnames

def replace_loader_dataset(dataset, batch_size, seed, shuffle):
    setup_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=shuffle,
    )

def get_configs(args):
    lora_configs = []
    for i in range(args.num_experts):
        alpha = 4
        lora_configs.append({'r': 2, 'lora_alpha': alpha, 'fan_in_fan_out': False, 'dropout_rate': 0.1})
    return lora_configs

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
    'TinyImageNet': 'a photo of a {}.',
    'CUB200': 'a photo of a {}, a type of bird.',
}

def evaluate_lora(args, clip_model, loader, classnames):
    clip_model.eval()
    with torch.no_grad():
        template = CUSTOM_TEMPLATES[args.dataset]
        texts = [template.format(classname.replace('_', ' ')) for classname in classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        text_features = text_features.float()  # Convert to float32 for consistency
    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target) in tqdm(enumerate(loader), total=len(loader), desc='Evaluating'):
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features.float()  # Convert to float32 for consistency
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
    acc /= tot_samples
    return acc

def group_sparse_multi_module(group_param, loss_type='mse'):
    loss_sum = torch.tensor(0.0).cuda()
    for param in group_param:
        if param is not None:
            loss_sum += torch.sum(param ** 2)
    return torch.sqrt(loss_sum)

def get_structure_loss(model, num_layers, group_type, encoder):
    learnable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    group_layers = []
    if group_type == 'block':
        for i in num_layers:
            group_item = []
            if encoder == 'vision':
                group_item.append(f"visual.transformer.resblocks.{i}.mlp.c_proj.w_lora_A")
                group_item.append(f"visual.transformer.resblocks.{i}.mlp.c_proj.w_lora_B")
            elif encoder == 'text':
                group_item.append(f"transformer.resblocks.{i}.mlp.c_proj.w_lora_A")
                group_item.append(f"transformer.resblocks.{i}.mlp.c_proj.w_lora_B")
            group_layers.append(group_item)
    elif group_type == 'lora':
        for i in num_layers:
            group_item = []
            if encoder == 'vision':
                group_item.append(f"visual.transformer.resblocks.{i}.mlp.c_proj.w_lora_A")
                group_item.append(f"visual.transformer.resblocks.{i}.mlp.c_proj.w_lora_B")
            elif encoder == 'text':
                group_item.append(f"transformer.resblocks.{i}.mlp.c_proj.w_lora_A")
                group_item.append(f"transformer.resblocks.{i}.mlp.c_proj.w_lora_B")
            group_layers.append(group_item)
    else:
        raise ValueError("Group type not supported")
    group_params = []
    for group in group_layers:
        group_param = []
        for layer in group:
            group_param.append(model.get_parameter(layer) if layer in learnable_params else None)
        group_params.append(group_param)
    group_sparse_loss = 0
    for group_param in group_params:
        group_sparse_loss += group_sparse_multi_module(group_param)
    return group_sparse_loss

def to_cuda(samples, targets, device):
    samples = samples.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    return samples, targets

class DataPrefetcher:
    def __init__(self, loader, device, prefetch=True):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()
    def preload(self):
        try:
            self.next_samples, self.next_targets = next(self.loader)
        except StopIteration:
            self.next_samples = None
            self.next_targets = None
            return
        with torch.cuda.stream(self.stream):
            self.next_samples, self.next_targets = to_cuda(
                self.next_samples, self.next_targets, self.device
            )
    def next(self):
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            targets = self.next_targets
            if samples is not None:
                samples.record_stream(torch.cuda.current_stream())
            if targets is not None:
                targets.record_stream(torch.cuda.current_stream())
            self.preload()
        else:
            try:
                samples, targets = next(self.loader)
                samples, targets = to_cuda(samples, targets, self.device)
            except StopIteration:
                samples = None
                targets = None
        return samples, targets

def clip_classifier(classnames, template, clip_model):
    clip_model = clip_model.float()
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings = class_embeddings.float()  # Convert to float32 for consistency
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights

def run_lora(args, clip_model, logit_scale, classnames, train_loader, retain_loader, test_loader, lora_configs):
    if args.encoder == 'text' or args.encoder == 'both':
        txt_pos = INDEX_POSITIONS_TEXT[args.position]
    if args.encoder == 'vision' or args.encoder == 'both':
        vis_pos = INDEX_POSITIONS_VISION[args.backbone][args.position]
    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda()
    clip_model = clip_model.float()
    print("LoRA layers: ", list_lora_layers)
    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, classnames=classnames)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return
    print("\nGetting textual features as CLIP's classifier.")
    template = CUSTOM_TEMPLATES[args.dataset]
    textual_features = clip_classifier(classnames, [template], clip_model)
    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters
    params = get_lora_parameters(clip_model)
    print(f'Total parameters:  {sum(p.numel() for p in clip_model.parameters())/1e6}M')
    print([name for name, param in clip_model.named_parameters() if param.requires_grad])
    print(f'Total trainable parameters: {sum(p.numel() for p in filter(lambda p: p.requires_grad, params))/1e6}M')
    optimizer = torch.optim.AdamW(params, weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0
    scaler = torch.amp.GradScaler('cuda')
    count_iters = 0
    finish = False
    while count_iters < total_iters:
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        # Only prefetch retain if using retain set
        if args.use_retain:
            prefetched_retain = DataPrefetcher(retain_loader, 'cuda', prefetch=True)
            images_retain, targets_retain = prefetched_retain.next()
            images_retain, targets_retain = images_retain.cuda(), targets_retain.cuda()
        if args.encoder == 'vision':
            text_features = textual_features.t().float()  # Convert to float32 instead of half
        for i, (images, target) in enumerate(tqdm(train_loader)):
            template = CUSTOM_TEMPLATES[args.dataset]
            texts = [template.format(classname.replace('_', ' ')) for classname in classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                text_features = text_features.float()  # Ensure float32
            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                    image_features = clip_model.encode_image(images)
                    if args.use_retain:
                        image_features_retain = clip_model.encode_image(images_retain)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                        image_features = clip_model.encode_image(images)
                        if args.use_retain:
                            image_features_retain = clip_model.encode_image(images_retain)
            
            # Use out-of-place operations to avoid inplace modification
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            if args.use_retain:
                image_features_retain = image_features_retain / image_features_retain.norm(dim=-1, keepdim=True)
            
            # Ensure logit_scale is float32
            logit_scale_float = logit_scale.float()
            cosine_similarity = logit_scale_float * image_features @ text_features.t()
            loss_forget = F.cross_entropy(cosine_similarity, target)
            loss_forget = torch.functional.F.relu(100 - loss_forget)
            if args.use_retain:
                loss_retain = F.cross_entropy(logit_scale_float * image_features_retain @ text_features.t(), targets_retain)
            else:
                loss_retain = 0.0
            loss_structure_text = 0
            loss_structure_vision = 0
            if args.encoder == 'vision' or args.encoder == 'both':
                loss_structure_vision = get_structure_loss(clip_model, vis_pos, args.group_type, 'vision')
            if args.encoder == 'text' or args.encoder == 'both':
                loss_structure_text = get_structure_loss(clip_model, len(txt_pos), args.group_type, 'text')
            loss_structure = loss_structure_text + loss_structure_vision
            if count_iters < int(0.2 * total_iters):
                loss_structure = 0
            loss_data = args.gs_beta * loss_forget + loss_retain
            loss = loss_data + args.gs_alpha * loss_structure
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            count_iters += 1
            if count_iters == total_iters:
                break
        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(current_lr, acc_train, loss_epoch))
    if args.save_path is not None:
        save_lora(args, list_lora_layers)
    return

def get_model(args):
    try:
        model, preprocess = clip.load(args.clip_path)
        orig_model, _ = clip.load(args.clip_path)
    except Exception:
        model, preprocess = clip.load(args.backbone, device='cuda')
        orig_model, _ = clip.load(args.backbone, device='cuda')
        if not os.path.exists(args.clip_path):
            os.makedirs(f'{args.root_path}/clip_models', exist_ok=True)
            torch.save(model.state_dict(), args.clip_path)
    return model, preprocess, orig_model

def main():
    args = parse_args()
    print(vars(args))
    args.data = args.data_dir
    
    setup_seed(args.seed)
    seed=args.seed
    args.train_seed = args.seed
    
    model, preprocess, orig_model = get_model(args)
    model.eval()
    logit_scale = model.logit_scale

    print("Preparing dataset.")
    if args.dataset in coop_datasets:
        unlearn_data_loaders, classnames = setup_coop_dataloaders(args,preprocess=preprocess)
        args.num_classes = len(classnames)
    else:
        (
            train_loader_full,
            val_loader,
            test_loader,
            marked_loader,
            classnames
        ) = utils.setup_model_dataset(args,preprocess)
        args.num_classes =len(classnames)
        forget_dataset = copy.deepcopy(marked_loader.dataset)
        if args.dataset == "svhn":
            try:
                marked = forget_dataset.targets < 0
            except:
                marked = forget_dataset.labels < 0
            forget_dataset.data = forget_dataset.data[marked]
            try:
                forget_dataset.targets = -forget_dataset.targets[marked] - 1
            except:
                forget_dataset.labels = -forget_dataset.labels[marked] - 1
            forget_loader = replace_loader_dataset(forget_dataset,batch_size=args.batch_size, seed=seed, shuffle=True)
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            try:
                marked = retain_dataset.targets >= 0
            except:
                marked = retain_dataset.labels >= 0
            retain_dataset.data = retain_dataset.data[marked]
            try:
                retain_dataset.targets = retain_dataset.targets[marked]
            except:
                retain_dataset.labels = retain_dataset.labels[marked]
            retain_loader = replace_loader_dataset(retain_dataset,batch_size=args.batch_size, seed=seed, shuffle=True)
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )
        else:
            try:
                marked = forget_dataset.targets < 0
                forget_dataset.data = forget_dataset.data[marked]
                forget_dataset.targets = -forget_dataset.targets[marked] - 1
                forget_loader = replace_loader_dataset(
                    forget_dataset,batch_size=args.batch_size, seed=seed, shuffle=True
                )
                retain_dataset = copy.deepcopy(marked_loader.dataset)
                marked = retain_dataset.targets >= 0
                retain_dataset.data = retain_dataset.data[marked]
                retain_dataset.targets = retain_dataset.targets[marked]
                retain_loader = replace_loader_dataset(
                    retain_dataset,batch_size=args.batch_size, seed=seed, shuffle=True
                )
                assert len(forget_dataset) + len(retain_dataset) == len(
                    train_loader_full.dataset
                )
            except:
                marked = forget_dataset.targets < 0
                forget_dataset.imgs = forget_dataset.imgs[marked]
                forget_dataset.targets = -forget_dataset.targets[marked] - 1
                forget_loader = replace_loader_dataset(
                    forget_dataset,batch_size=args.batch_size, seed=seed, shuffle=True
                )
                retain_dataset = copy.deepcopy(marked_loader.dataset)
                marked = retain_dataset.targets >= 0
                retain_dataset.imgs = retain_dataset.imgs[marked]
                retain_dataset.targets = retain_dataset.targets[marked]
                retain_loader = replace_loader_dataset(
                    retain_dataset,batch_size=args.batch_size,seed=seed, shuffle=True
                )
                assert len(forget_dataset) + len(retain_dataset) == len(
                    train_loader_full.dataset
                )
        from copy import deepcopy
        orig_forget_loader = deepcopy(forget_loader)
        print(f"number of retain dataset {len(retain_loader.dataset)}")
        print(f"number of forget dataset {len(forget_loader.dataset)}")
        import numpy as np
        num_forget_samples = args.shots
        rng = np.random.RandomState(seed)
        # Select indices for forget set
        if hasattr(forget_dataset, 'data'):
            total_indices_forget = np.arange(len(forget_dataset.data))
        elif hasattr(forget_dataset, 'imgs'):
            total_indices_forget = np.arange(len(forget_dataset.imgs))
        else:
            total_indices_forget = np.arange(len(forget_dataset))
        if len(total_indices_forget) < num_forget_samples:
            raise ValueError(f"Forget dataset has fewer than {num_forget_samples} samples.")
        selected_indices_forget = rng.choice(total_indices_forget, size=num_forget_samples, replace=False)
        # Subset the forget_dataset to only the selected indices
        if hasattr(forget_dataset, 'data'):
            forget_dataset.data = forget_dataset.data[selected_indices_forget]
            if hasattr(forget_dataset, 'targets'):
                forget_dataset.targets = forget_dataset.targets[selected_indices_forget]
            elif hasattr(forget_dataset, 'labels'):
                forget_dataset.labels = forget_dataset.labels[selected_indices_forget]
        elif hasattr(forget_dataset, 'imgs'):
            forget_dataset.imgs = [forget_dataset.imgs[i] for i in selected_indices_forget]
            if hasattr(forget_dataset, 'targets'):
                forget_dataset.targets = forget_dataset.targets[selected_indices_forget]
            elif hasattr(forget_dataset, 'labels'):
                forget_dataset.labels = forget_dataset.labels[selected_indices_forget]
        else:
            if hasattr(forget_dataset, 'indices'):
                forget_dataset.indices = [forget_dataset.indices[i] for i in selected_indices_forget]
        forget_train_loader = replace_loader_dataset(
            forget_dataset, batch_size=args.batch_size, seed=seed, shuffle=True
        )

        per_class_num = args.shots
        if hasattr(retain_dataset, 'targets'):
            targets = retain_dataset.targets
            if isinstance(targets, torch.Tensor):
                targets_np = targets.cpu().numpy()
            else:
                targets_np = np.array(targets)
            unique_classes = np.unique(targets_np)
            selected_indices_retain = []
            for cls in unique_classes:
                cls_indices = np.where(targets_np == cls)[0]
                if len(cls_indices) < per_class_num:
                    raise ValueError(f"Class {cls} in retain dataset has fewer than {per_class_num} samples.")
                selected_cls_indices = rng.choice(cls_indices, size=per_class_num, replace=False)
                selected_indices_retain.extend(selected_cls_indices)
            selected_indices_retain = np.array(selected_indices_retain)
        else:
            total_indices_retain = np.arange(len(retain_dataset))
            if len(total_indices_retain) < per_class_num:
                raise ValueError(f"Retain dataset has fewer than {per_class_num} samples.")
            selected_indices_retain = rng.choice(total_indices_retain, size=per_class_num, replace=False)
        if hasattr(retain_dataset, 'data'):
            retain_dataset.data = retain_dataset.data[selected_indices_retain]
            if hasattr(retain_dataset, 'targets'):
                retain_dataset.targets = retain_dataset.targets[selected_indices_retain]
            elif hasattr(retain_dataset, 'labels'):
                retain_dataset.labels = retain_dataset.labels[selected_indices_retain]
        elif hasattr(retain_dataset, 'imgs'):
            retain_dataset.imgs = [retain_dataset.imgs[i] for i in selected_indices_retain]
            if hasattr(retain_dataset, 'targets'):
                retain_dataset.targets = retain_dataset.targets[selected_indices_retain]
            elif hasattr(retain_dataset, 'labels'):
                retain_dataset.labels = retain_dataset.labels[selected_indices_retain]
        else:
            if hasattr(retain_dataset, 'indices'):
                retain_dataset.indices = [retain_dataset.indices[i] for i in selected_indices_retain]
        retain_train_loader = replace_loader_dataset(
            retain_dataset, batch_size=args.batch_size, seed=seed, shuffle=True
        )

        print(f"number of retain dataset (train subset) {len(retain_loader.dataset)}")
        print(f"number of forget dataset {len(forget_dataset)}")
        unlearn_data_loaders = OrderedDict(
            retain=retain_train_loader, 
            forget_eval=orig_forget_loader, forget=forget_train_loader, test=test_loader,
        )

    lora_configs = get_configs(args)
    args.num_tasks = 1
    args.n_cond_vectors = 1
    
    args.eval_only = False
    clip_model = model.to('cuda')
    clip_model = clip_model.float()
    
    os.makedirs(args.save_path, exist_ok=True)
    print(classnames)
    
    if args.use_retain:
        retain_loader_to_use = unlearn_data_loaders['retain']
    else:
        class DummyLoader:
            def __iter__(self): return self
            def __next__(self): raise StopIteration
            def __len__(self): return 0
        retain_loader_to_use = DummyLoader()
    
    run_lora(
        args, clip_model, logit_scale, classnames,
        unlearn_data_loaders['forget'],
        retain_loader_to_use,
        unlearn_data_loaders['test'],
        lora_configs
    )
    
    args.eval_only = True
    acc_fgt = evaluate_lora(args, clip_model, unlearn_data_loaders['forget'], classnames)
    print("**** Final forget accuracy: {:.2f}. ****\n".format(acc_fgt))

    acc_fgt_full = evaluate_lora(args, clip_model, unlearn_data_loaders['forget_eval'], classnames)
    print("**** Final forget accuracy: {:.2f}. ****\n".format(acc_fgt_full))

    acc_test = evaluate_lora(args, clip_model, unlearn_data_loaders['test'], classnames)
    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))

    args.save_dir=args.save_path
    eval_result  = {'test':acc_test,'forget':acc_fgt,'forget_eval':acc_fgt_full}

    n = len(unlearn_data_loaders['forget_eval'].dataset)
    k = len(unlearn_data_loaders['forget'].dataset)

    acc_all_forget_eval = acc_fgt_full  
    acc_16_forget = acc_fgt            

    if n > k:
        acc_rest_forget_eval = (acc_all_forget_eval * n - acc_16_forget * k) / (n - k)
    else:
        acc_rest_forget_eval = float('nan')

    eval_result['forget_eval_acc_rest'] = acc_rest_forget_eval

    import json

    def is_jsonable(x):
        try:
            json.dumps(x)
            return True
        except Exception:
            return False

    args_dict = {}
    for k, v in vars(args).items():
        if is_jsonable(v):
            args_dict[k] = v
    eval_result['args'] = args_dict
    print(len(args_dict))
    utils.save_eval_result(args,eval_result)

if __name__ == "__main__":
    main()    
