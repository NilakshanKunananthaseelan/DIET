
import random
import argparse  
import numpy as np 
import torch

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="VLM Unlearning Experiments")

    ##################################### General Settings ############################################
    parser.add_argument('--seed', default=1, type=int, help="random seed")
    parser.add_argument('--gpu', type=int, default=0, help="gpu device id")
    parser.add_argument('--workers', type=int, default=4, help="number of workers in dataloader")
    parser.add_argument('--print_freq', default=20, type=int, help="print frequency")
    parser.add_argument('--resume', action='store_true', help="resume from checkpoint")
    parser.add_argument('--checkpoint', type=str, default=None, help="checkpoint file")
    parser.add_argument('--save_dir', type=str, default=None, help="directory to save trained models")
    parser.add_argument('--model_path', type=str, default=None, help="path of original model")
    parser.add_argument('--eval_only', default=False, action='store_true', help="only evaluate the model")

    ##################################### Dataset Settings ############################################
    parser.add_argument('--root_path', type=str, default='', help="root path for data")
    parser.add_argument('--data', type=str, default="../data", help="location of the data corpus")
    parser.add_argument('--dataset', type=str, default='dtd', help="dataset name")
    parser.add_argument('--shots', default=16, type=int, help="number of shots for few-shot learning")
    parser.add_argument('--input_size', type=int, default=224, help="size of input images")
    parser.add_argument('--data_dir', type=str, default="./data", help="directory for dataset")
    parser.add_argument('--num_workers', type=int, default=4, help="number of dataloader workers")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--no-aug', action='store_true', default=False, help="no augmentation in training")

    ##################################### Model Architecture ############################################
    parser.add_argument('--backbone', default='ViT-B/16', type=str, help="vision backbone model")
    parser.add_argument('--arch', type=str, default="resnet18", help="model architecture")
    parser.add_argument('--imagenet_arch', action='store_true', help="architecture for imagenet size samples")

    ##################################### Training Settings ############################################
    parser.add_argument('--batch_size', type=int, default=32, help="batch size")
    parser.add_argument('--lr', default=2e-4, type=float, help="initial learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help="momentum")
    parser.add_argument('--weight_decay', default=5e-4, type=float, help="weight decay")
    parser.add_argument('--epochs', default=182, type=int, help="number of total epochs to run")
    parser.add_argument('--n_iters', default=5, type=int, help="number of iterations")
    parser.add_argument('--warmup', default=0, type=int, help="warm up epochs")
    parser.add_argument('--decreasing_lr', default="91,136", help="decreasing strategy")

    ##################################### LoRA Settings ############################################
    parser.add_argument('--position', type=str, default='all', 
                       choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3','top'], 
                       help="where to put the LoRA modules")
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both',
                       help="which encoder to apply LoRA to")
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'],
                       help="list of attention matrices where putting a LoRA")
    parser.add_argument('--r', default=2, type=int, help="rank of the low-rank matrices")
    parser.add_argument('--alpha', default=1, type=int, help="scaling (see LoRA paper)")
    parser.add_argument('--dropout_rate', default=0.1, type=float, help="dropout rate applied before LoRA module")
    parser.add_argument('--save_path', default=None, help="path to save the lora modules after training")
    parser.add_argument('--filename', default='lora_weights', help="file name to save the lora weights")

    ##################################### Pruning Settings ############################################
    parser.add_argument('--prune', type=str, default="omp", help="method to prune")
    parser.add_argument('--pruning_times', default=1, type=int, help="overall times of pruning (only works for IMP)")
    parser.add_argument('--rate', default=0.95, type=float, help="pruning rate")
    parser.add_argument('--prune_type', default="rewind_lt", type=str, help="IMP type (lt, pt or rewind_lt)")
    parser.add_argument('--random_prune', action='store_true', help="whether using random prune")
    parser.add_argument('--rewind_epoch', default=0, type=int, help="rewind checkpoint")
    parser.add_argument('--rewind_pth', default=None, type=str, help="rewind checkpoint to load")
    parser.add_argument('--no-l1-epochs', default=0, type=int, help="non l1 epochs")

    ##################################### Unlearning Settings ############################################
    parser.add_argument('--unlearn', type=str, default="retrain", help="method to unlearn")
    parser.add_argument('--unlearn_lr', default=0.01, type=float, help="initial learning rate for unlearning")
    parser.add_argument('--unlearn_epochs', default=10, type=int, help="number of total epochs for unlearn to run")
    parser.add_argument('--num_indexes_to_replace', type=int, default=None, help="Number of data to forget")
    parser.add_argument('--class_to_replace', type=int, default=-1, help="Specific class to forget")
    parser.add_argument('--indexes_to_replace', type=list, default=None, help="Specific index data to forget")
    parser.add_argument('--alpha_unlearn', default=0.2, type=float, help="unlearn noise")
    parser.add_argument('--mask_path', default=None, type=str, help="path of saliency map")
    parser.add_argument('--norm_r',default=1.5,type=float)
    ##################################### Hyperbolic/OT Settings ############################################
    parser.add_argument('--lambda_hyp', default=10.0, type=float, 
                       help="Hyperbolic loss lambda (weight for hyperbolic loss term)")
    parser.add_argument('--lambda_ot', default=10.0, type=float, 
                       help="Optimal Transport loss lambda")
    parser.add_argument('--lambda_retain', default=10.0, type=float, 
                       help="Retain loss lambda (weight for retain loss term)")
    parser.add_argument('--prototype_type', default="random", type=str, 
                       help="Type of prototype to use (e.g., random, centroid, etc.)")
    parser.add_argument('--cost_type', default="busemann", type=str, 
                       help="Type of cost function to use (e.g., busemann, euclidean, etc.)")
    parser.add_argument('--ot_type', default="sinkhorn", type=str, 
                       help="Type of optimal transport to use (e.g., sinkhorn, emd, etc.)")

    ##################################### Label Files ############################################
    parser.add_argument('--train_y_file', type=str, default="./labels/train_ys.pth", 
                       help="labels for training files")
    parser.add_argument('--val_y_file', type=str, default="./labels/val_ys.pth", 
                       help="labels for validation files")
    

    ##################################### GS LoRA Settings ############################################
    parser.add_argument('--gs_alpha', default=0.1, type=float, help="weight for GS structure loss")
    parser.add_argument('--gs_beta', default=0.3, type=float, help="beta for GS structure loss")
    parser.add_argument('--group_type', default="block", type=str, help="grouping type for GS LoRA (e.g., block, channel, etc.)")
    
    args = parser.parse_args()
    return args
    

        
