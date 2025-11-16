# -*- coding: utf-8 -*-
import torch
import torch.nn as nn 
import torch.nn.functional as F
import time
import math
import numpy as np
from tqdm import tqdm
import clip
import random
import utils
import copy
from clip_utils import clip_classifier, cls_acc, pre_load_features,evaluate_lora
from unlearn_utils import (
    norm_clip,
    busemann_cost_matrix,
    pot_emd,
    pot_sinkhorn,
    _mobius_add,
    get_b_nonuniform,
    transport_plan
)
from utils import (
    setup_logging, 
    log_metrics,
    make_cosine_acc_scheduler,

)
from loralib.utils import (
    mark_only_lora_as_trainable,
    apply_lora,
    get_lora_parameters,
    save_lora,
    load_lora
)


def get_forget_class_similarity(image_features, text_features, target_labels, class_to_replace):
    cosine_similarity = image_features @ text_features.t()  # [batch_size, num_classes]
    predicted_class = cosine_similarity.argmax(dim=1)  # [batch_size]
    predicted_similarity = cosine_similarity[torch.arange(len(target_labels)), predicted_class]
    actual_forget_similarity = cosine_similarity[torch.arange(len(target_labels)), target_labels]
    return predicted_similarity, actual_forget_similarity, predicted_class

def update_scheduler_with_val_acc(args, clip_model, val_loader, template, classnames, on_epoch_end, epoch_idx: int):
    val_acc = evaluate_lora(args, clip_model, val_loader, template, classnames)
    on_epoch_end(epoch_idx, float(val_acc))
    return float(val_acc)

# def evaluate_lora(args, clip_model, loader, template, classnames):
#     """
#     Evaluates the LoRA-adapted CLIP model on a given data loader.
#     Args:
#         args: Argument namespace
#         clip_model: CLIP model
#         loader: DataLoader for evaluation
#         template: Text template for classnames
#         classnames: List of class names
#     Returns:
#         Classification accuracy
#     """
#     clip_model.eval()
#     with torch.no_grad():
#         # Prepare text features
#         texts = [template.format(classname.replace('_', ' ')) for classname in classnames]
#         with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
#             texts = clip.tokenize(texts).cuda()
#             class_embeddings = clip_model.encode_text(texts)
#         text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

#         acc = 0.0
#         tot_samples = 0
#         for images, target in loader:
#             images, target = images.cuda(), target.cuda()
#             with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
#                 image_features = clip_model.encode_image(images)
#             image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#             cosine_similarity = image_features @ text_features.t()
#             acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
#             tot_samples += len(cosine_similarity)
#         acc /= tot_samples
#     return acc

def compute_retain_regularizer(C_retain, Pi, margin=1.0):
    """
    Computes a regularization loss to encourage retention of non-selected prototypes.
    Args:
        C_retain: Cost matrix between features and prototypes
        Pi: Optimal transport plan (soft assignment)
        margin: Margin for the regularizer
    Returns:
        Regularization loss (scalar)
    """
    assigned_k = Pi.argmax(dim=1)  # Hard assignment for each sample
    N, K = C_retain.shape
    mask = torch.ones_like(C_retain, dtype=torch.bool)
    mask[torch.arange(N), assigned_k] = False
    C_nonassigned = C_retain[mask].view(N, K - 1)
    reg_loss = torch.clamp(margin - C_nonassigned, min=0).mean()
    return reg_loss

def run_lora(
    args,
    clip_model,
    logit_scale,
    classnames,
    template,
    train_loader,
    val_loader,
    test_loader,
    loss_fn=None,
    proto_h=None
):
    """
    Main training loop for LoRA-based semantic unlearning with hyperbolic geometry.
    Args:
        args: Argument namespace
        clip_model: CLIP model
        logit_scale: Scaling factor for logits
        classnames: List of class names
        template: Text template for classnames
        train_loader: DataLoader for training/forgetting set
        test_loader: DataLoader for test set
        loss_fn: Optional custom loss function
        proto_h: Prototypes in hyperbolic space
    Returns:
        Trained CLIP model
    """
    log_file = setup_logging(args)
    args.log_file = log_file
    forget_loader = train_loader

    # Hyperparameters
    lambda_hyp = args.lambda_hyp
    lambda_ot = args.lambda_ot
    lambda_retain = args.lambda_retain
    norm_r = args.norm_r
    manifold = args.manifold
    c = manifold.c.item()

    # Compute textual features for classifier
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(classnames, [template], clip_model)

    # Evaluate zero-shot accuracy on test set
    test_features, test_labels = pre_load_features(clip_model, test_loader)
    test_features = test_features.cuda()
    test_labels = test_labels.cuda()
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc_t = cls_acc(clip_logits, test_labels)
    print(f"\n**** Zero-shot CLIP's test accuracy: {zs_acc_t:.2f}. ****\n")
    test_features = test_features.cpu()
    test_labels = test_labels.cpu()

    # Evaluate zero-shot accuracy on forget set
    forget_features, forget_labels = pre_load_features(clip_model, train_loader)
    forget_features = forget_features.cuda()
    forget_labels = forget_labels.cuda()
    clip_logits = logit_scale * forget_features @ textual_features
    zs_acc_f = cls_acc(clip_logits, forget_labels)
    print(f"\n**** Zero-shot CLIP's forget accuracy: {zs_acc_f:.2f}. ****\n")

    # Create a deepcopy of the original model
    
    ref_clip_model = copy.deepcopy(clip_model)
    

    # Apply LoRA adaptation to the model
    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda()
    clip_model = clip_model.float()

    # If only evaluation is required, load LoRA weights and evaluate
    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, template, classnames)
        print(f"**** Test accuracy: {acc_test:.2f}. ****\n")
        return clip_model

    # Set LoRA parameters as trainable
    mark_only_lora_as_trainable(clip_model)
    trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    print(f"Trainable Params: {trainable_params/1e6:.2f}M")
    total_iters = args.unlearn_epochs * len(train_loader)
    print(f'Total Iters: {total_iters}')

    # Optimizer and scheduler setup
    optimizer = torch.optim.AdamW(
        get_lora_parameters(clip_model),
        weight_decay=1e-5,
        betas=(0.9, 0.999),
        lr=args.unlearn_lr
    )
    scheduler, on_epoch_end = make_cosine_acc_scheduler(
        optimizer,
        total_iters=total_iters,
        eta_min=1e-5,
        acc_threshold=2.0,
        drop_factor=0.1,
        cooldown_epochs=2
    )

    # Select cost and OT functions
    cost_type = getattr(args, "cost_type", "busemann")
    if cost_type == "busemann":
        cost_fn = busemann_cost_matrix
    else:
        raise ValueError(f"Unknown cost_type: {cost_type}")

    ot_type = getattr(args, "ot_type", "sinkhorn")
    if ot_type == "sinkhorn":
        ot_fn = pot_sinkhorn
        ot_kwargs = dict(
            eps=getattr(args, "sinkhorn_epsilon", 0.5),
            max_iter=getattr(args, "sinkhorn_max_iter", 2000)
        )
    elif ot_type == "emd":
        ot_fn = pot_emd
        ot_kwargs = {}
    else:
        raise ValueError(f"Unknown ot_type: {ot_type}")

    print('Start Training....')

    loader_len = len(forget_loader)
    all_proto_idx_per_epoch = []
    for epoch in range(args.unlearn_epochs):
        # Initialize meters for tracking loss and accuracy
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        hyp_losses = utils.AverageMeter()

        feats_h_norms = []
        epoch_proto_idx = []


        start = time.time()
        clip_model.train()
        clip_model.float()
        acc_train = 0.0
        tot_samples = 0
        loss_epoch = 0.0
        hyp_loss_epoch = 0.0
        feats_avg_norm = 0.0

        for i, (images, target) in enumerate(tqdm(train_loader)):
            # Prepare text and image features
            texts = [template.format(classname.replace('_', ' ')) for classname in classnames]
            images, target = images.cuda(), target.cuda()

            # Text encoder branch
            if args.encoder in ['text', 'both']:
                texts_tok = clip.tokenize(texts).cuda()
                class_embeddings = clip_model.encode_text(texts_tok)
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            # Vision encoder branch
            if args.encoder in ['vision', 'both']:
                texts_feat = clip.tokenize(texts).cuda()
                class_embeddings = clip_model.encode_text(texts_feat).detach()
                text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                image_features = clip_model.encode_image(images)
                
                with torch.no_grad():
                    ref_image_features = ref_clip_model.encode_image(images)
                    ref_feats = norm_clip(ref_image_features,norm_r)
                    ref_feats_h = manifold.expmap0(ref_feats)


            # Create ideal prototypes from text
            proto_h = proto_h.float()
            proto_dir = proto_h / proto_h.norm(dim=-1, keepdim=True)
            radius = 1.0 / math.sqrt(c)
            proto_boundary = proto_dir * (radius - 1e-6)
            proto_boundary = proto_boundary.float()
            

            # Project features to hyperbolic space
            feats = norm_clip(image_features, norm_r*1.25)
            feats = image_features
            feats_h = manifold.expmap0(feats)
            feats_h = feats_h.float()

            feats_h_norm = feats_h.norm(dim=-1)
            feats_h_norms.append(feats_h_norm.detach().cpu())

            # Compute OT plan between features and prototypes
            C = cost_fn(feats_h, proto_boundary, c=c)


            if args.assignment == 'closest':
                proto_idx = C.argmin(dim=1)
                print(proto_idx)
                selected_prototypes = proto_boundary[proto_idx]
                OT = torch.zeros_like(C)
                OT[torch.arange(C.size(0)), proto_idx] = 1.0
                OT_loss = torch.tensor(0.0, device=feats.device, dtype=feats_h.dtype)
            elif args.assignment == 'OT':
                B, K = C.shape
                a = torch.full((B,), 1.0 / B, device=feats.device, dtype=feats_h.dtype)
                b_fixed = getattr(args, "proto_capacity", None)
                b_temp = getattr(args, "proto_capacity_tau", 0.2)
                b = get_b_nonuniform(
                    C, mode=args.OT_mode, fixed=b_fixed, temperature=b_temp,
                    device=feats.device, dtype=feats_h.dtype
                )
                OT, OT_loss = transport_plan(C, a, b, ot_fn=ot_fn, ot_kwargs=ot_kwargs)
                selection = 'argmax'
                if selection=='argmax':
                    proto_idx = OT.argmax(dim=1)
                    print('>>>>>',proto_idx)
                    selected_prototypes = proto_boundary[proto_idx]
                    proto_cost = C[torch.arange(C.size(0)), proto_idx].detach().cpu()
                    epoch_proto_idx.append([{
                        "proto_idx": proto_idx.detach().cpu(),
                        "proto_cost": proto_cost.detach().cpu()
                    }])
                elif selection=='topk':
                    # Select the top 3 prototypes for each sample
                    topk = 5
                    print('using top',topk)
                    OT_topk_weights, proto_idx_topk = OT.topk(topk, dim=1)
                    
                    selected_prototypes_topk = proto_boundary[proto_idx_topk]  # [B, 3, D]
                    # Compute proto_cost for the top-k selected prototypes for each sample
                    proto_cost_topk = C.gather(1, proto_idx_topk).detach().cpu()  # [B, 3]
                    epoch_proto_idx.append([{
                        "proto_idx_topk": proto_idx_topk.detach().cpu(),
                        "proto_cost_topk": proto_cost_topk
                    }])

                    

                    def mobius_weighted_sum(weights, points, c):
                        # weights: [B, k], points: [B, k, D]
                        # Returns: [B, D]
                        B, k, D = points.shape
                        result = points[:, 0, :]
                        for i in range(1, k):
                            w = weights[:, i].unsqueeze(-1)
                            result = _mobius_add(result, w * points[:, i, :], c=c)
                        return result

                    # Normalize weights to sum to 1 for each sample
                    OT_topk_weights = OT_topk_weights / (OT_topk_weights.sum(dim=1, keepdim=True) + 1e-8)
                    selected_prototypes = mobius_weighted_sum(OT_topk_weights,selected_prototypes_topk,c=c)


            


            # Compute hyperbolic loss
            if loss_fn is None:
                hyp_loss, _, _ = busemann_loss_euclid(selected_prototypes, feats_h, c=c)
                total_loss = lambda_hyp * hyp_loss + lambda_ot * OT_losss
            else:
                
                if args.loss_type=='regular':
                    
                    hyp_loss = torch.mean(loss_fn(feats_h, selected_prototypes))
                    total_loss = lambda_hyp * hyp_loss + lambda_ot * OT_loss
                    

                elif args.loss_type=='npo':
                    current_hyp_loss = -loss_fn(feats_h, selected_prototypes)
                    with torch.no_grad():
                        ref_hyp_loss = -loss_fn(ref_feats_h,selected_prototypes)

                    neg_log_ratios = current_hyp_loss - ref_hyp_loss
                    hyp_loss = -F.logsigmoid(args.npo_beta * neg_log_ratios).mean() * 2 / args.npo_beta 

                    total_loss = 1 * hyp_loss + lambda_ot * OT_loss
                    

            # Retain regularizer (if not using random prototypes)
            # if getattr(args, "proto_type", None) != 'random':
                # C_retain = cost_fn(feats_h, proto_h, c=c)
                # retain_regularizer_loss = compute_retain_regularizer(C_retain, OT)
                # total_loss += lambda_retain * retain_regularizer_loss
                # cosine_similarity = logit_scale * image_features @ text_features.t()

            if getattr(args, "proto_type", None) != 'random':
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                cosine_similarity = logit_scale * image_features @ text_features.t()
                image_sim = image_features @ image_features.t()
               
                
                # Add a loss if similarity decreases below a threshold
                similarity_threshold = getattr(args, "similarity_threshold", 0.85)
                sim_loss_weight = lambda_retain
                sim_loss = sim_loss_weight * (similarity_threshold - image_sim).mean()
                total_loss += sim_loss

            # Update accuracy and loss statistics
            pred_sim, actual_forget_sim, pred_class = get_forget_class_similarity(
                image_features, text_features, target, args.class_to_replace
            )

            loss_epoch += total_loss.item() * target.shape[0]
            hyp_loss_epoch += hyp_loss.item() * target.shape[0]
            tot_samples += target.shape[0]
            feats_avg_norm += feats.norm(dim=-1).mean().item()

            # Backpropagation and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            # Update meters
            losses.update(torch.tensor(loss_epoch / tot_samples), image_features.size(0))
            top1.update(torch.tensor(acc_train / tot_samples), image_features.size(0))
            hyp_losses.update(hyp_loss_epoch / tot_samples, image_features.size(0))

            # Log the mean and std of the norm of image embeddings in hyperbolic space for this batch
            print()
            print(f"Batch {i+1} - Hyperbolic norm mean: {feats_h_norm.mean().item():.6f}, std: {feats_h_norm.std().item():.6f}")
            # Print average norm of selected_prototypes and proto_h
            if 'selected_prototypes' in locals():
                selected_proto_norm = selected_prototypes.norm(dim=-1).mean().item()
                print(f"Batch {i+1} - Selected prototypes norm mean: {selected_proto_norm:.6f}")
            if 'proto_h' in locals() and proto_h is not None:
                proto_h_norm = proto_h.norm(dim=-1).mean().item()
                print(f"Batch {i+1} - proto_h norm mean: {proto_h_norm:.6f}")
            epoch_metrics = {
                "forget_loss": loss_epoch / tot_samples,
                "forget_accuracy": acc_train / tot_samples,
                "hyp_loss": hyp_losses.avg,
                "lambda_hyp": lambda_hyp,
                "lambda_ot": lambda_ot,
                "cost_type": cost_type,
                "ot_type": ot_type,
                "time": round(time.time() - start, 2),
                "feats_h_norm_mean": feats_h_norm.mean().item(),
                "feats_h_norm_std": feats_h_norm.std().item(),
               
            }
            log_metrics(args.log_file, epoch, epoch_metrics)
        
        all_proto_idx_per_epoch.append(epoch_proto_idx)

        epoch_acc = acc_train / tot_samples
        # on_epoch_end(epoch, epoch_acc)
        
        val_acc = update_scheduler_with_val_acc(args, clip_model, val_loader, template, classnames, on_epoch_end, epoch)


        # Print progress every 2 epochs
        if epoch % 2 == 0 and epoch > 0:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()
            end = time.time()
            print(
                f"Epoch: [{epoch}][{i+1}/{loader_len}]\t"
                f"Loss {loss_epoch:.4f}\t"
                f"Accuracy {epoch_acc:.3f}\t"
                f"Val Acc {val_acc:.3f}\t"
                f"OT Loss {OT_loss.item():.4f}\t"
                f"Retain Loss{sim_loss:.4f}\t"
                f"Hyp Loss {hyp_loss.item():.4f}\n"
                
                f"LR {current_lr}\t"
                f"Lambda Hyp {lambda_hyp:.4f}\t"
                f"Lambda OT {lambda_ot:.4f}\t"
                f"Norm {feats_avg_norm:.4f}\t"
                f"Time {end - start:.2f}"
            )
            print()
            start = time.time()
    # Save LoRA weights if a save path is provided
    if args.save_path is not None:
        save_lora(args, list_lora_layers)
        np.save(f"{args.save_path}/proto_idx_per_epoch.npy", np.array(all_proto_idx_per_epoch, dtype=object))

    return clip_model
