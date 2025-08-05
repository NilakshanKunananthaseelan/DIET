# -*- coding: utf-8 -*-
import torch
import time
import math
from tqdm import tqdm
import clip
import utils
from clip_utils import clip_classifier, cls_acc, pre_load_features
from unlearn_utils import (
    norm_clip,
    busemann_cost_matrix,
    pot_emd,
    pot_sinkhorn
)
from utils import setup_logging, log_metrics
from loralib.utils import (
    mark_only_lora_as_trainable,
    apply_lora,
    get_lora_parameters,
    save_lora,
    load_lora
)

def busemann_loss_euclid(eucl_p, eucl_z, c=1.0, eps=1e-6):
    """
    Computes the Busemann loss in Euclidean coordinates.
    Args:
        eucl_p: Prototype points (batch, dim)
        eucl_z: Data points (batch, dim)
        c: Curvature parameter
        eps: Small value for numerical stability
    Returns:
        Mean Busemann loss, mean squared distance, mean denominator
    """
    squared_euclidean_dist = torch.sum((eucl_p - eucl_z).pow(2), dim=-1)
    poincare_denominator = (1.0 / c) - eucl_z.norm(dim=-1, p=2).pow(2)
    log_numerator = torch.log(squared_euclidean_dist + eps)
    log_denominator = torch.log(poincare_denominator + eps)
    bval = log_numerator - log_denominator
    return bval.mean(), squared_euclidean_dist.mean().item(), poincare_denominator.mean().item()

def evaluate_lora(args, clip_model, loader, template, classnames):
    """
    Evaluates the LoRA-adapted CLIP model on a given data loader.
    Args:
        args: Argument namespace
        clip_model: CLIP model
        loader: DataLoader for evaluation
        template: Text template for classnames
        classnames: List of class names
    Returns:
        Classification accuracy
    """
    clip_model.eval()
    with torch.no_grad():
        # Prepare text features
        texts = [template.format(classname.replace('_', ' ')) for classname in classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

        acc = 0.0
        tot_samples = 0
        for images, target in loader:
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
        acc /= tot_samples
    return acc

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

    # Apply LoRA adaptation to the model
    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda()
    clip_model = clip_model.float()

    # If only evaluation is required, load LoRA weights and evaluate
    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, template, classnames)
        print(f"**** Test accuracy: {acc_test:.2f}. ****\n")
        return

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=2e-5)

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
            max_iter=getattr(args, "sinkhorn_max_iter", 1000)
        )
    elif ot_type == "emd":
        ot_fn = pot_emd
        ot_kwargs = {}
    else:
        raise ValueError(f"Unknown ot_type: {ot_type}")

    loader_len = len(forget_loader)
    for epoch in range(args.unlearn_epochs):
        # Initialize meters for tracking loss and accuracy
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        hyp_losses = utils.AverageMeter()

        start = time.time()
        clip_model.train()
        clip_model.float()
        acc_train = 0.0
        tot_samples = 0
        loss_epoch = 0.0
        hyp_loss_epoch = 0.0

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
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Project features to hyperbolic space
            feats = norm_clip(image_features, norm_r)
            feats_h = manifold.expmap0(feats)
            proto_dir = proto_h / proto_h.norm(dim=-1, keepdim=True)
            radius = 1.0 / math.sqrt(c)
            proto_boundary = proto_dir * (radius - 1e-6)

            # Ensure consistent dtype
            feats_h = feats_h.float()
            proto_boundary = proto_boundary.float()
            proto_h = proto_h.float()

            # Compute OT plan between features and prototypes
            num_prototypes = proto_boundary.shape[0]
            prototype_weights = torch.full((num_prototypes,), 1.0 / num_prototypes, device=feats.device, dtype=feats_h.dtype)
            C = cost_fn(feats_h, proto_boundary, c=c)
            a = torch.full((feats_h.size(0),), 1.0 / feats_h.size(0), device=feats.device, dtype=feats_h.dtype)
            b = prototype_weights
            OT = ot_fn(a, b, C, **ot_kwargs)
            OT_loss = (OT * C).sum()

            # Select prototypes for each sample based on OT plan
            proto_idx = OT.argmax(dim=1)
            selected_prototypes = proto_boundary[proto_idx]

            # Compute hyperbolic loss
            if loss_fn is None:
                hyp_loss, _, _ = busemann_loss_euclid(selected_prototypes, feats_h, c=c)
            else:
                hyp_loss = loss_fn(feats_h, selected_prototypes)
            total_loss = lambda_hyp * hyp_loss + lambda_ot * OT_loss

            # Retain regularizer (if not using random prototypes)
            if getattr(args, "proto_type", None) != 'random':
                C_retain = cost_fn(feats_h, proto_h, c=c)
                retain_regularizer_loss = compute_retain_regularizer(C_retain, OT)
                total_loss += lambda_retain * retain_regularizer_loss
                cosine_similarity = logit_scale * image_features @ text_features.t()

            # Update accuracy and loss statistics
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            if epoch > 0 and acc_train < 10:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] / 10

            loss_epoch += total_loss.item() * target.shape[0]
            hyp_loss_epoch += hyp_loss.item() * target.shape[0]
            tot_samples += target.shape[0]

            # Backpropagation and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            # Update meters
            losses.update(torch.tensor(loss_epoch / tot_samples), image_features.size(0))
            top1.update(torch.tensor(acc_train / tot_samples), image_features.size(0))
            hyp_losses.update(hyp_loss_epoch / tot_samples, image_features.size(0))

            # Log metrics for this epoch
            epoch_metrics = {
                "forget_loss": loss_epoch / tot_samples,
                "forget_accuracy": acc_train / tot_samples,
                "hyp_loss": hyp_losses.avg,
                "lambda_hyp": lambda_hyp,
                "lambda_ot": lambda_ot,
                "cost_type": cost_type,
                "ot_type": ot_type,
                "time": round(time.time() - start, 2)
            }
            log_metrics(args.log_file, epoch, epoch_metrics)

        # Print progress every 2 epochs
        if epoch % 2 == 0 and epoch > 0:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()
            end = time.time()
            print(
                f"Epoch: [{epoch}][{i+1}/{loader_len}]\t"
                f"Loss {loss_epoch:.4f}\t"
                f"Accuracy {acc_train:.3f}\t"
                f"OT Loss {OT_loss.item():.4f}\t"
                f"Hyp Loss {hyp_loss.item()} (log: {torch.log(hyp_loss).item()})\n"
                f"LR {current_lr}\t"
                f"Lambda Hyp {lambda_hyp:.4f}\t"
                f"Lambda OT {lambda_ot:.4f}\t"
                f"Time {end - start:.2f}"
            )
            start = time.time()

    # Save LoRA weights if a save path is provided
    if args.save_path is not None:
        save_lora(args, list_lora_layers)
    return clip_model
