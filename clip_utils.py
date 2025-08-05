from tqdm import tqdm
import torch
import clip

def cls_acc(output, target, topk=1):
    """
    Compute classification accuracy for top-k predictions.
    Args:
        output (Tensor): Model output logits or similarity scores.
        target (Tensor): Ground truth labels.
        topk (int): Number of top predictions to consider.
    Returns:
        float: Accuracy in percentage.
    """
    # Get top-k predictions
    pred = output.topk(topk, 1, True, True)[1].t()
    # Check if predictions are correct
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # Count correct predictions
    acc = float(correct[:topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

def clip_classifier(classnames, template, clip_model):
    """
    Build a classifier weight matrix from classnames and a template using a CLIP model.
    Args:
        classnames (list): List of class names.
        template (list): List of string templates for prompt engineering.
        clip_model: CLIP model instance.
    Returns:
        Tensor: Stacked classifier weights for each class.
    """
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Prepare prompt for each class
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # Encode text prompts
            class_embeddings = clip_model.encode_text(texts)
            # Normalize embeddings
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            # Average embeddings for the class
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding = class_embedding / class_embedding.norm()
            clip_weights.append(class_embedding)
        # Stack all class weights into a matrix
        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights

def pre_load_features(clip_model, loader):
    """
    Precompute and cache features and labels for a dataset using a CLIP model.
    Args:
        clip_model: CLIP model instance.
        loader (DataLoader): DataLoader for the dataset.
    Returns:
        (Tensor, Tensor): Features and labels tensors.
    """
    features, labels = [], []
    with torch.no_grad():
        for images, target in tqdm(loader):
            images, target = images.cuda(), target.cuda()
            # Encode images and normalize features
            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            features.append(image_features.cpu())
            labels.append(target.cpu())
        features = torch.cat(features)
        labels = torch.cat(labels)
    return features, labels

def evaluate_lora(args, clip_model, loader, template, classnames):
    """
    Evaluate a CLIP model (possibly with LoRA) on a dataset.
    Args:
        args: Arguments/configuration.
        clip_model: CLIP model instance.
        loader (DataLoader): DataLoader for evaluation.
        template (str): Prompt template string.
        classnames (list): List of class names.
    Returns:
        float: Accuracy in percentage.
    """
    clip_model.eval()
    with torch.no_grad():
        # Prepare text features for all classes
        texts = [template.format(classname.replace('_', ' ')) for classname in classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.0
    tot_samples = 0
    with torch.no_grad():
        for images, target in loader:
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # Compute cosine similarity between image and text features
            cosine_similarity = image_features @ text_features.t()
            # Accumulate accuracy
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
    acc /= tot_samples
    return acc