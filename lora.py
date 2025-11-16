import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from clip_utils import *

from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from loralib import layers as lora_layers

def analyze_class_similarity_changes(args, clip_model, classnames, template, before_embeddings=None):
    """
    Analyze how class embeddings change after unlearning to show catastrophic forgetting.
    Returns similarity matrices and change analysis.
    """
    clip_model.eval()

    # Get current class embeddings
    with torch.no_grad():
        texts = [template.format(classname.replace('_', ' ')) for classname in classnames]
        texts = clip.tokenize(texts).cuda()
        current_embeddings = clip_model.encode_text(texts)
        current_embeddings = current_embeddings / current_embeddings.norm(dim=-1, keepdim=True)

    # Compute current similarity matrix
    current_sim_matrix = cosine_similarity(current_embeddings.cpu().numpy())

    if before_embeddings is not None:
        # Compute similarity change
        before_sim_matrix = cosine_similarity(before_embeddings.cpu().numpy())
        similarity_change = current_sim_matrix #- before_sim_matrix

        # Find most affected classes (excluding the forgotten class)
        forgotten_class_idx = args.class_to_replace
        affected_classes = []

        for i in range(len(classnames)):
            if i != forgotten_class_idx:
                # Calculate average similarity change with other classes
                avg_change = np.mean(similarity_change[i, :])
                affected_classes.append((i, avg_change, classnames[i]))

        # Sort by impact (most negative change = most affected)
        affected_classes.sort(key=lambda x: x[1])

        return {
            'current_similarity': current_sim_matrix,
            'similarity_change': similarity_change,
            'most_affected': affected_classes[:10],  # Top 10 most affected
            'least_affected': affected_classes[-10:],  # Top 10 least affected
            'current_embeddings': current_embeddings
        }

    return {
        'current_similarity': current_sim_matrix,
        'current_embeddings': current_embeddings
    }

def visualize_similarity_changes(similarity_data, classnames, args, save_path=None):
    """
    Create visualizations to show the catastrophic forgetting effect.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Current similarity matrix heatmap
    sns.heatmap(similarity_data['current_similarity'],
                xticklabels=classnames, yticklabels=classnames,
                cmap='viridis', ax=axes[0, 0])
    axes[0, 0].set_title('Current Class Similarity Matrix')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].tick_params(axis='y', rotation=0)

    if 'similarity_change' in similarity_data:
        # 2. Similarity change heatmap
        sns.heatmap(similarity_data['similarity_change'],
                    xticklabels=classnames, yticklabels=classnames,
                    cmap='RdBu_r', center=0, ax=axes[0, 1])
        axes[0, 1].set_title('Similarity Change After Unlearning')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].tick_params(axis='y', rotation=0)

        # 3. Most affected classes bar plot
        most_affected = similarity_data['most_affected']
        most_classes = [item[2] for item in most_affected]
        most_changes = [item[1] for item in most_affected]

        axes[1, 0].barh(range(len(most_classes)), most_changes)
        axes[1, 0].set_yticks(range(len(most_classes)))
        axes[1, 0].set_yticklabels(most_classes)
        axes[1, 0].set_xlabel('Average Similarity Change')
        axes[1, 0].set_title('Most Affected Classes (Catastrophic Forgetting)')

        # 4. Least affected classes bar plot
        least_affected = similarity_data['least_affected']
        least_classes = [item[2] for item in least_affected]
        least_changes = [item[1] for item in least_affected]

        axes[1, 1].barh(range(len(least_classes)), least_changes)
        axes[1, 1].set_yticks(range(len(least_classes)))
        axes[1, 1].set_yticklabels(least_classes)
        axes[1, 1].set_xlabel('Average Similarity Change')
        axes[1, 1].set_title('Least Affected Classes')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

def demonstrate_catastrophic_forgetting(args, clip_model, classnames, template, test_loader):
    """
    Demonstrate catastrophic forgetting by showing accuracy changes for different class groups.
    """
    # Group classes by similarity to the forgotten class
    forgotten_class = classnames[args.class_to_replace]
    print(forgotten_class)

    # For demonstration, let's assume we can categorize classes
    # In practice, you'd use semantic similarity or domain knowledge
    similar_classes = []
    dissimilar_classes = []

    # Example for OxfordPets dataset
    if args.dataset == 'OxfordPets':
        # Lowercase classnames for matching
        classnames_lower = [c.lower() for c in classnames]
        dog_breed_names = [
            'american_pit_bull_terrier',
             "American_Bulldog",  "Basset_Hound",
            "Beagle", "Boxer"
            
           
        ]

        dog_breed_names = [i.lower() for i in dog_breed_names]
        cat_breed_names = [
            "Bengal", "Birman", "Bombay"
        ]
        cat_breed_names = [i.lower() for i in cat_breed_names]

        # Map breed names to indices in classnames
        dog_breeds = [i for i, name in enumerate(classnames_lower) if name in dog_breed_names]
        cat_breeds = [i for i, name in enumerate(classnames_lower) if name in cat_breed_names]
        print(forgotten_class in dog_breed_names,dog_breed_names)
        # Determine if forgotten class is a dog or cat
        if forgotten_class.lower() in dog_breed_names:
            similar_classes = dog_breeds
            dissimilar_classes = cat_breeds
        
            similar_classes = []
            dissimilar_classes = []

    # Evaluate accuracy on different class groups
    results = {}

    # Overall accuracy
    overall_acc = evaluate_lora(args, clip_model, test_loader, classnames, template)
    results['overall'] = overall_acc

    # Similar classes accuracy
    if similar_classes:
        similar_acc = evaluate_class_group(args, clip_model, test_loader, similar_classes, classnames, template)
        results['similar_classes'] = similar_acc

    # Dissimilar classes accuracy
    if dissimilar_classes:
        dissimilar_acc = evaluate_class_group(args, clip_model, test_loader, dissimilar_classes, classnames, template)
        results['dissimilar_classes'] = dissimilar_acc

    return results

def evaluate_class_group(args, clip_model, loader, class_indices, classnames, template):
    """
    Evaluate accuracy on a specific group of classes.
    """
    clip_model.eval()

    # Create class-specific text features
    with torch.no_grad():
        texts = [template.format(classnames[i].replace('_', ' ')) for i in class_indices]
        texts = clip.tokenize(texts).cuda()
        class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.0
    tot_samples = 0

    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            # Filter for target classes only
            mask = torch.isin(target, torch.tensor(class_indices, device=target.device))
            if not mask.any():
                continue

            images = images[mask].cuda()
            target = target[mask].cuda()

            # Map target indices to local indices
            # If target is batched, this is safe
            target_mapped = torch.tensor([class_indices.index(t.item()) for t in target], device=target.device)

            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            cosine_sim = image_features @ text_features.t()
            acc += cls_acc(cosine_sim, target_mapped) * len(cosine_sim)
            tot_samples += len(cosine_sim)

    return acc / tot_samples if tot_samples > 0 else 0.0

def evaluate_lora(args, clip_model, loader, classnames, template):
    clip_model.eval()
    with torch.no_grad():
        texts = [template.format(classname.replace('_', ' ')) for classname in classnames]
        texts = clip.tokenize(texts).cuda()
        class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
    acc /= tot_samples

    return acc


def run_lora(args, 
            clip_model,
            logit_scale, 
            classnames,
            template, 
            train_loader, 
            # val_loader, 
            test_loader,
            loss_fn=None,
            proto_h=None
            ):
    VALIDATION = False
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(classnames, [template], clip_model)

    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda() 
    
    if args.eval_only:
        load_lora(args, list_lora_layers)
        print(
            'eval only'
        )
        # acc_test = evaluate_lora(args, clip_model, test_loader, template,classnames)
        # print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return clip_model

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots
    
    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    
    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0
    
    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    for epoch in range(args.unlearn_epochs):
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision': 
            text_features = textual_features.t()
        for i, (images, target) in enumerate(tqdm(train_loader)):
            texts = [template.format(classname.replace('_', ' ')) for classname in classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                texts = clip.tokenize(texts).cuda()
                class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
                
            if args.encoder == 'vision' or args.encoder == 'both':
                image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            
            cosine_similarity = logit_scale * image_features @ text_features.t()
            loss = -F.cross_entropy(cosine_similarity, target)
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            count_iters += 1
            
            # if count_iters == total_iters:
            #     break
            
        # if count_iters < total_iters:
        acc_train /= tot_samples
        loss_epoch /= tot_samples
        
        print('Acc: {:.4f}, Loss: {:.4f}'.format( acc_train, loss_epoch))

    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return clip_model
