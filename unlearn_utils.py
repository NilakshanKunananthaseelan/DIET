
import numpy as np
import torch
import math
import ot
from tqdm import tqdm

# ---------------------- VECTOR NORMALIZATION ----------------------
def norm_clip(x, r):
    """
    Clips the norm of each vector in x to be at most r.
    Args:
        x: Tensor of shape (..., d)
        r: Maximum allowed norm (float)
    Returns:
        Tensor with same shape as x, with norms clipped to r.
    """
    norm = torch.norm(x, dim=-1)
    scale = torch.clamp(r / norm, max=1.0)
    return scale.unsqueeze(-1) * x

# ---------------------- OPTIMAL TRANSPORT UTILS -------------------
def safe_log(x, eps=1e-8):
    """
    Numerically stable log for tensors or scalars.
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    return torch.log(torch.clamp(x, min=eps))

def safe_sqrt(x, eps=1e-8):
    """
    Numerically stable sqrt for tensors or scalars.
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    return torch.sqrt(torch.clamp(x, min=eps))

def busemann_cost_matrix(x, y, c=1.0, eps=1e-8):
    """
    Computes pairwise Busemann distances between x and y.
    Args:
        x: Tensor (n, d)
        y: Tensor (m, d)
        c: Curvature parameter
        eps: Small value for numerical stability
    Returns:
        Cost matrix (n, m)
    """
    radius = 1.0 / safe_sqrt(c, eps)
    diff2 = ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(-1)
    denom = radius ** 2 - (x ** 2).sum(-1, keepdim=True)
    cost = torch.abs(safe_log(diff2 + eps) - safe_log(denom + eps))
    return cost

def poincare_distance_matrix(x, y, c=1.0, eps=1e-8):
    """
    Computes pairwise Poincaré distances between x and y.
    Args:
        x: Tensor (n, d)
        y: Tensor (m, d)
        c: Curvature parameter
        eps: Small value for numerical stability
    Returns:
        Distance matrix (n, m)
    """
    x_norm_sq = torch.sum(x ** 2, dim=1, keepdim=True)
    y_norm_sq = torch.sum(y ** 2, dim=1, keepdim=True)
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    diff_norm_sq = torch.sum(diff ** 2, dim=-1)
    denom = (1.0 - c * x_norm_sq) @ (1.0 - c * y_norm_sq).T
    denom = torch.clamp(denom, min=eps)
    arg = 1.0 + (2.0 * c * diff_norm_sq) / denom
    arg = torch.clamp(arg, min=1.0 + eps)
    return torch.acosh(arg)

def pot_emd(a, b, C):
    """
    Computes the optimal transport plan using EMD (Earth Mover's Distance).
    Args:
        a: Source distribution (1D tensor)
        b: Target distribution (1D tensor)
        C: Cost matrix (2D tensor)
    Returns:
        Transport plan (2D tensor)
    """
    if ot is None:
        raise ImportError("POT library required for optimal transport")
    a_np = a.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    C_np = C.detach().cpu().numpy()
    a_np = a_np / a_np.sum()
    b_np = b_np / b_np.sum()
    try:
        pi_np = ot.emd(a_np, b_np, C_np)
    except Exception as e:
        print(f"EMD failed: {e}")
        raise
    pi = torch.tensor(pi_np, device=C.device, dtype=C.dtype)
    return pi

def pot_sinkhorn(a, b, C, eps=0.1, max_iter=1000):
    """
    Computes the optimal transport plan using the Sinkhorn algorithm.
    Falls back to EMD if Sinkhorn fails.
    Args:
        a: Source distribution (1D tensor)
        b: Target distribution (1D tensor)
        C: Cost matrix (2D tensor)
        eps: Entropic regularization parameter
        max_iter: Maximum number of iterations
    Returns:
        Transport plan (2D tensor)
    """
    if ot is None:
        raise ImportError("POT library required for optimal transport")
    a_np = a.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    C_np = C.detach().cpu().numpy()
    a_np = a_np / a_np.sum()
    b_np = b_np / b_np.sum()
    try:
        pi_np = ot.sinkhorn(a_np, b_np, C_np, eps, numItermax=max_iter, verbose=False, log=False, warn=True)
    except Exception as e:
        print(f"Sinkhorn failed: {e}. Falling back to EMD.")
        pi_np = ot.emd(a_np, b_np, C_np)
    pi = torch.tensor(pi_np, device=C.device, dtype=C.dtype)
    return pi

# ---------------------- SYNTHETIC PROTOTYPE GENERATION -------------
def get_synthetic_orthogonal_prototypes(args, dim, manifold, device='cuda'):
    """
    Generates synthetic orthogonal prototypes in the tangent space.
    Args:
        args: Arguments (unused)
        dim: Feature dimension
        manifold: Manifold object with expmap0
        device: Device string
    Returns:
        Prototypes on the manifold (tensor)
    """
    print("Creating synthetic orthogonal prototypes.")
    num_protos = 1000
    random_vectors = torch.randn(num_protos, dim, device=device)
    q_matrix, _ = torch.linalg.qr(random_vectors, mode='reduced')
    orthogonal_protos = q_matrix[:num_protos]
    orthogonal_protos *= 0.5
    prototypes = manifold.expmap0(orthogonal_protos)
    print(f"Generated {prototypes.shape[0]} orthogonal prototypes.")
    return prototypes

def get_synthetic_random_prototypes(args, dim, manifold, device='cuda'):
    """
    Generates synthetic random direction prototypes in the tangent space.
    Args:
        args: Arguments (unused)
        dim: Feature dimension
        manifold: Manifold object with expmap0
        device: Device string
    Returns:
        Prototypes on the manifold (tensor)
    """
    print("Creating synthetic random direction prototypes.")
    num_protos = 1000
    random_vectors = torch.randn(num_protos, dim, device=device)
    random_directions = random_vectors / random_vectors.norm(dim=1, keepdim=True)
    random_directions *= 0.5
    prototypes = manifold.expmap0(random_directions)
    print(f"Generated {prototypes.shape[0]} random direction prototypes.")
    return prototypes

# ---------------------- HYPERBOLIC SPACE UTILS ---------------------
def lorenz_factor(x, c=1.0, dim=-1, keepdim=False):
    """
    Computes the Lorenz factor for points on the Klein disk.
    Args:
        x: Tensor
        c: Curvature
        dim: Dimension to sum over
        keepdim: Whether to keep dimension
    Returns:
        Lorenz factor tensor
    """
    return 1 / torch.sqrt(1 - c * x.pow(2).sum(dim=dim, keepdim=keepdim))

def k2p(x, c):
    """
    Projects a point from Klein model to Poincaré model.
    Args:
        x: Tensor
        c: Curvature
    Returns:
        Projected tensor
    """
    denom = 1 + torch.sqrt(1 - c * x.pow(2).sum(-1, keepdim=True))
    return x / denom

def p2k(x, c):
    """
    Projects a point from Poincaré model to Klein model.
    Args:
        x: Tensor
        c: Curvature
    Returns:
        Projected tensor
    """
    denom = 1 + c * x.pow(2).sum(-1, keepdim=True)
    return 2 * x / denom

def poincare_mean(x, dim=0, c=1.0):
    """
    Computes the mean of points in Poincaré ball using the Klein model.
    Args:
        x: Tensor of points in Poincaré model
        dim: Dimension to average over
        c: Curvature
    Returns:
        Mean point in Poincaré model
    """
    x_klein = p2k(x, c)
    lamb = lorenz_factor(x_klein, c=c, keepdim=True)
    mean_klein = torch.sum(lamb * x_klein, dim=dim, keepdim=True) / torch.sum(lamb, dim=dim, keepdim=True)
    mean_poincare = k2p(mean_klein, c)
    return mean_poincare.squeeze(dim)

def get_hyp_prototype(args, manifold, text_features, device='cuda', n_samples_per_concept=None, r=1.0):
    """
    Generates prototypes for each class/concept using text features.
    Args:
        args: Arguments (unused)
        manifold: Manifold object with expmap0
        text_features: Tensor (num_classes, feature_dim)
        device: Device string
        n_samples_per_concept: Not used
        r: Norm clipping value
    Returns:
        Prototypes on the manifold (tensor)
    """
    text_features = norm_clip(text_features, r=r)
    proto_list = []
    for i in range(text_features.shape[0]):
        feat = text_features[i].unsqueeze(0)
        feat = norm_clip(feat, r)
        proto = manifold.expmap0(feat)
        proto_list.append(proto.squeeze(0))
    return torch.stack(proto_list)

def get_infinity_targets(num_targets, model, manifold, device='cuda'):
    """
    Generates points near the boundary of the Poincaré ball in random directions.
    Args:
        num_targets: Number of target points
        model: Model with output dimension
        manifold: Manifold object with .c attribute
        device: Device string
    Returns:
        Tensor of shape (num_targets, feature_dim)
    """
    print(f"Creating {num_targets} synthetic targets at infinity.")
    c = manifold.c.item()
    radius = 1.0 / math.sqrt(c)
    # Infer feature dimension
    feature_dim = None
    try:
        feature_dim = model.fc.out_features
    except AttributeError:
        for layer in reversed(list(model.modules())):
            if isinstance(layer, torch.nn.Linear):
                feature_dim = layer.out_features
                break
    if feature_dim is None:
        raise ValueError("Could not determine feature dimension from model.")
    random_dirs = torch.randn(num_targets, feature_dim, device=device)
    random_dirs = random_dirs / random_dirs.norm(dim=-1, keepdim=True)
    targets = random_dirs * (radius - 1e-6)
    return targets

# ---------------------- BUSEMANN PENALTY MODULE --------------------
import torch.nn as nn

class BusePenalty(nn.Module):
    """
    Computes a Busemann-style penalty loss between two sets of points.
    """
    def __init__(self, dimension, mult=1.0):
        super().__init__()
        self.dimension = dimension
        self.penalty_constant = mult * self.dimension

    def forward(self, z, p):
        """
        Args:
            z: Data points (batch, d)
            p: Prototype points (batch, d)
        Returns:
            Scalar loss
        """
        diff = p - z
        diff_norm = torch.norm(diff, dim=1)
        diff_log = 2 * torch.log(diff_norm)
        z_norm = torch.norm(z, dim=1)
        proto_term = (1 - z_norm.pow(2) + 1e-6)
        proto_log = (1 + self.penalty_constant) * torch.log(proto_term)
        loss = diff_log - proto_log
        return torch.mean(loss)