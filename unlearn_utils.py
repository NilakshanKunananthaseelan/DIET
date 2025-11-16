
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

# def busemann_cost_matrix(x, y, c=1.0, eps=1e-8):
#     """
#     Computes pairwise Busemann distances between x and y.
#     Args:
#         x: Tensor (n, d)
#         y: Tensor (m, d)
#         c: Curvature parameter
#         eps: Small value for numerical stability
#     Returns:
#         Cost matrix (n, m)
#     """
#     radius = 1.0 / safe_sqrt(c, eps)
#     diff2 = ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(-1)
#     denom = radius ** 2 - (x ** 2).sum(-1, keepdim=True)
#     cost = torch.abs(safe_log(diff2 + eps) - safe_log(denom + eps))
#     return cost

def busemann_cost_matrix(x, xi, *, c=1.0, eps=1e-8):
    """Compute the pairwise Busemann distances between points in x and y."""
    radius = 1.0 / safe_sqrt(c, eps)
    diff2 = ((x.unsqueeze(1) - xi.unsqueeze(0))**2).sum(-1)
    denom = radius**2 - (x**2).sum(-1, keepdim=True)
    C = 0.95*(safe_log(diff2 + eps) - 0.05*safe_log(denom + eps))
    return C

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
def get_b_nonuniform(
    C,
    mode="uniform",
    fixed=None,
    temperature=0.2,
    device=None, dtype=None,
):
    B, K = C.shape
    device = device or C.device
    dtype = dtype or C.dtype

    if mode == "uniform":
        b = torch.full((K,), 1.0 / K, device=device, dtype=dtype)
    elif mode == "fixed":
        if fixed is None:
            raise ValueError("mode='fixed' requires `fixed` capacities.")
        b = torch.as_tensor(fixed, device=device, dtype=dtype)
        if b.numel() != K:
            raise ValueError(f"`fixed` has length {b.numel()} but K={K}.")
        b = torch.clamp(b, min=0)
        s = b.sum()
        if s <= 0:
            raise ValueError("`fixed` capacities must have positive sum.")
        b = b / s
    elif mode == "softmax_cost":
        scores = -C.detach().mean(dim=0)                 # [K]
        b = torch.softmax(scores / max(temperature, 1e-6), dim=0)
    elif mode == "argmin_counts":
        counts = torch.bincount(C.argmin(dim=1).detach(), minlength=K).float().to(device)
        b = (counts + 1e-6) / (counts.sum() + 1e-6 * K)
    else:
        raise ValueError(f"Unknown mode for non-uniform b: {mode}")
    return b

def transport_plan(
    C, a, b, ot_fn, ot_kwargs,
):
    T = ot_fn(a, b, C, **ot_kwargs)
    OT_loss = (T * C).sum()
    return T, OT_loss

def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()


# Exponential Map
def expmap0(u, *, c=1.0, t=1.0):
    r"""
    Exponential map for Poincare ball model from :math:`0`.
    .. math::
        \operatorname{Exp}^c_0(u) = \tanh(\sqrt{c}/2 \|u\|_2) \frac{u}{\sqrt{c}\|u\|_2}
    Parameters
    ----------
    u : tensor
        speed vector on poincare ball
    c : float|tensor
        ball negative curvature
    t : float|tensor
        tanh hyper parameter
    Returns
    -------
    tensor
        :math:`\gamma_{0, u}(1)` end point
    """
    # print('in exmap0')
    c = torch.as_tensor(c).type_as(u)
    return _expmap0(u, c, t=t)


def _expmap0(u, c, t=1.0):
    sqrt_c = c ** 0.5
    u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), 1e-5)
    gamma_1 = tanh(sqrt_c * u_norm * t) * u / (sqrt_c * u_norm)
    return gamma_1


# Poincare distance
def dist(x, y, *, c=1.0, keepdim=False):
    r"""
    Distance on the Poincare ball
    .. math::
        d_c(x, y) = \frac{2}{\sqrt{c}}\tanh^{-1}(\sqrt{c}\|(-x)\oplus_c y\|_2)
    .. plot:: plots/extended/poincare/distance.py
    Parameters
    ----------
    x : tensor
        point on poincare ball
    y : tensor
        point on poincare ball
    c : float|tensor
        ball negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    Returns
    -------
    tensor
        geodesic distance between :math:`x` and :math:`y`
    """
    c = torch.as_tensor(c).type_as(x)
    return _dist(x, y, c, keepdim=keepdim)


def _dist(x, y, c, keepdim: bool = False):
    sqrt_c = c ** 0.5
    dist_c = artanh(sqrt_c * _mobius_add(-x, y, c).norm(dim=-1, p=2, keepdim=keepdim))
    return dist_c * 2 / sqrt_c


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        res = (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


def artanh(x):
    return Artanh.apply(x)


# Mobius addition
def _mobius_add(x, y, c):
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / (denom + 1e-5)

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
        return loss
        # return torch.mean(loss)

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