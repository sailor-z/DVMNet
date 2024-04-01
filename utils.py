import os, random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import pickle
from torch.utils.data import DataLoader

cifar_labels = "airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck".split(",")
alphabet_labels = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z".split(" ")

def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z
    
def relative_pose_estimation(R1, T1, R2, T2):
    P1 = R1.new_zeros([R1.shape[0], 4, 4])
    P2 = R2.new_zeros([R2.shape[0], 4, 4])

    P1[:, :3, :3] = R1
    P1[:, 3, :3] = T1
    P1[:, 3, 3] = 1

    P2[:, :3, :3] = R2
    P2[:, 3, :3] = T2
    P2[:, 3, 3] = 1

    delta_P = closed_form_inverse(P1).bmm(P2)

    delta_R = delta_P[:, :3, :3]
    delta_T = delta_P[:, 3, :3]

    return delta_R, delta_T

def closed_form_inverse(se3):
    """
    Computes the inverse of each 4x4 SE3 matrix in the batch.

    Args:
    - se3 (Tensor): Nx4x4 tensor of SE3 matrices.

    Returns:
    - Tensor: Nx4x4 tensor of inverted SE3 matrices.
    """
    R = se3[:, :3, :3]
    T = se3[:, 3:, :3]

    # Compute the transpose of the rotation
    R_transposed = R.transpose(1, 2)

    # Compute the left part of the inverse transformation
    left_bottom = -T.bmm(R_transposed)
    left_combined = torch.cat((R_transposed, left_bottom), dim=1)

    # Keep the right-most column as it is
    right_col = se3[:, :, 3:].detach().clone()
    inverted_matrix = torch.cat((left_combined, right_col), dim=-1)

    return inverted_matrix

def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try: torch.manual_seed(s)
    except NameError: pass
    try: torch.cuda.manual_seed_all(s)
    except NameError: pass
    try: np.random.seed(s%(2**32-1))
    except NameError: pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def to_cuda(data):
    if type(data)==list:
        results = []
        for i, item in enumerate(data):
            results.append(to_cuda(item))
        return results
    elif type(data)==dict:
        results={}
        for k,v in data.items():
            results[k]=to_cuda(v)
        return results
    elif type(data).__name__ == "Tensor":
        return data.cuda()
    else:
        return data

def one_batch(dl):
    return next(iter(dl))


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def mk_folders(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)

def normalize_point_cloud(point_cloud):
    """
    Normalize a point cloud to an unit sphere.
    input: B x 3 x N
    """
    centroid = point_cloud.mean(dim=2, keepdim=True)
    point_cloud = point_cloud - centroid
    furthest_distance = torch.max(torch.sqrt(torch.sum(point_cloud ** 2, dim=1, keepdim=True)), dim=2, keepdim=True)[0]
    point_cloud = point_cloud / furthest_distance.clamp(min=1e-8)
    return point_cloud, furthest_distance

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, pkl_path):
    os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

def get_permutations(num_images, eval_time=False):
    if not eval_time:
        permutations = []
        for i in range(1, num_images):
            for j in range(num_images - 1):
                if i > j:
                    permutations.append((j, i))
    else:
        permutations = []
        for i in range(0, num_images):
            for j in range(0, num_images):
                if i != j:
                    permutations.append((j, i))

    return permutations

def sample_6d(num):
    samples = []
    for i in range(num):
        x = np.asarray([np.random.normal() for j in range(3)]).squeeze()
        y = np.asarray([np.random.normal() for j in range(3)]).squeeze()

        x = x / max(np.linalg.norm(x, ord=2), 1e-8)
        y = y / max(np.linalg.norm(y, ord=2), 1e-8)

        samples.append(np.concatenate([x, y], axis=-1))

    return np.asarray(samples)

def get_calibration_matrix_K_from_blender(lens, sensor_width, resolution_x, resolution_y):
    f_in_mm = lens
    resolution_x_in_px = resolution_x
    resolution_y_in_px = resolution_y
    scale = 1
    sensor_width_in_mm = sensor_width
    sensor_height_in_mm = sensor_width

    pixel_aspect_ratio = 1

    s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
    s_v = resolution_y_in_px * scale / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels

    K = np.array(
        ((alpha_u, skew,    u_0),
        (    0  , alpha_v, v_0),
        (    0  , 0,        1 )))
    return K

def generate_coords(h, w, device="cuda"):
    coords=torch.stack(torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device)), -1)
    return coords[..., (1, 0)]

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def cosine_similarity(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    src = F.normalize(src, dim=-1, p=2)
    dst = F.normalize(dst, dim=-1, p=2)
    sim = (src[:, :, None] * dst[:, None]).sum(dim=-1)
    return sim

def patchify(imgs, patch_size):
    """
    imgs: (B, 3, H, W)
    x: (B, L, patch_size**2 *3)
    """
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % patch_size == 0

    h = w = imgs.shape[2] // patch_size
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, patch_size, w, patch_size))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, patch_size**2 * 3))

    return x

def unpatchify(x, patch_size, channels=3):
    """
    x: (N, L, patch_size**2 *channels)
    imgs: (N, 3, H, W)
    """
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
    x = x.reshape(x.shape[0], h, w, patch_size, patch_size, channels)
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], channels, h * patch_size, w * patch_size))
    return imgs


def skew_symmetric(T):
    T = T.reshape(3)
    Tx = np.array([[0, -T[2], T[1]],
                    [T[2], 0, -T[0]],
                    [-T[1], T[0], 0]])
    return Tx

def resize_pad(im, dim, mode=T.InterpolationMode.BILINEAR):
    _, h, w = im.shape
    im = T.functional.resize(im, int(dim * min(w, h) / max(w, h)), interpolation=mode)
    left = int(np.ceil((dim - im.shape[2]) / 2))
    right = int(np.floor((dim - im.shape[2]) / 2))
    top = int(np.ceil((dim - im.shape[1]) / 2))
    bottom = int(np.floor((dim - im.shape[1]) / 2))
    im = T.functional.pad(im, (left, top, right, bottom))

    return im

def square_bbox(bbox, padding=0.0, astype=None):
    """
    Computes a square bounding box, with optional padding parameters.

    Args:
        bbox: Bounding box in xyxy format (4,).

    Returns:
        square_bbox in xyxy format (4,).
    """
    if astype is None:
        astype = type(bbox[0])
    bbox = np.array(bbox)
    center = (bbox[:2] + bbox[2:]) / 2
    extents = (bbox[2:] - bbox[:2]) / 2
    s = max(extents) * (1 + padding)
    square_bbox = np.array(
        [center[0] - s, center[1] - s, center[0] + s, center[1] + s],
        dtype=astype,
    )
    return square_bbox

def bbx_resize(bbx, img_w, img_h, scale_ratio=1.0):
    w, h = bbx[2] - bbx[0], bbx[3] - bbx[1]
    dim = scale_ratio * max(w, h)

    left = int(np.ceil((dim - w) / 2))
    right = int(np.floor((dim - w) / 2))
    top = int(np.ceil((dim - h) / 2))
    bottom = int(np.floor((dim - h) / 2))

    bbx[0] = max(bbx[0] - left, 0)
    bbx[1] = max(bbx[1] - top, 0)
    bbx[2] = min(bbx[2] + right, img_w)
    bbx[3] = min(bbx[3] + bottom, img_h)

    return bbx

def crop(img, bbx):
    if len(img.shape) < 4:
        crop_img = img[int(bbx[1]):int(bbx[3]), int(bbx[0]):int(bbx[2])]
    else:
        crop_img = [img[i, int(bbx[i, 1]):int(bbx[i, 3]), int(bbx[i, 0]):int(bbx[i, 2])] for i in range(img.shape[0])]
    return crop_img

def jitter_bbox(bbox, jitter_scale, jitter_trans, img_shape):
    s = (jitter_scale[1] - jitter_scale[0]) * torch.rand(1).item() + jitter_scale[0]
    tx = (jitter_trans[1] - jitter_trans[0]) * torch.rand(1).item() + jitter_trans[0]
    ty = (jitter_trans[1] - jitter_trans[0]) * torch.rand(1).item() + jitter_trans[0]

    side_length = np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]])
    center = (bbox[:2] + bbox[2:]) / 2 + np.array([tx, ty]) * side_length
    extent = side_length / 2 * s

    # Final coordinates need to be integer for cropping.
    ul = center - extent
    lr = ul + 2 * extent

    ul = np.maximum(ul, np.zeros(2))
    lr = np.minimum(lr, np.array([img_shape[1], img_shape[0]]))

    return np.concatenate((ul, lr))

def load(model_cpkt_path):
    checkpoint = torch.load(model_cpkt_path)
    self.feature_aligner.module.load_state_dict(checkpoint['predictor_state_dict'])
    self.feature_extractor.module.load_state_dict(checkpoint['feature_extractor_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch
