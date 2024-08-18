import os

from PIL import Image
from torch.utils import data
from torchvision.transforms import transforms
import warnings

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def get_l2_psnr_mse_ssim(translate_path,source_path):
    '''

    Args:
        translate_path: str, directory of saving the images of a translated video
        source_path: str, directory of saving the images of the source video

    Returns:
        l2_distance, mse, psnr, ssim: floats, the scores interested in
    '''
    path1 = translate_path
    path2 = source_path

    l2_distance = calculate_l2_given_paths(path1, path2)
    print('l2:{}'.format(l2_distance))

    mse = calculate_mse(path1, path2)
    print('mse:{}'.format(mse))

    psnr_value = calculate_psnr(path1, path2)
    print('psnr:{}'.format(psnr_value))

    ssim = calculate_ssim(path1, path2)
    print('ssim:{}'.format(ssim))

    return l2_distance, mse, psnr, ssim

def imageresize2tensor(path,image_size):
    img = Image.open(path)
    convert = transforms.Compose(
        [transforms.Resize(image_size,interpolation=Image.BICUBIC), transforms.ToTensor()]
    )
    return convert(img)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    if isinstance(dir,list):
        for i in range(len(dir)):
            dir_i = dir[i]
            for root, _, fnames in sorted(os.walk(dir_i, followlinks=True)):
                for fname in fnames:
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        images.append(path)
    else:
        for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    return images[:min(max_dataset_size, len(images))]

class aligndataset(torch.utils.data.Dataset):
    def __init__(self,pathA,pathB, mask_path, transform):
        self.pathsA = sorted(make_dataset(pathA))
        self.pathsB = sorted(make_dataset(pathB))
        self.mask_paths = sorted(make_dataset(mask_path))
        self.size = len(self.pathsA)
        self.transform_image = transform[0]
        self.transform_mask = transform[1]
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        pathA = self.pathsA[index]
        imgA = Image.open(pathA).convert('RGB')
        pathB = self.pathsB[index]
        imgB = Image.open(pathB).convert('RGB')

        if self.transform_image is not None:
            imgA = self.transform_image(imgA)
            imgB = self.transform_image(imgB)

        if len(self.mask_paths) ==0 :
            mask = torch.zeros_like(imgA)
        else:
            pathmask = self.mask_paths[index]
            mask = Image.open(pathmask).convert('RGB')
            mask = self.transform_mask(mask)

        return imgA,imgB,mask

def get_dataset(source_path,target_path, mask_path, metric='fvd'):
    if metric == 'fvd':
        tran_transform = transforms.Compose(
            [
                transforms.Resize(224, interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )
        tran_transform_mask = transforms.Compose(
            [
                transforms.Resize(224, interpolation=InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )
        tran_transform = [tran_transform,tran_transform_mask]
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(512, interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )
        tran_transform_mask = transforms.Compose(
            [
                transforms.Resize(512, interpolation=InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )
        tran_transform = [tran_transform, tran_transform_mask]

    dataset = aligndataset(source_path,target_path,mask_path,tran_transform)
    return dataset

def calculate_l2_given_paths(path1,path2):
    file_name = os.listdir(path1)
    total = 0
    for name in file_name:
        s = imageresize2tensor(os.path.join(path1,name),512)
        name_i = name.split('.')[0]
        name = name_i + '.png'
        t = imageresize2tensor(os.path.join(path2,name),512)
        l2_i = torch.norm(s-t,p=2)
        total += l2_i
    return total/len(file_name)

def mse(x,y):
    r""" Computes `Peak signal-to-noise ratio (PSNR)"
            Input:
                x,y : the input image with shape (B,C,H,W)
                data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            Output:
                psnr for each image:(B,)
            """
    dim = tuple(range(1, y.ndim))
    mse_error = torch.pow(x.double() - y.double(), 2).mean(dim=dim)
    return mse_error

def calculate_mse(source_path,target_path,batch_size=50,num_workers=1):
    dataset = get_dataset(source_path, target_path)
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    total_rmse = 0
    for i, (x, y) in enumerate(dataloader):
        mse_i = mse(x, y)
        total_rmse += mse_i.sum()
    mean = total_rmse / len(dataset)
    return mean

def psnr(x,y,data_range = 1.0):
    r""" Computes `Peak signal-to-noise ratio (PSNR)"
            Input:
                x,y : the input image with shape (B,C,H,W)
                data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            Output:
                psnr for each image:(B,)
            """
    dim = tuple(range(1, y.ndim))
    mse_error = torch.pow(x.double() - y.double(), 2).mean(dim=dim)
    psnr = 10.0 * torch.log10(data_range ** 2 / (mse_error + 1e-10))
    return psnr

def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)

def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out

def _ssim(X, Y, data_range, win, size_average=True, K=(0.01, 0.03)):

    r""" Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
    Returns:
        torch.Tensor: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs

def ssim(
    X,
    Y,
    data_range=255,
    size_average=False,
    win_size=11,
    win_sigma=1.5,
    win=None,
    K=(0.01, 0.03),
    nonnegative_ssim=False,
):
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu
    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if not X.type() == Y.type():
        raise ValueError("Input images should have the same dtype.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)

def calculate_psnr(source_path,target_path,mask_paths, batch_size=50,num_workers=1):
    dataset = get_dataset(source_path, target_path,mask_paths)
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    total_psnr = 0
    for i, (x, y, mask) in enumerate(dataloader):
        unedited_mask = 1 - mask
        x = x * unedited_mask
        y = y * unedited_mask
        psnr_i = psnr(x, y, data_range=1.0)
        total_psnr += psnr_i.sum()
    mean = total_psnr / len(dataset)
    return mean

def calculate_ssim(source_path,target_path,mask_paths, batch_size=1,num_workers=1):
    dataset = get_dataset(source_path, target_path, mask_paths)
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    total_ssim = 0
    for i, (x, y,mask) in enumerate(dataloader):
        unedited_mask = 1 - mask
        x = x * unedited_mask
        y = y * unedited_mask
        ssim_ = ssim(x, y, data_range=1, size_average=False)
        total_ssim += ssim_.sum()
    mean = total_ssim / len(dataset)
    return mean

import lpips
def calculate_lpips(source_path,target_path,mask_paths):
    loss_fn = lpips.LPIPS(net='alex')
    loss_fn.cuda()
    device = torch.device('cuda')
    dist_ = []
    dataset = get_dataset(source_path, target_path,mask_paths)
    dataloader = data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )
    for i, (x, y, mask) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        unedited_mask = 1 - mask
        x = (2.0*x - 1)*unedited_mask
        y = (2.0*y - 1)*unedited_mask
        dist = loss_fn.forward(x, y)
        dist_.append(dist.mean().item())
    mean = sum(dist_) / len(dataset)
    return mean


from typing import Tuple
import scipy
import numpy as np

def compute_fvd(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0) # [d]
    sigma = np.cov(feats, rowvar=False) # [d, d]

    return mu, sigma

from tool.fvd import load_fvd_model

@torch.no_grad()
def get_feats(source_path, target_path, mask_paths):
    device = torch.device('cuda')
    i3d = load_fvd_model(device)

    dataset = get_dataset(source_path, target_path, mask_paths)
    dataloader = data.DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )

    for i, (x, y, mask) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        unedited_mask = 1 - mask
        x = (2.0 * x - 1) * unedited_mask
        y = (2.0 * y - 1) * unedited_mask
        x = x.unsqueeze(dim=0).permute(0, 2, 1, 3, 4)
        y = y.unsqueeze(dim=0).permute(0, 2, 1, 3, 4)
        x = i3d(x)
        y = i3d(y)
    return x, y

