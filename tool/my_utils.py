import torch
import numpy as np
import os
import datetime
import shutil
import pprint
import argparse
import einops
from PIL import Image
import cv2


def video2images(datapath, saveroot, fps):
    os.makedirs(saveroot,exist_ok=True)
    import cv2
    vidcap = cv2.VideoCapture(datapath)
    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        hasFrames, image = vidcap.read()
        c = "%05d" % count
        savepath = os.path.join(saveroot, str(c) + ".jpg")
        if hasFrames:
            cv2.imwrite(savepath, image)  # save frame as JPG file
        return hasFrames

    sec = 0
    frameRate = 1.0/fps  # //it will capture image in each 0.5 second
    count = 1
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)

def imgcat2video(video_dir, fps,source_path, translate_path):
    """
    This function will concatenate two list of images and translate it to a video

    Args:
    :param video_dir: the path to save the video
    :param fps: Frames per second for saving video
    :param source_path: the path of source image
    :param translate_path: the path of translated image
    """

    image_list = sorted(os.listdir(translate_path))
    image_example = cv2.imread(os.path.join(translate_path, image_list[1]))
    H, W, C = image_example.shape
    img_size = (2*W + 20, H + 10)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # opencv3.0
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

    for idx in image_list:
        if idx.endswith('.jpg'):
            img1 = os.path.join(source_path, idx)
            img2 = os.path.join(translate_path, idx)
            print(img1)
            frame1 = cv2.imread(img1)
            frame2 = cv2.imread(img2)
            # match the size of the images in translate path
            frame1 = cv2.resize(frame1, (W, H), interpolation=cv2.INTER_LANCZOS4)
            frame1 = cv2.copyMakeBorder(frame1, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            frame2 = cv2.copyMakeBorder(frame2, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            frame = cv2.hconcat([frame1, frame2])
            videoWriter.write(frame)
    videoWriter.release()

def list_filenames(datapath, savepath):
    """
        list the filename as text
        Args:
            datapath: the path need to list filenames
            savepath: the path to save txt file
    """
    files = sorted(os.listdir(datapath))
    file_write_obj = open(savepath, 'w')
    for var in files:
        file_write_obj.writelines(var)
        file_write_obj.write('\n')
    file_write_obj.close()

def resize_mask(mask,s):
    """
    Resize mask
    Args:
        mask: input mask with shape (B, 1, H, W)
    Returns:
        resized_mask: the resized mask with shape (B, 1, H//s, W//s)
    """
    B, C, H, W = mask.size()
    resized_mask = torch.nn.functional.interpolate(
        mask,
        size=(H // s, W // s),  # latent shape
        mode='nearest'
    )
    return resized_mask

def rescale(img, type='forward'):
    '''
    rescale for image
    Args:
        img: given image
        type: {'forward',''backward''}, 'forward' means transforming image from [0,1] to [-1,1]
        and 'backward' means transforming image from [-1,1] to [0,1]
    Returns:
        img: the rescaled image
    '''
    if type == 'backward':
        img = (img + 1.0)/2.0
        img = torch.clamp(img, min=0.0, max=1.0)
    elif type == 'forward':
        img = 2.0 * img - 1.0
        img = torch.clamp(img, min=-1.0, max=1.0)
    else:
        raise ValueError(type)
    return img

def load_sd_img(path):
    """
    load image for stable diffusion
    Args:
        path: the path of image
    """
    img = Image.open(path).convert("RGB")
    H, W = img.size
    H, W = map(lambda x: x - x % 64, (H, W))  # resize to integer multiple of 64
    img = img.resize((H, W), resample=Image.Resampling.LANCZOS)
    img = np.array(img)
    img = einops.rearrange(img, 'H W C -> C H W')
    img = torch.from_numpy(img.astype(np.float32))
    img = rescale(img/255.0, type='forward')# {0,……,255} to [-1,1]
    return img

def save_sd_image(img: torch.Tensor, path):
    """
    save image for stable diffusion
    Args:
        img: the image for saving with shape (C,H,W) and scale [-1,1]
        path: the path for save image
    """
    img = rescale(img, type='backward') * 255 #[-1,1] to {0,……,255}
    img = einops.rearrange(img.cpu().numpy(), 'C H W -> H W C')
    Image.fromarray(img.astype(np.uint8)).save(path)


def load_sd_imgs(path):
    """
    load a list of images for stable diffusion
    Args:
        path: the path of images
    """
    images = []
    names = []
    frame_paths = sorted(os.listdir(path))
    for frame_path in frame_paths:
        if frame_path.endswith('.jpg'):
            name = frame_path.split('/')[-1]
            img = load_sd_img(os.path.join(path, frame_path))
            images.append(img)
            names.append(name)
    return images, names

def set_seed(seed=1234):
    """
    set seed
    Args:
        seed: given seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def backup_codes(path, names):
    """
    backup code
    Args:
        path: the path for saving code
        names: the name of folder need to backup code
    """
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_path = os.path.realpath(os.path.join(current_path, os.pardir))

    path = os.path.join(path, "codes_{}".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    os.makedirs(path, exist_ok=True)

    for name in names:
        if os.path.exists(os.path.join(root_path, name)):
            shutil.copytree(os.path.join(root_path, name), os.path.join(path, name))

    pyfiles = filter(lambda x: x.endswith(".py"), os.listdir(root_path))
    for pyfile in pyfiles:
        shutil.copy(os.path.join(root_path, pyfile), os.path.join(path, pyfile))


def backup_profile(profile: dict, path):
    """
    backup args profile
    Args:
        profile: the args profile need to backup code
        path: the path for saving args profile
    """
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "profile_{}.txt".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    s = pprint.pformat(profile)
    with open(path, 'w') as f:
        f.write(s)

def available_devices(threshold=5000,n_devices=None):
    """
    search for available GPU devices
    Args:
        threshold: the devices with larger memory than threshold is available
        n_devices: the number of devices
    Returns:
        device: the id for available GPU devices
    """
    memory = list(os.popen('nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader'))
    mem = [int(x.strip()) for x in memory]
    devices = []
    for i in range(len(mem)):
        if mem[i] > threshold:
            devices.append(i)
    device = devices if n_devices is None else devices[:n_devices]
    return device

def format_devices(devices):
    if isinstance(devices, list):
        return ','.join(map(str,devices))


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

import logging

def set_logger(path, file_path=None):
    os.makedirs(path,exist_ok=True)
    #logger to print information
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    handler1 = logging.StreamHandler()
    if file_path is not None:
        handler2 = logging.FileHandler(os.path.join(path,file_path), mode='w')
    else:
        handler2 = logging.FileHandler(os.path.join(path, "logs.txt"), mode='w')
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)