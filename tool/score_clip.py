import csv
import os
import torch
_ = torch.manual_seed(42)
import jsonlines
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging

version = "openai/clip-vit-large-patch14"
device = "cuda"
processor = CLIPProcessor.from_pretrained(version)
tokenizer = CLIPTokenizer.from_pretrained(version)
clip = CLIPModel.from_pretrained(version).to(device)

def set_logger(path,file_path=None):
    os.makedirs(path,exist_ok=True)
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

def get_t2m(img_paths, txt_inputs, batch_size=8):
    assert len(img_paths) == len(txt_inputs)
    print(txt_inputs)
    print(len(txt_inputs))
    images = [Image.open(i) for i in img_paths]
    img_features, txt_features = [], []
    for i in tqdm(range(0, len(images), batch_size)):
        batch_images = images[i: min(i + batch_size, len(images))]
        batch_img_inputs = processor(images=batch_images, return_tensors="pt").to(device)
        batch_img_features = clip.get_image_features(**batch_img_inputs).cpu().detach().numpy()
        img_features.append(batch_img_features)

        batch_texts = txt_inputs[i: min(i + batch_size, len(images))]
        batch_txt_inputs = tokenizer(batch_texts, padding=True, return_tensors="pt").to(device)
        # batch_txt_inputs = tokenizer(batch_texts, truncation = True, max_length = 77, return_length = False,
        #                              padding = "max_length", return_tensors = "pt").to(device)
        batch_txt_features = clip.get_text_features(**batch_txt_inputs).cpu().detach().numpy()
        txt_features.append(batch_txt_features)

    img_features = np.concatenate(img_features, axis=0)
    txt_features = np.concatenate(txt_features, axis=0)
    assert len(img_features) == len(images)

    img_features, txt_features = [
        x / np.linalg.norm(x, axis=-1, keepdims=True)
        for x in [img_features, txt_features]
    ]
    return np.mean((img_features * txt_features).sum(axis=-1))

def get_m2m(img_paths):
    images = [Image.open(i) for i in img_paths]
    img1_features, img2_features = [], []
    for i in tqdm(range(0, len(images))):
        img1 = images[i: min(i + 1, len(images))]
        # img2 = images[i + 1: min(i + 2, len(images))]
        # assert len(img1) == len(img2)
        img1_inputs = processor(images=img1, return_tensors="pt").to(device)
        batch_img1_features = clip.get_image_features(**img1_inputs).cpu().detach().numpy()
        # img2_inputs = processor(images=img2, return_tensors="pt").to(device)
        # batch_img2_features = clip.get_image_features(**img2_inputs).cpu().detach().numpy()
        img1_features.append(batch_img1_features)
        # img2_features.append(batch_img2_features)

    img2_features = np.concatenate(img1_features, axis=0)[1:len(img1_features), :]
    img1_features = np.concatenate(img1_features, axis=0)[0:len(img1_features)-1, :]

    img1_features, img2_features = [
        x / np.linalg.norm(x, axis=-1, keepdims=True)
        for x in [img1_features, img2_features]
    ]
    return np.mean((img1_features * img2_features).sum(axis=-1))

def get_clip_from_list(type, text_list, translate_list):
    '''

    Args:
        text_list: list of str
        translate_list: list of dir
        source_list: list of dir

    Returns:

    '''
    assert len(text_list) == len(translate_list)
    # n = 0
    score = []
    for text, translate_dir in zip(text_list, translate_list):
        # 若这个video未应用这种方法，则跳过
        if not os.path.exists(translate_dir):
            score.append(0)
            continue
        print(f'Text Input:  {text}')
        print(f'Trans Dir:  {translate_dir}')
        img_paths = []
        txt_inputs = []
        for name in os.listdir(translate_dir):    # '0.png', '1.png', ..., '7.png'
            img_paths.append(os.path.join(translate_dir, name))       # './outputs/baseline1/car10-a red car/results/fate/0.png'
            txt_inputs.append(text)
        print('Sample Cnt: {}'.format(len(img_paths)))
        if type == 't2m':
            mean_clip_score = get_t2m(img_paths, txt_inputs, 1)
        else:
            mean_clip_score = get_m2m(img_paths)
        print('Clip Score: {}'.format(mean_clip_score))
        # print(type(mean_clip_score)) # numpy.float32
        score.append(mean_clip_score)
        logging.info(f'CLIP SCORE-{text}: {mean_clip_score}')
        print(f'CLIP SCORE-{text}: {mean_clip_score}')
        # n += 1
    # if n == 0:
    #     score = 0
    # else:
    #     score = sum(score)/len(score)
    return score

def cal_score(videos, task, method):
    '''

    Args:
        task: str, type of score to calculate, e.g.: 'CLIP'
        text_list: list, e.g.: ['prompt0', 'prompt1', ..., 'prompt42']
        translate_list: list, directory of output images, e.g.: ['./baseline/video0/results/fate', './baseline/video1/results/fate', ...]
        source_list:  list, directory of source images, e.g.: ['./baseline/video0/results/origin', './baseline/video1/results/origin', ...]

    Returns:
        n: int, length of videos
        score: float

    '''
    # assert len(text_list) == len(translate_list) == len(source_list)
    out_root = '/workspace/home/zhaomin/cephfs-thu/zhaomin/projects/rongzhen/Video-ControlNet-min/outputs/test3'

    # text_list: 每个视频对应的prompt
    text_list = []
    # translate_list: 每个视频对应的输出结果的目录
    translate_list = []
    # source_list: 每个视频对应的原视频目录
    source_list = []
    for idx, video in enumerate(videos):
        text_list.append(video['target'][0])
        translate_list.append(os.path.join(out_root, f"{video['name']}-{video['target'][0]}", f"results/{method}"))
        source_list.append(os.path.join(out_root, f"{video['name']}-{video['target'][0]}", f"results/origin"))

    # if task == 'CLIP':
    #     n, score = get_clip_from_list(text_list, translate_list)
    #     print(f"{task}: {score}")
    # else:
    #     print('NO SUCH SCORE TYPE!')
    # 设置存储路径：最后会保存为 samplepath/fate.txt
    set_logger(os.path.join(out_root, "A-score"), f'{method}.txt')
    # # 记录信息
    score_t2m = get_clip_from_list('t2m', text_list, translate_list)
    score_m2m = get_clip_from_list('m2m', text_list, translate_list)


    return score_t2m, score_m2m

def non_zero_mean(np_arr):
    exist = (np_arr != 0)
    num = sum(np_arr)
    den = sum(exist)
    return num / den

if __name__ == '__main__':
    # parser = get_parser()
    # opt = parser.parse_args()
    # 输入：task
    task = 'CLIP'
    # 可设置：待测试的方法
    controls = ['ours-canny', 'ours-hed', 'ours-depth', 'ours-openpose', 'ours-mlsd']
    other_methods = ['origin', 'fate', 'p2p', 'v2v', 'tuneavideo']
    methods = controls + other_methods

    # 可设置：待测video_list
    video_list = '/workspace/home/zhaomin/cephfs-thu/zhaomin/projects/rongzhen/Video-ControlNet-min/videos.jsonl'
    with jsonlines.open(video_list, 'r') as reader:  # The reader and can be used directly. It contains the lines in the json file.
        videos = [video for video in reader]  # a list of dict
    reader.close()

    # videos = videos[0:2]

    save_t2m = np.zeros(shape = (len(videos), len(methods)))
    save_m2m = np.zeros(shape = (len(videos), len(methods)))

    # 对每种方法在video_list上计算得分，并保存结果
    for idx, method in enumerate(methods):
        save_t2m[:,idx], save_m2m[:,idx] = cal_score(videos, task, method)

    # Ours 对每种control取平均
    save_t2m = np.hstack((save_t2m, save_t2m[:,0:len(controls)].max(axis=1).reshape(-1, 1)))
    save_t2m = np.vstack((save_t2m, np.mean(save_t2m, axis=0).reshape(1, -1)))
    save_m2m = np.hstack((save_m2m, save_m2m[:,0:len(controls)].max(axis=1).reshape(-1, 1)))
    save_m2m = np.vstack((save_m2m, np.mean(save_m2m, axis=0).reshape(1, -1)))

    # 写入csv文件
    methods.append('ours')
    save_root = '/workspace/home/zhaomin/cephfs-thu/zhaomin/projects/rongzhen/Video-ControlNet-min/outputs/test3'
    with open(os.path.join(save_root, 'A-score/t2m.csv'), "w") as csvfile:
        writer = csv.writer(csvfile)
        # 先写入columns_name
        writer.writerow(methods)
        # 写入多行用writerows
        writer.writerows(save_t2m)
    with open(os.path.join(save_root, 'A-score/m2m.csv'), "w") as csvfile:
        writer = csv.writer(csvfile)
        # 先写入columns_name
        writer.writerow(methods)
        # 写入多行用writerows
        writer.writerows(save_m2m)










