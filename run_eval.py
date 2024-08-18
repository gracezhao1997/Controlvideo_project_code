import os
from tool.my_utils import available_devices, format_devices, set_logger
device = available_devices(threshold=20000, n_devices=1)
os.environ["CUDA_VISIBLE_DEVICES"] = format_devices(device)
print("available_device: "+str(device))
import torch
import csv
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel
from tool.score_clip import get_t2m

_ = torch.manual_seed(42)
import jsonlines
import numpy as np
import logging


def cal_score(type, text_list, translate_list, source_list):
    '''

    Args:
        text_list: list of str
        translate_list: list of dir
        source_list: list of dir

    Returns:

    '''
    assert len(text_list) == len(translate_list) == len(source_list)
    score = []

    for (text, translate_dir, source_dir) in zip(text_list, translate_list, source_list):
        print(translate_dir,source_dir)
        # 若这个video未应用这种方法，则跳过
        if not os.path.exists(translate_dir):
            score.append(0)
            continue

        img_paths = []
        txt_inputs = []
        for name in os.listdir(translate_dir):    # '0.png', '1.png', ..., '7.png'
            img_paths.append(os.path.join(translate_dir, name))       # './outputs/baseline1/car10-a red car/results/fate/0.png'
            txt_inputs.append(text)
        print('Sample Cnt: {}'.format(len(img_paths)))

        if type == 'clip-text':
            mean_score = get_t2m(img_paths, txt_inputs, 1)
        score.append(mean_score)

        logging.info(f'{type} SCORE-video{i}-{text}: {mean_score}')
        print(f'{type} SCORE-video{i}-{text}: {mean_score}')
    return score

if __name__ == '__main__':

    # 之前保存结果的路径
    out_root = 'outputs/results'

    # 待测video_list
    video_list = 'demos.jsonl'
    with jsonlines.open(video_list, 'r') as reader:  # The reader and can be used directly. It contains the lines in the json file.
        videos = [video for video in reader]  # a list of dict
    reader.close()

    text_list = []        # text_list: 每个视频对应的prompt
    translate_list = []        # translate_list: 每个视频对应的输出结果的目录
    source_list = []        # source_list: 每个视频对应的原视频目录
    for i, video in enumerate(videos):
        text_list.append(video['target'][0])
        step = video['step']
        translate_list.append(os.path.join(out_root, f"{video['name']}-{video['target'][0]}", f"{step}/controlvideo"))
        source_list.append(os.path.join(out_root, f"{video['name']}-{video['target'][0]}", f"{step}/origin"))

    set_logger(os.path.join(out_root, "A-score"), f'scores.txt')
    scores = cal_score('clip-text', text_list, translate_list, source_list)
    print(np.mean(scores))





