import jsonlines
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
from util import available_devices, format_devices
# device = available_devices(threshold=50000, n_devices=1)
# os.environ["CUDA_VISIBLE_DEVICES"] = format_devices(device)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

video_list = 'demos.jsonl'
with jsonlines.open(video_list, 'r') as reader:
    videos = [video for video in reader]
reader.close()
for video in videos:
    type = video['type']
    pretrained_controlnet_path = f"./sd-controlnet-{type}"

    if "start" in video:
        start = video['start']
    else:
        start = "inversion"

    video_path = os.path.join('videos', video['name']+'.mp4')
    prompt = video['source']
    prompts = video['target'][0]
    max_train_steps = video['step']
    validation_steps = max_train_steps

    output_dir = 'outputs/results'

    command = 'python main.py --validation_steps %s --output_dir %s --type %s --pretrained_controlnet_path %s --start %s --video_path %s --prompt "%s" --prompts "%s" --max_train_steps %s' % (
    validation_steps, output_dir, type, pretrained_controlnet_path, start, video_path, prompt, prompts, max_train_steps)
    os.system(command)

