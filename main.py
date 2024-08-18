import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from omegaconf import OmegaConf
from train import main
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--function",
        type=str,
        default='basic',
        choices=["basic", "long", "image"],
        help="the function for ControlVideo"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default='outputs/',
        help="the path for saving"
    )

    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default= "./stable-diffusion-v1-5",
        help='the path of main unet, e.g., stable diffusion'
    )

    parser.add_argument(
        "--pretrained_controlnet_path",
        type=str,
        default="./sd-controlnet-hed",
        help='the path of controlnet'
    )

    # training setting
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1,
        help="the steps for training"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="the learning rate of the optimizer"
    )

    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1,
        help="the frequency of validation step"
    )

    # configs for controls
    parser.add_argument(
        "--type",
        type=str,
        default='hed',
        # choices=["hed", "canny"],
        help="the type of control"
    )

    parser.add_argument(
        "--control_scale",
        type=float,
        default=1.0,
        help="the control scale"
    )

    parser.add_argument(
        "--high_threshold",
        type=float,
        default=100.0,
        help="the threshold for canny"
    )

    parser.add_argument(
        "--low_threshold",
        type=float,
        default=100.0,
        help="the threshold for canny"
    )

    # configs for training data
    parser.add_argument(
        "--video_path",
        type=str,
        default='videos/car11.mp4',
        help="the path to the input video"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default='a car, autumn',
        help="the prompt for source video"
    )

    parser.add_argument(
        "--n_sample_frames",
        type=int,
        default=8,
        help="the number of sampled frames of the input video"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="the width"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="the height"
    )

    parser.add_argument(
        "--sample_start_idx",
        type=int,
        default=0,
        help='the start index for sampling input video and 0 indicates the first frame'
    )

    parser.add_argument(
        "--sample_frame_rate",
        type=int,
        default=0,
        help="the number of skip frames for sampling input video and 0 indicate uniform sampling"
    )

    # configs for validation
    parser.add_argument(
        "--prompts",
        type=str,
        default='a Swarovski crystal swan is swimming in a river',
        help='target prompts',
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=12.0,
        help="classifier-free guidance"
    )

    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="the number of denoising steps for sampling, e.g. 50 for DDIM"
    )

    parser.add_argument(
        "--start",
        type=str,
        choices=["inversion", "noise", "sdedit"],
        default="inversion",
        help='the type of forward diffusion process'
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.8,
        help='the strength for sdedit, which is between 0 and 1'
    )

    parser.add_argument(
        "--edit_type",
        type=str,
        choices=["DDIM", "DPMSOLVER"],
        default="DDIM",
        help='the sampling methods'
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=6,
        help='the overlap for long video editing'
    )

    parser.add_argument(
        "--sub_frames",
        type=int,
        default=12,
        help='the number of frames of short videos for long video editing'
    )

    parser.add_argument(
        "--key_weight",
        type=float,
        default=0.3,
        help='the weight for key frame video fusion'
    )

    parser.add_argument(
        "--var",
        type=float,
        default=0.1,
        help='the weight for nearby video fusion'
    )

    parser.add_argument(
        "--weights_type",
        type=str,
        default='Gaussian',
        help='the weight type for nearby video fusion'
    )

    # configs for lora
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default="ckpts/lora_dopfunk_evangelinelilly_16.safetensors",
        help='the path of lora'
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help='the weight for merge lora'
    )

    args = parser.parse_args()
    name = args.video_path.split('/')[-1]
    name = name.split('.')[0]
    config_root = "./configs/basic.yaml"
    para = OmegaConf.load(config_root)

    # update params in basic.yaml
    for keys in para.keys():
        if keys == 'train_data':
            for key in para[keys].keys():
                para.train_data[key] = getattr(args, key)
        elif keys == 'control_config':
            for key in para[keys].keys():
                para.control_config[key] = getattr(args, key)
        elif keys == 'validation_data':
            for key in para[keys].keys():
                if hasattr(args, key):
                    if key=='prompts':
                        para.validation_data[key] = [getattr(args, key)]
                    else:
                        para.validation_data[key] = getattr(args, key)
        elif keys == 'lora':
            for key in para[keys].keys():
                para.lora[key] = getattr(args, key)
        else:
            if hasattr(args, keys):
                para[keys] = getattr(args, keys)
    para.output_dir = os.path.join(para.output_dir, f"{name}-{args.prompts}")
    para.validation_data['video_length'] = para.train_data.n_sample_frames
    para.validation_data['width'] = para.train_data.width
    para.validation_data['height'] = para.train_data.height

    main(**para)







