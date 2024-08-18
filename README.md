## Environment
```
conda env create -f environment.yml
```

## Prepare Pretrained Text-to-Image Diffusion Model
Download the [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) and ControlNet 1.0 for [canny](https://huggingface.co/lllyasviel/sd-controlnet-canny/tree/main), [HED](https://huggingface.co/lllyasviel/sd-controlnet-hed), [depth](https://huggingface.co/lllyasviel/sd-controlnet-depth) and [pose](https://huggingface.co/lllyasviel/sd-controlnet-openpose). Put them in ```./``` .

## Run Demos 
Download the [data](https://drive.google.com/drive/folders/1RrYCaq6QxSVD2K4wJFrTyDnISli8f625?usp=sharing) and put them in ```videos/```.
```
python run_demos.py
```
## Test
```
python run_eval.py
```