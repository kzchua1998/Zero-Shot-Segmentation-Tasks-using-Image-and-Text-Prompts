# Transformer-Based-Zero-Shot-Segmentation-Methods-using-Image-and-Text-Prompts
This repository will experiment, compare and review transformer-based models in HuggingFace `transformers` for zero-shot segmentation tasks. The repository will be updated from time to time to cover more `transformers` and their combinations revolving zero-shot image segmentation tasks using text and image prompts.

# Set-up Environment

1. Install `transformers` from source for latest updates.

   ``` shell
   pip install -q git+https://github.com/huggingface/transformers.git
   ```
  
2. Install python requirements (Python >= 3.8).

   ``` shell
   pip install -r requirements.txt
   ```


# CLIPSeg

The CLIPSeg model was proposed in ["Image Segmentation Using Text and Image Prompts"](https://arxiv.org/abs/2112.10003)
- adds a minimal decoder on top of a frozen `CLIP` model for zero and one-shot image segmentation
- generate image segmentations based on arbitrary `image` or `text` prompts

<p align="center">
<img src="https://github.com/kzchua1998/Zero-Shot-Segmentation-Tasks-using-Image-and-Text-Prompts/assets/64066100/24ec533e-c141-4421-b8af-d1d85cc54119" width="550" height="250">
</p>

## Quickstart
The `clipseg.ipynb` notebook provides codes for using `CLIPSeg` pre-trained model. If you run locally, make sure to download the model from the link provided in the `Models` section below. 

## Models
- `CIDAS/clipseg-rd16`: `CLIPSeg` with reduce dimension 16 ([Download](https://huggingface.co/CIDAS/clipseg-rd16))
- `CIDAS/clipseg-rd64`: `CLIPSeg` with reduce dimension 64 ([Download](https://huggingface.co/CIDAS/clipseg-rd64))
- `CIDAS/clipseg-rd64-refined`: `clipseg-rd64` refined with complex convolution ([Download](https://huggingface.co/CIDAS/clipseg-rd64-refined))

To change model, simply modify the below cell in `clipseg.ipynb` as below
``` shell
# CIDAS/clipseg-rd16
processor = CLIPSegProcessor.from_pretrained(r"CIDAS/clipseg-rd16")
model = CLIPSegForImageSegmentation.from_pretrained(r"CIDAS/clipseg-rd16")

# CIDAS/clipseg-rd64
processor = CLIPSegProcessor.from_pretrained(r"CIDAS/clipseg-rd64")
model = CLIPSegForImageSegmentation.from_pretrained(r"CIDAS/clipseg-rd64")

# CIDAS/clipseg-rd64-refined
processor = CLIPSegProcessor.from_pretrained(r"CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained(r"CIDAS/clipseg-rd64-refined")
```
The corresponding `CLIPSeg` model will be downloaded automatically from hub.

## Result Visualization
The `CLIPSeg` model can generate image segmentations based on arbitrary `image` or `text` prompts. The `clipseg.ipynb` notebook demonstrates both use cases as shown below.

### Text Prompt
While further prompt engineering is highly recommended, the text prompt was set to `green milo can` for demo purpose.
``` shell
image = Image.open(r"sample_images/target.jpg")
text_prompts = ["green milo can"]
result = forward_pass_text(image,text_prompts,model,processor)
```
<p align="center">
<img src="https://github.com/kzchua1998/Zero-Shot-Segmentation-Tasks-using-Image-and-Text-Prompts/assets/64066100/a7126147-d5e1-4c01-8c2e-973e31a18eb5">
</p>

### Image Prompt
For image prompt, it is highly suggested to use masked image as shown in `sample_images/prompt.jpg` to minimize embedding noise. 
``` shell
image = Image.open(r"sample_images/target.jpg")
image_prompt = Image.open(r"sample_images/prompt.jpg")
result = forward_pass_image(image,image_prompt,model,processor)
```
<p align="center">
<img src="https://github.com/kzchua1998/Zero-Shot-Segmentation-Tasks-using-Image-and-Text-Prompts/assets/64066100/af57fad7-5e5b-43ca-9152-76707fff59c9">
</p>

### CLIPSeg Optmization and Quantization
coming soon

