# Transformer-Based-Zero-Shot-Segmentation-Methods-using-Image-and-Text-Prompts
This repository will experiment, compare and review transformer-based models in HuggingFace `transformers` for zero-shot segmentation tasks

# Quickstart

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
- adds a minimal decoder on top of a frozen CLIP model for zero and one-shot image segmentation
- generate image segmentations based on arbitrary image or text prompts

<br>
<p align="center">
<img src="https://github.com/kzchua1998/Zero-Shot-Segmentation-Tasks-using-Image-and-Text-Prompts/assets/64066100/24ec533e-c141-4421-b8af-d1d85cc54119" width="550" height="250">
</p>
</br>
