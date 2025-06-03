# Character Lora Dataset Workflow

This repository contains an example ComfyUI workflow and a helper script
to create a training dataset for a character LoRA. The workflow performs
the following steps:

1. Load each input image from `input_images/`.
2. Use the Segment Anything (SAM) model to find the person in the image.
3. Apply the mask to keep only the person and save a PNG to `masked/`.
4. Crop the masked image so only the head and shoulders remain and save
to `head_shoulders/`.

The workflow itself is stored in
[`workflows/character_lora_mask_and_crop.json`](workflows/character_lora_mask_and_crop.json).
Load this file in ComfyUI to process your images.

## Prerequisites

Install the OpenCV Python bindings before using the cropping script:

```bash
pip install opencv-python
```

After running the ComfyUI workflow you can further refine the cropped
images using the script in [`scripts/crop_dataset.py`](scripts/crop_dataset.py):

```bash
python scripts/crop_dataset.py masked/ head_shoulders/ --size 512
```

Use `--fraction` to control how much of the body to keep (default `0.6`).
The `--size` option resizes the result to a square image, and
`--keep-alpha` preserves the alpha channel instead of converting to RGB.
