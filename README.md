# DeepSeek Janus Pro 7B Demo Notebook

This notebook demonstrates how to use the DeepSeek Janus Pro 7B model for both image generation and image understanding tasks. The notebook showcases the model's capabilities in multimodal interactions, including text-to-image generation and image-to-text understanding.

## Features

- Text-to-Image Generation
- Image Understanding and Analysis
- Support for Multi-turn Conversations
- Configurable Generation Parameters

## Prerequisites

- Python 3.x
- PyTorch
- CUDA-capable GPU (recommended)
- ~15GB of storage space for model weights

## Installation

1. Clone the Janus repository:
```bash
git clone https://github.com/deepseek-ai/Janus
cd Janus
pip install -e .
```

## Usage

### Model Initialization

```python
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor

model_path = "deepseek-ai/Janus-Pro-7B"
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt = AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True
).to(torch.bfloat16).cuda().eval()
```

### Text-to-Image Generation

The notebook includes a powerful image generation function with configurable parameters:

- Temperature control
- Parallel generation (16 images by default)
- CFG weight adjustment
- Customizable image and patch sizes

Example usage:
```python
conversation = [
    {
        "role": "<|User|>",
        "content": "A stunning ginger Mainecoon cat, look at the camera in the style of a nat geo portrait",
    },
    {"role": "<|Assistant|>", "content": ""},
]

generate(
    vl_gpt,
    vl_chat_processor,
    prompt,
)
```

### Image Understanding

The model can also analyze and describe images:

```python
conversation = [
    {
        "role": "<|User|>",
        "content": f"<image_placeholder>\n What is this place. Tell me it's history",
        "images": ['path_to_image.png'],
    },
    {"role": "<|Assistant|>", "content": ""},
]

# Prepare inputs and generate response
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation, 
    images=pil_images, 
    force_batchify=True
).to(vl_gpt.device)
```

## Key Parameters

### Image Generation
- `temperature`: Controls randomness in generation (default: 1.0)
- `parallel_size`: Number of images to generate in parallel (default: 16)
- `cfg_weight`: Classifier-free guidance weight (default: 5.0)
- `image_token_num_per_image`: Token count per image (default: 576)
- `img_size`: Output image size (default: 384)
- `patch_size`: Size of image patches (default: 16)

## Output Examples

The notebook includes visualization utilities to display generated images in a 4x4 grid using matplotlib.

## Requirements

- torch
- transformers
- PIL
- numpy
- matplotlib

## Limitations

- Requires significant GPU memory
- Model weights are approximately 15GB
- Best performance on CUDA-capable GPUs

## License

Please refer to the DeepSeek Janus repository for licensing information.

## Acknowledgments

This notebook demonstrates the capabilities of the DeepSeek Janus Pro 7B model. For more information, visit the [DeepSeek AI repository](https://github.com/deepseek-ai/Janus).
