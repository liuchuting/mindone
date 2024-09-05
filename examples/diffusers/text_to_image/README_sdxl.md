# Stable Diffusion XL text-to-image fine-tuning

The `train_text_to_image_sdxl.py` script shows how to fine-tune Stable Diffusion XL (SDXL) on your own dataset.

🚨 This script is experimental. The script fine-tunes the whole model and often times the model overfits and runs into issues like catastrophic forgetting. It's recommended to try different hyperparameters to get the best result on your dataset. 🚨

## Running locally with MindSpore

### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

The training script is compute-intensive and only runs on an Ascend 910*. Please run the scripts with CANN version ([CANN 7.3.0.1.231](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC2.beta1)) and MindSpore version ([MS2.3.0](https://www.mindspore.cn/versions#2.3.0)); You can use
`cat {cann-install-path}/latest/version.cfg` check the CANN version. The cann-install-path indicates the installation path of CANN.

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:

```bash
git clone https://github.com/mindspore-lab/mindone
cd mindone
pip install -e .
```

Then cd in the `examples/diffusers/text_to_image` folder and run
```bash
pip install -r requirements_sdxl.txt
```

### Training

```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="YaYaB/onepiece-blip-captions"

python train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=512 --center_crop --random_flip \
  --proportion_empty_prompts=0.2 \
  --train_batch_size=1 \
  --max_train_steps=10000 \
  --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --validation_prompt="a man in a green coat holding two swords" --validation_epochs 5 \
  --checkpointing_steps=5000 \
  --output_dir="sdxl-onepiece-model-$(date +%Y%m%d%H%M%S)"
```

For parallel training, use `msrun` and along with `--distributed`:

```shell
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="YaYaB/onepiece-blip-captions"

msrun --worker_num=8 --local_worker_num=8 --log_dir=$output_dir  \
    train_text_to_image_sdxl.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --pretrained_vae_model_name_or_path=$VAE_NAME \
    --dataset_name=$DATASET_NAME \
    --resolution=512 --center_crop --random_flip \
    --proportion_empty_prompts=0.2 \
    --train_batch_size=1 \
    --max_train_steps=10000 \
    --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
    --mixed_precision="fp16" \
    --validation_prompt="a man in a green coat holding two swords" --validation_epochs 5 \
    --checkpointing_steps=5000 \
    --distributed \
    --output_dir="sdxl-onepiece-model-$(date +%Y%m%d%H%M%S)"
```

**Notes**:

*  The `train_text_to_image_sdxl.py` script pre-computes text embeddings and the VAE encodings and keeps them in memory. While for smaller datasets like [`lambdalabs/pokemon-blip-captions`](https://hf.co/datasets/lambdalabs/pokemon-blip-captions), it might not be a problem, it can definitely lead to memory problems when the script is used on a larger dataset. For those purposes, you would want to serialize these pre-computed representations to disk separately and load them during the fine-tuning process. Refer to [this PR](https://github.com/huggingface/diffusers/pull/4505) for a more in-depth discussion.
* The training command shown above performs intermediate quality validation in between the training epochs. `--report_to`, `--validation_prompt`, and `--validation_epochs` are the relevant CLI arguments here.
* SDXL's VAE is known to suffer from numerical instability issues. This is why we also expose a CLI argument namely `--pretrained_vae_model_name_or_path` that lets you specify the location of a better VAE (such as [this one](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)).

### Performance

For the above training example, we record the training speed as follows.

| NPUs | Global Batch size | Resolution   | Precision | Speed (s/step) | FPS (img/s) |
|------|-------------------|--------------|-----------|----------------|-------------|
| 1    | 1*1               | 512x512      | BF16      |                |             |
| 1    | 1*1               | 512x512      | FP16      | 0.720          | 1.38        |
| 8    | 1*8               | 512x512      | BF16      |                |             |
| 8    | 1*8               | 512x512      | FP16      |                |             |

### Inference

```python
from mindone.diffusers import DiffusionPipeline
import mindspore

model_path = "you-model-id-goes-here" # <-- change this
pipe = DiffusionPipeline.from_pretrained(model_path, mindspore_dtype=mindspore.float16)

prompt = "The boy rides a horse in space"
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5)[0][0]
image.save("onepiece.png")
```

To change the pipelines scheduler, use the from_config() method to load a different scheduler's pipeline.scheduler.config into the pipeline.

```python
from mindone.diffusers import EulerAncestralDiscreteScheduler

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5)[0][0]
image.save("onepiece.png")
```

Here are some images generated by inference under different Schedulers.

| EulerDiscreteScheduler (0.7s/step) | EulerAncestralDiscreteScheduler (0.8s/step) | LMSDiscreteScheduler (0.93s/step)       | DDIMParallelScheduler (0.86s/step) |
|------------------------------------|---------------------------------------------|-----------------------------------------|------------------------------------|
| <img src="" width=224>             | <img src="" width=224>                      | <img src="" width=224>                  |                                    |

## LoRA training example for Stable Diffusion XL (SDXL)

Low-Rank Adaption of Large Language Models was first introduced by Microsoft in [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) by *Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen*.

In a nutshell, LoRA allows adapting pretrained models by adding pairs of rank-decomposition matrices to existing weights and **only** training those newly added weights. This has a couple of advantages:

- Previous pretrained weights are kept frozen so that model is not prone to [catastrophic forgetting](https://www.pnas.org/doi/10.1073/pnas.1611835114).
- Rank-decomposition matrices have significantly fewer parameters than original model, which means that trained LoRA weights are easily portable.
- LoRA attention layers allow to control to which extent the model is adapted toward new training images via a `scale` parameter.

[cloneofsimo](https://github.com/cloneofsimo) was the first to try out LoRA training for Stable Diffusion in the popular [lora](https://github.com/cloneofsimo/lora) GitHub repository.

With LoRA, it's possible to fine-tune Stable Diffusion on a custom image-caption pair dataset
on consumer GPUs like Tesla T4, Tesla V100.

### Training

First, you need to set up your development environment as is explained in the [installation section](#installing-the-dependencies). Make sure to set the `MODEL_NAME` and `DATASET_NAME` environment variables and, optionally, the `VAE_NAME` variable. Here, we will use [Stable Diffusion XL 1.0-base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and the [OnePiece dataset](https://huggingface.co/datasets/YaYaB/onepiece-blip-captions).

**___Note: It is quite useful to monitor the training progress by regularly generating sample images during training. [Weights and Biases](https://docs.wandb.ai/quickstart) is a nice solution to easily see generating images during training. All you need to do is to run `pip install wandb` before training to automatically log images.___**

```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="YaYaB/onepiece-blip-captions"

python train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=1024 --center_crop --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=2 --checkpointing_steps=500 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --validation_prompt="a man in a green coat holding two swords" \
  --output_dir="sdxl-onepiece-model-lora-$(date +%Y%m%d%H%M%S)"
```

For parallel training, use `msrun` and along with `--distributed`:

```shell
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="YaYaB/onepiece-blip-captions"

msrun --worker_num=8 --local_worker_num=8 --log_dir=$output_dir  \
    train_text_to_image_lora_sdxl.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --pretrained_vae_model_name_or_path=$VAE_NAME \
    --dataset_name=$DATASET_NAME \
    --resolution=1024 --center_crop --random_flip \
    --train_batch_size=1 \
    --num_train_epochs=2 --checkpointing_steps=500 \
    --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
    --mixed_precision="fp16" \
    --seed=42 \
    --validation_prompt="a man in a green coat holding two swords" \
    --distributed \
    --output_dir="sdxl-onepiece-model-lora-$(date +%Y%m%d%H%M%S)"
```

The above command will also run inference as fine-tuning progresses and log the results to local files.

**Notes**:

* SDXL's VAE is known to suffer from numerical instability issues. This is why we also expose a CLI argument namely `--pretrained_vae_model_name_or_path` that lets you specify the location of a better VAE (such as [this one](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)).

### Performance

For the above training example, we record the training speed as follows.

| NPUs | Global Batch size | Resolution   | Precision | Speed (s/step) | FPS (img/s) |
|------|-------------------|--------------|-----------|----------------|-------------|
| 1    | 1*1               | 1024x1024    | BF16      | 0.760          | 1.316       |
| 1    | 1*1               | 1024x1024    | FP16      | 0.828          | 1.208       |
| 8    | 1*8               | 1024x1024    | BF16      |                |             |
| 8    | 1*8               | 1024x1024    | FP16      |                |             |

### Finetuning the text encoder and UNet

The script also allows you to finetune the `text_encoder` along with the `unet`.

🚨 Training the text encoder requires additional memory.

Pass the `--train_text_encoder` argument to the training script to enable finetuning the `text_encoder` and `unet`:

```bash
python train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=1024 --center_crop --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=2 --checkpointing_steps=500 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --validation_prompt="a man in a green coat holding two swords" \
  --train_text_encoder \
  --output_dir="sdxl-onepiece-model-lora-txt-$(date +%Y%m%d%H%M%S)"
```

For parallel training, use `msrun` and along with `--distributed`:

```shell
msrun --worker_num=8 --local_worker_num=8 --log_dir=$output_dir  \
    train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=1024 --center_crop --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=2 --checkpointing_steps=500 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --validation_prompt="a man in a green coat holding two swords" \
  --train_text_encoder \
  --distributed \
  --output_dir="sdxl-onepiece-model-lora-txt-$(date +%Y%m%d%H%M%S)"
```

### Performance

For the above training example, we record the training speed as follows.

| NPUs | Global Batch size | Resolution   | Precision | Speed (s/step) | FPS (img/s) |
|------|-------------------|--------------|-----------|----------------|-------------|
| 1    | 1*1               | 1024x1024    | BF16      | 0.994          | 1.006       |
| 1    | 1*1               | 1024x1024    | FP16      | 0.951          | 1.052       |
| 8    | 1*1               | 1024x1024    | FP32      | 1.89           | 0.529       |
| 8    | 1*8               | 1024x1024    | BF16      |                |             |
| 8    | 1*8               | 1024x1024    | FP16      |                |             |
| 8    | 1*8               | 1024x1024    | FP32      |                |             |

### Inference

Once you have trained a model using above command, the inference can be done simply using the `DiffusionPipeline` after loading the trained LoRA weights.  You
need to pass the `output_dir` for loading the LoRA weights which, in this case, is `sdxl-onepiece-model-lora`.

```python
import mindspore as ms
from mindone.diffusers import DiffusionPipeline

model_path = "takuoko/sd-pokemon-model-lora-sdxl"
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", mindspore_dtype=ms.float16)
pipe.load_lora_weights(model_path)

prompt = "A pokemon with green eyes and red legs."
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5)[0][0]
image.save("pokemon.png")
```

#### overfitting experiment

To verify the training script and the speed of convergence, we performed an overfitting experiment: training it on xx images selected from the xx dataset, with xxx steps of training on xx images.

The image generated by the checkpoint after xxx steps was similar to the original image, indicating that the convergence of the overfitting experiment was as good as we expected. Some of the generated images are as follows:

| a lively scene at a ski resort...   | a serene scene of a clear blue sky...  | a serene scene of a clear blue sky...| a serene scene of a clear blue sky... |
|-------------------------------------|----------------------------------------|--------------------------------------|---------------------------------------|
| <img src="" width=224>              | <img src="" width=224>                 | <img src="" width=224>               |                                       |
