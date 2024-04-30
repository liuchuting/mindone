import argparse
import datetime
import logging
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from opensora.models.autoencoder import SD_CONFIG, AutoencoderKL
from opensora.utils.model_utils import _check_cfgs_in_parser, str2bool

import mindspore as ms
from mindspore import ops

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from mindone.utils.logger import set_logger
from mindone.utils.seed import set_random_seed
from mindone.visualize.videos import save_videos

logger = logging.getLogger(__name__)

skip_vae = True


def init_env(args):
    # no parallel mode currently
    ms.set_context(mode=args.mode)  # needed for MS2.0
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.set_context(
        mode=args.mode,
        device_target=args.device_target,
        device_id=device_id,
    )
    return device_id


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="",
        type=str,
        help="path to load a config yaml file that describes the setting which will override the default arguments",
    )
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default="models/sd-vae-ft-ema.ckpt",
        help="VAE checkpoint file path which is used to load vae weight.",
    )
    parser.add_argument(
        "--sd_scale_factor", type=float, default=0.18215, help="VAE scale factor of Stable Diffusion model."
    )
    parser.add_argument(
        "--latent_path",
        type=str,
        nargs="+",
        default=["samples/denoised_latent_00.npz"],
        help="path(s) to save t5 embedding",
    )
    parser.add_argument(
        "--use_fp16",
        default=False,
        type=str2bool,
        help="whether use fp16.",
    )
    parser.add_argument(
        "--save_format",
        default="mp4",
        choices=["gif", "mp4"],
        type=str,
        help="video format for saving the sampling output, gif or mp4",
    )
    parser.add_argument("--fps", type=int, default=8, help="FPS in the saved video")
    # MS new args
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)")
    parser.add_argument("--seed", type=int, default=4, help="Inference seed")
    default_args = parser.parse_args()
    abs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ""))
    if default_args.config:
        logger.info(f"Overwrite default arguments with configuration file {default_args.config}")
        default_args.config = os.path.join(abs_path, default_args.config)
        with open(default_args.config, "r") as f:
            cfg = yaml.safe_load(f)
            _check_cfgs_in_parser(cfg, parser)
            parser.set_defaults(**cfg)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir = f"samples/{time_str}"
    os.makedirs(save_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)

    # 1. init env
    args = parse_args()
    init_env(args)
    set_random_seed(args.seed)

    # 2. model initiate and weight loading
    logger.info("vae init")
    vae = AutoencoderKL(
        SD_CONFIG,
        4,
        ckpt_path=args.vae_checkpoint,
        use_fp16=args.use_fp16,
    )
    vae = vae.set_train(False)
    for param in vae.get_parameters():  # freeze vae
        param.requires_grad = False

    def vae_decode(x):
        """
        Args:
            x: (b c h w), denoised latent
        Return:
            y: (b H W 3), batch of images, normalized to [0, 1]
        """
        b, c, h, w = x.shape

        y = vae.decode(x / args.sd_scale_factor)
        y = ops.clip_by_value((y + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)

        # (b 3 H W) -> (b H W 3)
        y = ops.transpose(y, (0, 2, 3, 1))

        return y

    def vae_decode_video(x):
        out = []
        for x_sample in x:
            # c t h w -> t c h w
            x_sample = x_sample.permute(1, 0, 2, 3)
            out.append(vae_decode(x_sample))
        out = ops.stack(out, axis=0)

        return out

    for lpath in args.latent_path:
        z = np.load(lpath)
        z = ms.Tensor(z)

        logger.info(f"Decoding for latent of shape {z.shape}, from {lpath}")
        vids = vae_decode_video(z)
        vids = vids.asnumpy()

        base = Path(os.path.basename(lpath)).with_suffix("")
        for i in range(vids.shape[0]):
            save_fp = f"{save_dir}/{base}-{i:03d}.{args.save_format}"
            save_videos(vids[i : i + 1], save_fp, fps=args.fps)
            logger.info(f"Video saved in {save_fp}")