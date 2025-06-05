#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动化迁移脚本：将 HuggingFace Transformers 中的 torch 实现迁移到 MindSpore（mindone）实现。
主要功能：
 1. 识别源代码中所有 torch.xx、torch.nn.xx、torch.Tensor 等接口；
 2. 按照映射规则，把它们替换为对应的 mindspore 接口，如 mint.xx 或 mint.nn.xx；
 3. 把变换后的文件写入到 mindone 对应目录结构下，并在日志中记录“找不到映射”的算子名称，方便后续人工补充。

使用示例：
    python migrate_torch_to_mindspore.py \
        --src_root /path/to/transformers/src/transformers \
        --dst_root /path/to/mindone/mindone/transformers

注意：
 - 请提前在脚本中完整补充 mint_map 和 mint_nn_map 字典；
 - 本脚本只做文本级别替换，无法保证迁移后完全能跑通，还需手动测试并调整细节。
"""

import os
import re
import argparse
import shutil

# -----------------------------------------------
# 1. 准备映射字典（请根据 MindSpore 文档补充完整）
# -----------------------------------------------

# 1.1 mint 算子级映射（mint.xx -> torch.xx）
#    例如：mint_map['add'] = 'add' 表示 torch.add -> mint.add
#    实际请参考：https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/mindspore.mint.html
mint_map = {
    "torch.arange": "mint.arange",
    "torch.ge": "mint.ge",
    "torch.bernoulli": "mint.bernoulli",
    "torch.isnan": "mint.isnan",
    "torch.bincount": "mint.bincount",
    "torch.clone": "mint.clone",
    "torch.eye": "mint.eye",
    "torch.einsum": "mint.einsum",
    "torch.empty": "mint.empty",
    "torch.empty_like": "mint.empty_like",
    "torch.full_like": "mint.full_like",
    "torch.linspace": "mint.linspace",
    "torch.ones": "mint.ones",
    "torch.ones_like": "mint.ones_like",
    "torch.randint": "mint.randint",
    "torch.randint_like": "mint.randint_like",
    "torch.randn": "mint.randn",
    "torch.randn_like": "mint.randn_like",
    "torch.randperm": "mint.randperm",
    "torch.zeros": "mint.zeros",
    "torch.zeros_like": "mint.zeros_like",
    "torch.cat": "mint.cat",
    "torch.chunk": "mint.chunk",
    "torch.concat": "mint.concat",
    "torch.count_nonzero": "mint.count_nonzero",
    "torch.gather": "mint.gather",
    "torch.index_add": "mint.index_add",
    "torch.index_select": "mint.index_select",
    "torch.masked_select": "mint.masked_select",
    "torch.permute": "mint.permute",
    "torch.reshape": "mint.reshape",
    "torch.scatter": "mint.scatter",
    "torch.scatter_add": "mint.scatter_add",
    "torch.split": "mint.split",
    "torch.narrow": "mint.narrow",
    "torch.nonzero": "mint.nonzero",
    "torch.tile": "mint.tile",
    "torch.tril": "mint.tril",
    "torch.select": "mint.select",
    "torch.squeeze": "mint.squeeze",
    "torch.stack": "mint.stack",
    "torch.swapaxes": "mint.swapaxes",
    "torch.transpose": "mint.transpose",
    "torch.triu": "mint.triu",
    "torch.unbind": "mint.unbind",
    "torch.unique_consecutive": "mint.unique_consecutive",
    "torch.unsqueeze": "mint.unsqueeze",
    "torch.where": "mint.where",
    "torch.multinomial": "mint.multinomial",
    "torch.normal": "mint.normal",
    "torch.rand_like": "mint.rand_like",
    "torch.rand": "mint.rand",
    "torch.abs": "mint.abs",
    "torch.add": "mint.add",
    "torch.addmv": "mint.addmv",
    "torch.acos": "mint.acos",
    "torch.acosh": "mint.acosh",
    "torch.arccos": "mint.arccos",
    "torch.arccosh": "mint.arccosh",
    "torch.arcsin": "mint.arcsin",
    "torch.arcsinh": "mint.arcsinh",
    "torch.arctan": "mint.arctan",
    "torch.arctan2": "mint.arctan2",
    "torch.arctanh": "mint.arctanh",
    "torch.asin": "mint.asin",
    "torch.asinh": "mint.asinh",
    "torch.atan": "mint.atan",
    "torch.atan2": "mint.atan2",
    "torch.atanh": "mint.atanh",
    "torch.bitwise_and": "mint.bitwise_and",
    "torch.bitwise_or": "mint.bitwise_or",
    "torch.bitwise_xor": "mint.bitwise_xor",
    "torch.ceil": "mint.ceil",
    "torch.clamp": "mint.clamp",
    "torch.cos": "mint.cos",
    "torch.cosh": "mint.cosh",
    "torch.cross": "mint.cross",
    "torch.diff": "mint.diff",
    "torch.div": "mint.div",
    "torch.divide": "mint.divide",
    "torch.erf": "mint.erf",
    "torch.erfc": "mint.erfc",
    "torch.erfinv": "mint.erfinv",
    "torch.exp": "mint.exp",
    "torch.exp2": "mint.exp2",
    "torch.expm1": "mint.expm1",
    "torch.fix": "mint.fix",
    "torch.float_power": "mint.float_power",
    "torch.floor": "mint.floor",
    "torch.fmod": "mint.fmod",
    "torch.frac": "mint.frac",
    "torch.lerp": "mint.lerp",
    "torch.log": "mint.log",
    "torch.log1p": "mint.log1p",
    "torch.log2": "mint.log2",
    "torch.log10": "mint.log10",
    "torch.logaddexp": "mint.logaddexp",
    "torch.logaddexp2": "mint.logaddexp2",
    "torch.logical_and": "mint.logical_and",
    "torch.logical_not": "mint.logical_not",
    "torch.logical_or": "mint.logical_or",
    "torch.logical_xor": "mint.logical_xor",
    "torch.mul": "mint.mul",
    "torch.mv": "mint.mv",
    "torch.nansum": "mint.nansum",
    "torch.nan_to_num": "mint.nan_to_num",
    "torch.neg": "mint.neg",
    "torch.negative": "mint.negative",
    "torch.pow": "mint.pow",
    "torch.polar": "mint.polar",
    "torch.ravel": "mint.ravel",
    "torch.reciprocal": "mint.reciprocal",
    "torch.remainder": "mint.remainder",
    "torch.roll": "mint.roll",
    "torch.round": "mint.round",
    "torch.rsqrt": "mint.rsqrt",
    "torch.sigmoid": "mint.sigmoid",
    "torch.sign": "mint.sign",
    "torch.sin": "mint.sin",
    "torch.sinc": "mint.sinc",
    "torch.sinh": "mint.sinh",
    "torch.softmax": "mint.softmax",
    "torch.sqrt": "mint.sqrt",
    "torch.square": "mint.square",
    "torch.sub": "mint.sub",
    "torch.t": "mint.t",
    "torch.tan": "mint.tan",
    "torch.tanh": "mint.tanh",
    "torch.trunc": "mint.trunc",
    "torch.xlogy": "mint.xlogy",
    "torch.amax": "mint.amax",
    "torch.amin": "mint.amin",
    "torch.argmax": "mint.argmax",
    "torch.argmin": "mint.argmin",
    "torch.argsort": "mint.argsort",
    "torch.all": "mint.all",
    "torch.any": "mint.any",
    "torch.cumprod": "mint.cumprod",
    "torch.histc": "mint.histc",
    "torch.logsumexp": "mint.logsumexp",
    "torch.max": "mint.max",
    "torch.mean": "mint.mean",
    "torch.median": "mint.median",
    "torch.min": "mint.min",
    "torch.norm": "mint.norm",
    "torch.prod": "mint.prod",
    "torch.sum": "mint.sum",
    "torch.std": "mint.std",
    "torch.std_mean": "mint.std_mean",
    "torch.unique": "mint.unique",
    "torch.var": "mint.var",
    "torch.var_mean": "mint.var_mean",
    "torch.allclose": "mint.allclose",
    "torch.argsort": "mint.argsort",
    "torch.eq": "mint.eq",
    "torch.equal": "mint.equal",
    "torch.greater": "mint.greater",
    "torch.greater_equal": "mint.greater_equal",
    "torch.gt": "mint.gt",
    "torch.isclose": "mint.isclose",
    "torch.isfinite": "mint.isfinite",
    "torch.isinf": "mint.isinf",
    "torch.isneginf": "mint.isneginf",
    "torch.le": "mint.le",
    "torch.less": "mint.less",
    "torch.less_equal": "mint.less_equal",
    "torch.lt": "mint.lt",
    "torch.maximum": "mint.maximum",
    "torch.minimum": "mint.minimum",
    "torch.ne": "mint.ne",
    "torch.not_equal": "mint.not_equal",
    "torch.topk": "mint.topk",
    "torch.sort": "mint.sort",
    "torch.addbmm": "mint.addbmm",
    "torch.addmm": "mint.addmm",
    "torch.baddbmm": "mint.baddbmm",
    "torch.bmm": "mint.bmm",
    "torch.dot": "mint.dot",
    "torch.inverse": "mint.inverse",
    "torch.matmul": "mint.matmul",
    "torch.meshgrid": "mint.meshgrid",
    "torch.mm": "mint.mm",
    "torch.outer": "mint.outer",
    "torch.trace": "mint.trace",
    "torch.broadcast_to": "mint.broadcast_to",
    "torch.cdist": "mint.cdist",
    "torch.cummax": "mint.cummax",
    "torch.cummin": "mint.cummin",
    "torch.cumsum": "mint.cumsum",
    "torch.diag": "mint.diag",
    "torch.flatten": "mint.flatten",
    "torch.flip": "mint.flip",
    "torch.repeat_interleave": "mint.repeat_interleave",
    "torch.searchsorted": "mint.searchsorted",
    "torch.tril": "mint.tril",
    "torch.triangular_solve": "mint.triangular_solve",
    "torch.clip": "mint.clamp",
    "torch.concatenate": "mint.cat",
}


mint_nn_map = {
    "torch.nn.Conv2d": "mint.nn.Conv2d",
    "torch.nn.Conv3d": "mint.nn.Conv3d",
    "torch.nn.ConvTranspose2d": "mint.nn.ConvTranspose2d",
    "torch.nn.Fold": "mint.nn.Fold",
    "torch.nn.Unfold": "mint.nn.Unfold",
    "torch.nn.BatchNorm1d": "mint.nn.BatchNorm1d",
    "torch.nn.BatchNorm2d": "mint.nn.BatchNorm2d",
    "torch.nn.BatchNorm3d": "mint.nn.BatchNorm3d",
    "torch.nn.GroupNorm": "mint.nn.GroupNorm",
    "torch.nn.LayerNorm": "mint.nn.LayerNorm",
    "torch.nn.SyncBatchNorm": "mint.nn.SyncBatchNorm",
    "torch.nn.ELU": "mint.nn.ELU",
    "torch.nn.GELU": "mint.nn.GELU",
    "torch.nn.GLU": "mint.nn.GLU",
    "torch.nn.Hardshrink": "mint.nn.Hardshrink",
    "torch.nn.Hardsigmoid": "mint.nn.Hardsigmoid",
    "torch.nn.Hardswish": "mint.nn.Hardswish",
    "torch.nn.LogSigmoid": "mint.nn.LogSigmoid",
    "torch.nn.LogSoftmax": "mint.nn.LogSoftmax",
    "torch.nn.Mish": "mint.nn.Mish",
    "torch.nn.PReLU": "mint.nn.PReLU",
    "torch.nn.ReLU": "mint.nn.ReLU",
    "torch.nn.ReLU6": "mint.nn.ReLU6",
    "torch.nn.SELU": "mint.nn.SELU",
    "torch.nn.SiLU": "mint.nn.SiLU",
    "torch.nn.Sigmoid": "mint.nn.Sigmoid",
    "torch.nn.Softmax": "mint.nn.Softmax",
    "torch.nn.Softshrink": "mint.nn.Softshrink",
    "torch.nn.Tanh": "mint.nn.Tanh",
    "torch.nn.Embedding": "mint.nn.Embedding",
    "torch.nn.Linear": "mint.nn.Linear",
    "torch.nn.Dropout": "mint.nn.Dropout",
    "torch.nn.Dropout2d": "mint.nn.Dropout2d",
    "torch.nn.AdaptiveAvgPool1d": "mint.nn.AdaptiveAvgPool1d",
    "torch.nn.AdaptiveAvgPool2d": "mint.nn.AdaptiveAvgPool2d",
    "torch.nn.AdaptiveAvgPool3d": "mint.nn.AdaptiveAvgPool3d",
    "torch.nn.AdaptiveMaxPool1d": "mint.nn.AdaptiveMaxPool1d",
    "torch.nn.AvgPool2d": "mint.nn.AvgPool2d",
    "torch.nn.AvgPool3d": "mint.nn.AvgPool3d",
    "torch.nn.MaxUnpool2d": "mint.nn.MaxUnpool2d",
    "torch.nn.ConstantPad1d": "mint.nn.ConstantPad1d",
    "torch.nn.ConstantPad2d": "mint.nn.ConstantPad2d",
    "torch.nn.ConstantPad3d": "mint.nn.ConstantPad3d",
    "torch.nn.ReflectionPad1d": "mint.nn.ReflectionPad1d",
    "torch.nn.ReflectionPad2d": "mint.nn.ReflectionPad2d",
    "torch.nn.ReflectionPad3d": "mint.nn.ReflectionPad3d",
    "torch.nn.ReplicationPad1d": "mint.nn.ReplicationPad1d",
    "torch.nn.ReplicationPad2d": "mint.nn.ReplicationPad2d",
    "torch.nn.ReplicationPad3d": "mint.nn.ReplicationPad3d",
    "torch.nn.ZeroPad1d": "mint.nn.ZeroPad1d",
    "torch.nn.ZeroPad2d": "mint.nn.ZeroPad2d",
    "torch.nn.ZeroPad3d": "mint.nn.ZeroPad3d",
    "torch.nn.BCELoss": "mint.nn.BCELoss",
    "torch.nn.BCEWithLogitsLoss": "mint.nn.BCEWithLogitsLoss",
    "torch.nn.CrossEntropyLoss": "mint.nn.CrossEntropyLoss",
    "torch.nn.KLDivLoss": "mint.nn.KLDivLoss",
    "torch.nn.L1Loss": "mint.nn.L1Loss",
    "torch.nn.MSELoss": "mint.nn.MSELoss",
    "torch.nn.NLLLoss": "mint.nn.NLLLoss",
    "torch.nn.SmoothL1Loss": "mint.nn.SmoothL1Loss",
    "torch.nn.PixelShuffle": "mint.nn.PixelShuffle",
    "torch.nn.Upsample": "mint.nn.Upsample",
    "torch.nn.Identity": "mint.nn.Identity",
    "torch.nn.functional.conv2d": "mint.nn.functional.conv2d",
    "torch.nn.functional.conv3d": "mint.nn.functional.conv3d",
    "torch.nn.functional.conv_transpose2d": "mint.nn.functional.conv_transpose2d",
    "torch.nn.functional.fold": "mint.nn.functional.fold",
    "torch.nn.functional.unfold": "mint.nn.functional.unfold",
    "torch.nn.functional.adaptive_avg_pool1d": "mint.nn.functional.adaptive_avg_pool1d",
    "torch.nn.functional.adaptive_avg_pool2d": "mint.nn.functional.adaptive_avg_pool2d",
    "torch.nn.functional.adaptive_avg_pool3d": "mint.nn.functional.adaptive_avg_pool3d",
    "torch.nn.functional.adaptive_max_pool1d": "mint.nn.functional.adaptive_max_pool1d",
    "torch.nn.functional.avg_pool1d": "mint.nn.functional.avg_pool1d",
    "torch.nn.functional.avg_pool2d": "mint.nn.functional.avg_pool2d",
    "torch.nn.functional.avg_pool3d": "mint.nn.functional.avg_pool3d",
    "torch.nn.functional.max_pool2d": "mint.nn.functional.max_pool2d",
    "torch.nn.functional.max_unpool2d": "mint.nn.functional.max_unpool2d",
    "torch.nn.functional.batch_norm": "mint.nn.functional.batch_norm",
    "torch.nn.functional.elu": "mint.nn.functional.elu",
    "torch.nn.functional.elu_": "mint.nn.functional.elu_",
    "torch.nn.functional.gelu": "mint.nn.functional.gelu",
    "torch.nn.functional.glu": "mint.nn.functional.glu",
    "torch.nn.functional.group_norm": "mint.nn.functional.group_norm",
    "torch.nn.functional.hardshrink": "mint.nn.functional.hardshrink",
    "torch.nn.functional.hardsigmoid": "mint.nn.functional.hardsigmoid",
    "torch.nn.functional.hardswish": "mint.nn.functional.hardswish",
    "torch.nn.functional.layer_norm": "mint.nn.functional.layer_norm",
    "torch.nn.functional.leaky_relu": "mint.nn.functional.leaky_relu",
    "torch.nn.functional.log_softmax": "mint.nn.functional.log_softmax",
    "torch.log_softmax": "mint.nn.functional.log_softmax",
    "torch.nn.functional.logsigmoid": "mint.nn.functional.logsigmoid",
    "torch.nn.functional.mish": "mint.nn.functional.mish",
    "torch.nn.functional.prelu": "mint.nn.functional.prelu",
    "torch.nn.functional.relu": "mint.nn.functional.relu",
    "torch.nn.functional.relu6": "mint.nn.functional.relu6",
    "torch.nn.functional.relu_": "mint.nn.functional.relu_",
    "torch.nn.functional.selu": "mint.nn.functional.selu",
    "torch.nn.functional.sigmoid": "mint.nn.functional.sigmoid",
    "torch.nn.functional.silu": "mint.nn.functional.silu",
    "torch.nn.functional.softmax": "mint.nn.functional.softmax",
    "torch.nn.functional.softplus": "mint.nn.functional.softplus",
    "torch.nn.functional.softshrink": "mint.nn.functional.softshrink",
    "torch.nn.functional.tanh": "mint.nn.functional.tanh",
    "torch.nn.functional.normalize": "mint.nn.functional.normalize",
    "torch.nn.functional.linear": "mint.nn.functional.linear",
    "torch.nn.functional.dropout": "mint.nn.functional.dropout",
    "torch.nn.functional.dropout2d": "mint.nn.functional.dropout2d",
    "torch.nn.functional.embedding": "mint.nn.functional.embedding",
    "torch.nn.functional.one_hot": "mint.nn.functional.one_hot",
    "torch.nn.functional.cross_entropy": "mint.nn.functional.cross_entropy",
    "torch.nn.functional.binary_cross_entrop": "mint.nn.functional.binary_cross_entropy",
    "torch.nn.functional.binary_cross_entrop": "mint.nn.functional.binary_cross_entropy_with_logits",
    "torch.nn.functional.kl_div": "mint.nn.functional.kl_div",
    "torch.nn.functional.l1_loss": "mint.nn.functional.l1_loss",
    "torch.nn.functional.mse_loss": "mint.nn.functional.mse_loss",
    "torch.nn.functional.nll_loss": "mint.nn.functional.nll_loss",
    "torch.nn.functional.smooth_l1_loss": "mint.nn.functional.smooth_l1_loss",
    "torch.nn.functional.interpolate": "mint.nn.functional.interpolate",
    "torch.nn.functional.grid_sample": "mint.nn.functional.grid_sample",
    "torch.nn.functional.pad": "mint.nn.functional.pad",
    "torch.nn.functional.pixel_shuffle": "mint.nn.functional.pixel_shuffle",
    "torch.nn.Module": "nn.Cell",
    "torch.nn.ModuleList": "nn.CellList",
    "torch.nn.Flatten": "nn.Flatten",
    "torch.nn.Parameter": "ms.Parameter",
}

ops_map = {
    "torch.addcmul": "ops.addcmul",
    "torch.argwhere": "ops.argwhere",
    "torch.bucketize": "ops.bucketize",
    "torch.conj": "ops.conj",
    "torch.cosine_similarity": "ops.cosine_similarity",
    "torch.deg2rad": "ops.deg2rad",
    "torch.hann_window": "ops.hann_window",
    "torch.hstack": "ops.hstack",
    "torch.masked_fill": "ops.masked_fill",
    "torch.multiply": "ops.multiply",
    "torch.numel": "ops.numel",
    "torch.range": "ops.range",
    "torch.relu": "ops.relu",
    "torch.nn.functional.ctc_loss": "ops.ctc_loss",
    "torch.nn.functional.gumbel_softmax": "ops.gumbel_softmax",
    "torch.full": "ops.full",
    "torch.fill": "ops.fill",
}

nn_map = {
    "torch.nn.CTCLoss": "nn.CTCLoss",
}

dtype_map = {
    "torch.Tensor": "ms.Tensor",
    "torch.tensor": "ms.Tensor",
    "torch.ByteTensor": "ms.Tensor",
    "torch.IntTensor": "ms.Tensor",
    "torch.FloatTensor": "ms.Tensor",
    "torch.LongTensor": "ms.Tensor", 
    "torch.BoolTensor":"ms.Tensor", 
    "torch.float": "ms.float32",
    "torch.double": "ms.float64",
    "torch.float32": "ms.float32",
    "torch.float64": "ms.float64",
    "torch.float16": "ms.float16",
    "torch.bfloat16": "ms.bfloat16",
    "torch.int8": "ms.int8",
    "torch.uint8": "ms.uint8",
    "torch.int16": "ms.int16",
    "torch.int": "ms.int32",
    "torch.int32": "ms.int32",
    "torch.int64": "ms.int64",
    "torch.long": "ms.int64",
    "torch.bool": "ms.bool_",
    "torch.dtype": "ms.dtype",
    "torch.Generator": "ms.Generator",
    "torch.complex64": "ms.complex64",
    "torch.no_grad": "ms._no_grad",
    "torch.version": "ms.version",
    "torch.vmap": "ms.vmap"
    }

function_map = {
    "finfo": "dtype_to_min",
}



unmapped_ops = set()

def parse_args():
    parser = argparse.ArgumentParser(
        description="将 transformers 中的 torch 代码迁移到 mindspore（mindone）目录，并做接口替换")
    parser.add_argument(
        "--src_root", type=str, required=False, default="../../transformers/src/transformers/models/fsmt",
        help="源代码目录（transformers/src/transformers）")
    parser.add_argument(
        "--dst_root", type=str, required=False, default="../../mindone/mindone/transformers/models/fsmt",
        help="目标代码目录（mindone/mindone/transformers）")
    return parser.parse_args()


def ensure_dst_path(src_file: str, src_root: str, dst_root: str) -> str:
    """
    根据 src_root 和 dst_root 生成 dst_file 路径，同时创建必要目录。
    例如 src_root=/xxx/transformers/src/transformers，src_file=/xxx/transformers/src/transformers/models/m2m_100/modeling.py
         dst_root=/zzz/mindone/mindone/transformers，则会生成
         dst_file=/zzz/mindone/mindone/transformers/models/m2m_100/modeling.py
    """
    rel_path = os.path.relpath(src_file, start=src_root)
    dst_file = os.path.join(dst_root, rel_path)
    dst_dir = os.path.dirname(dst_file)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
    return dst_file


def replace_import(content: str) -> str:
    # 匹配from torch import xx -> 将所有xx替换成torch.xx
    torch_imports = re.findall(r"^\s*from\s+torch\s+import\s+([^\n#]+)", content, flags=re.MULTILINE)
    for import_line in torch_imports:
        items = [item.strip().split(" as ")[0] for item in import_line.split(",")]
        # print(items)
        for item in items:
            pattern = r"(?<!torch\.)\b" + re.escape(item) + r"\b"
            content = re.sub(pattern, f"torch.{item}", content)
    content = re.sub(r"^\s*import\s+torch\b.*$", "import mindspore as ms", content, flags=re.MULTILINE)


    # 同样处理 torch.nn
    torch_nn_imports = re.findall(r"^\s*from\s+torch\.nn\s+import\s+([^\n#]+)", content, flags=re.MULTILINE)
    for import_line in torch_nn_imports:
        items = [item.strip().split(" import ")[0] for item in import_line.split(",")]
        for item in items:
            # pattern = r"(?<!from torch\.nn import )" + re.escape(item) + r"\b"
            pattern = r"(?<!from torch\.nn import )(?<!torch\.nn\.)(?<!nn\.)" + re.escape(item) + r"\b"
            content = re.sub(pattern, f"torch.nn.{item}", content)


    content = re.sub(r"from torch\.nn import ([A-Za-z0-9_]+)\b", lambda m: f"from mindspore.mint.nn import {m.group(1)}", content)
    content = re.sub(r"^\s*from\s+torch\b(.*)$", "from mindspore import mint, nn", content, flags=re.MULTILINE)

    return content


def replace_content(content: str) -> str:
    content = re.sub(r"^\s*def\s+forward\s*\(", "    def construct(", content, flags=re.MULTILINE)
    content = re.sub(r"\bsuper\(\)\.forward\(", "super().construct(", content, flags=re.MULTILINE)
    content = re.sub(r"\bPyTorch\b", "MindSpore", content, flags=re.MULTILINE)
    content = re.sub(r"torch\.finfo\(([^)]+)\)\.min", r"dtype_to_min(\1)", content, flags=re.MULTILINE)
    content = re.sub(r"torch\.finfo\(([^)]+)\)\.max", r"dtype_to_max(\1)", content, flags=re.MULTILINE)

    def replace_mint_nn_layer(match):
        full = match.group(0)          # torch.nn.Linear
        if full in mint_nn_map:
            return f"{mint_nn_map[full]}"
        else:
            unmapped_ops.add(full)
            return full
    content = re.sub(r"\btorch\.nn\.functional\.([A-Za-z0-9_]+)\b", replace_mint_nn_layer, content)
    content = re.sub(r"\btorch\.nn\.([A-Za-z0-9_]+)\b", replace_mint_nn_layer, content)


    def replace_mint_op(match):
        opname = match.group(0).strip("(")
        if opname in mint_map:
            return f"{mint_map[opname]}("
        else:
            unmapped_ops.add(opname)
            # print(11, unmapped_ops)
            return match.group(0)
    content = re.sub(r"\btorch\.([A-Za-z0-9_]+)\s*\(", replace_mint_op, content)
    content = re.sub(r"\btorch\.([A-Za-z0-9_]+)\b", replace_mint_op, content)
    # 可以匹配换行
    content = re.sub(r"\.to\s*\(\s*device\s*=\s*[A-Za-z0-9_\.]+\.device\s*\)", "", content, flags=re.DOTALL)
    content = re.sub(r"device\s*=\s*[A-Za-z0-9_\.]+\.device\s*,?", "", content)


    def replace_ops_op(match):
        opname = match.group(0).strip("(")
        if opname in ops_map:
            if opname in unmapped_ops:
                unmapped_ops.discard(opname)
            return f"{ops_map[opname]}("
        else:
            unmapped_ops.add(opname)
            return match.group(0)
    content = re.sub(r"\btorch\.([A-Za-z0-9_]+)\s*\(", replace_ops_op, content)


    def replace_nn_op(match):
        opname = match.group(0).strip("(")
        if opname in nn_map:
            if opname in unmapped_ops:
                unmapped_ops.discard(opname)
            return f"{nn_map[opname]}("
        else:
            unmapped_ops.add(opname)
            return match.group(0)
    content = re.sub(r"\btorch\.nn\.([A-Za-z0-9_]+)\s*\(", replace_nn_op, content)


    def replace_dtype(match):
        full = match.group(0)          # torch.nn.Linear
        if full in dtype_map:
            if full in unmapped_ops:
                unmapped_ops.discard(full)
            return f"{dtype_map[full]}"
        else:
            unmapped_ops.add(full)
            return full
    content = re.sub(r"\btorch\.([A-Za-z0-9_]+)\b", replace_dtype, content)
    # -----------------------------------------------
    # 2.9 size() -> shape； size(x) -> shape[x]
    # -----------------------------------------------
    # size() -> shape
    content = re.sub(r"\.size\s*\(\s*\)", ".shape", content)
    # size(x) -> shape[x]
    content = re.sub(r"\.size\s*\(\s*([0-9]+)\s*\)", r".shape[\1]", content)

    return content



def process_file(src_file: str, dst_file: str):
    """
    读取 src_file，对内容做替换，然后写入 dst_file。
    """
    try:
        with open(src_file, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        # 如果有非 utf-8 编码的文件，可以先跳过或按需求调整
        print(f"[WARN] 无法以 utf-8 解码文件：{src_file}，跳过")
        return

    new_content = replace_import(content)
    new_content = replace_content(new_content)
    
    with open(dst_file, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"[INFO] 迁移并替换：{src_file} -> {dst_file}")


def main():
    args = parse_args()
    src_root = os.path.abspath(args.src_root)
    dst_root = os.path.abspath(args.dst_root)

    # 如果目标目录不存在，先创建
    if not os.path.exists(dst_root):
        os.makedirs(dst_root, exist_ok=True)

    # 遍历 src_root 下所有 .py 文件
    for root, dirs, files in os.walk(src_root):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            src_file = os.path.join(root, fname)
            dst_file = ensure_dst_path(src_file, src_root, dst_root)
            process_file(src_file, dst_file)

    # 最后，把 unmapped_ops 和 unmapped_mint_nn_layers 写到日志文件：
    # log_path = os.path.join(dst_root, "unmapped_ops.log")
    # with open(log_path, "w", encoding="utf-8") as logf:
    print("\n# 以下torch接口未找到对应mindspore映射，请手动修改: ")
    for op in sorted(unmapped_ops):
        print(op)
    # print(unmapped_dtype)
    # print("\n# 以下层（nn layer）未找到对应的 mint.nn 映射，请人工补充到 mint_nn_map: ")
    # for op in sorted(unmapped_mint_nn_layers):
    #     print(op)
    # print("\n# 以下算子（ops）未找到对应的 mint 映射，请人工补充到 ops_map: ")
    # for op in sorted(unmapped_ops):
    #     print(op)
    # print("\n# 以下torch.xxx未找到对应的 mint 映射，请人工补充到 nn_map: ")
    # for op in sorted(unmapped_nn_layers):
    #     print(op)
    # print("\n# 以下算子类型未找到对应的 dtype 映射，请人工补充到 dtype_map: ")
    # for op in sorted(unmapped_dtype):
    #     print(op)



if __name__ == "__main__":
    main()
