import os
import argparse
import shutil
import ast
import astor
from typing import Dict, Set
import black
import libcst as cst
import libcst.matchers as m
# 映射关系字典（用户可根据需要补充）
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
    "torch.log_softmax": "mint.nn.functional.log_softmax",
}


mint_nn_map = {
    "nn.Conv2d": "mint.nn.Conv2d",
    "nn.Conv3d": "mint.nn.Conv3d",
    "nn.ConvTranspose2d": "mint.nn.ConvTranspose2d",
    "nn.Fold": "mint.nn.Fold",
    "nn.Unfold": "mint.nn.Unfold",
    "nn.BatchNorm1d": "mint.nn.BatchNorm1d",
    "nn.BatchNorm2d": "mint.nn.BatchNorm2d",
    "nn.BatchNorm3d": "mint.nn.BatchNorm3d",
    "nn.GroupNorm": "mint.nn.GroupNorm",
    "nn.LayerNorm": "mint.nn.LayerNorm",
    "nn.SyncBatchNorm": "mint.nn.SyncBatchNorm",
    "nn.ELU": "mint.nn.ELU",
    "nn.GELU": "mint.nn.GELU",
    "nn.GLU": "mint.nn.GLU",
    "nn.Hardshrink": "mint.nn.Hardshrink",
    "nn.Hardsigmoid": "mint.nn.Hardsigmoid",
    "nn.Hardswish": "mint.nn.Hardswish",
    "nn.LogSigmoid": "mint.nn.LogSigmoid",
    "nn.LogSoftmax": "mint.nn.LogSoftmax",
    "nn.Mish": "mint.nn.Mish",
    "nn.PReLU": "mint.nn.PReLU",
    "nn.ReLU": "mint.nn.ReLU",
    "nn.ReLU6": "mint.nn.ReLU6",
    "nn.SELU": "mint.nn.SELU",
    "nn.SiLU": "mint.nn.SiLU",
    "nn.Sigmoid": "mint.nn.Sigmoid",
    "nn.Softmax": "mint.nn.Softmax",
    "nn.Softshrink": "mint.nn.Softshrink",
    "nn.Tanh": "mint.nn.Tanh",
    "nn.Embedding": "mint.nn.Embedding",
    "nn.Linear": "mint.nn.Linear",
    "nn.Dropout": "mint.nn.Dropout",
    "nn.Dropout2d": "mint.nn.Dropout2d",
    "nn.AdaptiveAvgPool1d": "mint.nn.AdaptiveAvgPool1d",
    "nn.AdaptiveAvgPool2d": "mint.nn.AdaptiveAvgPool2d",
    "nn.AdaptiveAvgPool3d": "mint.nn.AdaptiveAvgPool3d",
    "nn.AdaptiveMaxPool1d": "mint.nn.AdaptiveMaxPool1d",
    "nn.AvgPool2d": "mint.nn.AvgPool2d",
    "nn.AvgPool3d": "mint.nn.AvgPool3d",
    "nn.MaxUnpool2d": "mint.nn.MaxUnpool2d",
    "nn.ConstantPad1d": "mint.nn.ConstantPad1d",
    "nn.ConstantPad2d": "mint.nn.ConstantPad2d",
    "nn.ConstantPad3d": "mint.nn.ConstantPad3d",
    "nn.ReflectionPad1d": "mint.nn.ReflectionPad1d",
    "nn.ReflectionPad2d": "mint.nn.ReflectionPad2d",
    "nn.ReflectionPad3d": "mint.nn.ReflectionPad3d",
    "nn.ReplicationPad1d": "mint.nn.ReplicationPad1d",
    "nn.ReplicationPad2d": "mint.nn.ReplicationPad2d",
    "nn.ReplicationPad3d": "mint.nn.ReplicationPad3d",
    "nn.ZeroPad1d": "mint.nn.ZeroPad1d",
    "nn.ZeroPad2d": "mint.nn.ZeroPad2d",
    "nn.ZeroPad3d": "mint.nn.ZeroPad3d",
    "nn.BCELoss": "mint.nn.BCELoss",
    "nn.BCEWithLogitsLoss": "mint.nn.BCEWithLogitsLoss",
    "nn.CrossEntropyLoss": "mint.nn.CrossEntropyLoss",
    "nn.KLDivLoss": "mint.nn.KLDivLoss",
    "nn.L1Loss": "mint.nn.L1Loss",
    "nn.MSELoss": "mint.nn.MSELoss",
    "nn.NLLLoss": "mint.nn.NLLLoss",
    "nn.SmoothL1Loss": "mint.nn.SmoothL1Loss",
    "nn.PixelShuffle": "mint.nn.PixelShuffle",
    "nn.Upsample": "mint.nn.Upsample",
    "nn.Identity": "mint.nn.Identity",
    "nn.functional.conv2d": "mint.nn.functional.conv2d",
    "nn.functional.conv3d": "mint.nn.functional.conv3d",
    "nn.functional.conv_transpose2d": "mint.nn.functional.conv_transpose2d",
    "nn.functional.fold": "mint.nn.functional.fold",
    "nn.functional.unfold": "mint.nn.functional.unfold",
    "nn.functional.adaptive_avg_pool1d": "mint.nn.functional.adaptive_avg_pool1d",
    "nn.functional.adaptive_avg_pool2d": "mint.nn.functional.adaptive_avg_pool2d",
    "nn.functional.adaptive_avg_pool3d": "mint.nn.functional.adaptive_avg_pool3d",
    "nn.functional.adaptive_max_pool1d": "mint.nn.functional.adaptive_max_pool1d",
    "nn.functional.avg_pool1d": "mint.nn.functional.avg_pool1d",
    "nn.functional.avg_pool2d": "mint.nn.functional.avg_pool2d",
    "nn.functional.avg_pool3d": "mint.nn.functional.avg_pool3d",
    "nn.functional.max_pool2d": "mint.nn.functional.max_pool2d",
    "nn.functional.max_unpool2d": "mint.nn.functional.max_unpool2d",
    "nn.functional.batch_norm": "mint.nn.functional.batch_norm",
    "nn.functional.elu": "mint.nn.functional.elu",
    "nn.functional.elu_": "mint.nn.functional.elu_",
    "nn.functional.gelu": "mint.nn.functional.gelu",
    "nn.functional.glu": "mint.nn.functional.glu",
    "nn.functional.group_norm": "mint.nn.functional.group_norm",
    "nn.functional.hardshrink": "mint.nn.functional.hardshrink",
    "nn.functional.hardsigmoid": "mint.nn.functional.hardsigmoid",
    "nn.functional.hardswish": "mint.nn.functional.hardswish",
    "nn.functional.layer_norm": "mint.nn.functional.layer_norm",
    "nn.functional.leaky_relu": "mint.nn.functional.leaky_relu",
    "nn.functional.log_softmax": "mint.nn.functional.log_softmax",
    "nn.functional.logsigmoid": "mint.nn.functional.logsigmoid",
    "nn.functional.mish": "mint.nn.functional.mish",
    "nn.functional.prelu": "mint.nn.functional.prelu",
    "nn.functional.relu": "mint.nn.functional.relu",
    "nn.functional.relu6": "mint.nn.functional.relu6",
    "nn.functional.relu_": "mint.nn.functional.relu_",
    "nn.functional.selu": "mint.nn.functional.selu",
    "nn.functional.sigmoid": "mint.nn.functional.sigmoid",
    "nn.functional.silu": "mint.nn.functional.silu",
    "nn.functional.softmax": "mint.nn.functional.softmax",
    "nn.functional.softplus": "mint.nn.functional.softplus",
    "nn.functional.softshrink": "mint.nn.functional.softshrink",
    "nn.functional.tanh": "mint.nn.functional.tanh",
    "nn.functional.normalize": "mint.nn.functional.normalize",
    "nn.functional.linear": "mint.nn.functional.linear",
    "nn.functional.dropout": "mint.nn.functional.dropout",
    "nn.functional.dropout2d": "mint.nn.functional.dropout2d",
    "nn.functional.embedding": "mint.nn.functional.embedding",
    "nn.functional.one_hot": "mint.nn.functional.one_hot",
    "nn.functional.cross_entropy": "mint.nn.functional.cross_entropy",
    "nn.functional.binary_cross_entrop": "mint.nn.functional.binary_cross_entropy",
    "nn.functional.binary_cross_entrop": "mint.nn.functional.binary_cross_entropy_with_logits",
    "nn.functional.kl_div": "mint.nn.functional.kl_div",
    "nn.functional.l1_loss": "mint.nn.functional.l1_loss",
    "nn.functional.mse_loss": "mint.nn.functional.mse_loss",
    "nn.functional.nll_loss": "mint.nn.functional.nll_loss",
    "nn.functional.smooth_l1_loss": "mint.nn.functional.smooth_l1_loss",
    "nn.functional.interpolate": "mint.nn.functional.interpolate",
    "nn.functional.grid_sample": "mint.nn.functional.grid_sample",
    "nn.functional.pad": "mint.nn.functional.pad",
    "nn.functional.pixel_shuffle": "mint.nn.functional.pixel_shuffle",
    "nn.Module": "nn.Cell",
    "nn.Sequential": "nn.SequentialCell",
    "nn.ModuleList": "nn.CellList",
    "nn.Flatten": "nn.Flatten",
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

t2m_map = {
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
    "torch.vmap": "ms.vmap",
    "torch.nn.Parameter": "ms.Parameter",
    "torch.from_numpy": "ms.Tensor.from_numpy",
    }



class TorchToMindsporeCST(cst.CSTTransformer):
    def __init__(self):
        self.unmapped: Set[str] = set()
        self.need_mint_import = False
        self.need_ms_import = False
        self.need_ops_import = False

    def leave_Module(self, original_node, updated_node):
        # 插入必要导入
        insert_lines = []
        if self.need_ms_import:
            insert_lines.append(cst.SimpleStatementLine([cst.Import(names=[cst.ImportAlias(name=cst.Name("mindspore"), asname=cst.AsName(name=cst.Name("ms")))])]))
        if self.need_ops_import:
            insert_lines.append(cst.SimpleStatementLine([cst.ImportFrom(module=cst.Name("mindspore"), names=[cst.ImportAlias(name=cst.Name("ops"))])]))
        if self.need_mint_import:
            insert_lines.append(cst.SimpleStatementLine([cst.ImportFrom(module=cst.Name("mindspore"), names=[cst.ImportAlias(name=cst.Name("mint")), cst.ImportAlias(name=cst.Name("nn"))])]))
        if insert_lines:
            return updated_node.with_changes(body=insert_lines + list(updated_node.body))
        return updated_node

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        if original_node.name.value == "forward":
            return updated_node.with_changes(name=cst.Name("construct"))
        return updated_node

    def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> cst.BaseExpression:
        full = self._get_fullname(updated_node)
        mapped = self._map_fullname(full)
        if mapped:
            return self._str_to_attr(mapped)
        return updated_node

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.BaseExpression:
        if m.matches(updated_node.func, m.Attribute(attr=m.Name("size"))):
            target = updated_node.func.value
            if not updated_node.args:
                return cst.Attribute(value=target, attr=cst.Name("shape"))
            if len(updated_node.args) == 1:
                return cst.Subscript(
                    value=cst.Attribute(value=target, attr=cst.Name("shape")),
                    slice=[cst.SubscriptElement(slice=updated_node.args[0].value)]
                )
        # 处理 super().forward() → super().construct()
        if m.matches(updated_node.func, m.Attribute(attr=m.Name("forward"))):
            if m.matches(updated_node.func.value, m.Call(func=m.Name("super"))):
                return updated_node.with_changes(func=cst.Attribute(value=updated_node.func.value, attr=cst.Name("construct")))

        new_args = [
            arg for arg in updated_node.args
            if not (arg.keyword and arg.keyword.value == "device")
            and not m.matches(arg.value, m.Attribute(attr=m.Name("device")))
        ]
        return updated_node.with_changes(args=new_args)
        # return updated_node

    def leave_Import(self, original_node: cst.Import, updated_node: cst.Import) -> cst.BaseStatement:
        new_names = []
        for alias in updated_node.names:
            if alias.name.value == "torch":
                self.need_ms_import = True
                new_alias = alias.with_changes(name=cst.Name("mindspore"))
                new_names.append(new_alias)
            else:
                new_names.append(alias)
        return updated_node.with_changes(names=new_names)

    def leave_ImportFrom(self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom) -> cst.BaseStatement:
        if updated_node.module is None:
            return updated_node

        module_str = self._get_fullname(updated_node.module)
        if not module_str.startswith("torch"):
            return updated_node

        new_module_str = module_str.replace("torch", "mindspore", 1)
        if new_module_str.startswith("mindspore.mint"):
            self.need_mint_import = True
        if new_module_str.startswith("mindspore.ops"):
            self.need_ops_import = True

        new_module_expr = self._str_to_attr(new_module_str)
        return updated_node.with_changes(module=new_module_expr)


    def _map_fullname(self, name: str) -> str | None:
        if name in mint_nn_map:
            self.need_mint_import = True
            return mint_nn_map[name]
        elif name.strip("torch.") in mint_nn_map:
            self.need_mint_import = True
            return mint_nn_map[name.strip("torch.")]
        elif name in mint_map:
            self.need_mint_import = True
            return mint_map[name]
        elif name in t2m_map:
            self.need_ms_import = True
            return t2m_map[name]
        elif name in ops_map:
            self.need_ops_import = True
            return ops_map[name]
        elif name.startswith("torch"):
            self.unmapped.add(name)
        return None

    def _get_fullname(self, node: cst.CSTNode) -> str:
        """递归解析 Attribute/Name 组成的链，避免依赖 .code 属性。"""
        if isinstance(node, cst.Name):
            return node.value
        if isinstance(node, cst.Attribute):
            left = self._get_fullname(node.value)
            right = self._get_fullname(node.attr)
            return f"{left}.{right}" if left else right
        return ""

    def _str_to_attr(self, dotted: str) -> cst.BaseExpression:
        parts = dotted.split(".")
        expr: cst.BaseExpression = cst.Name(parts[0])
        for part in parts[1:]:
            expr = cst.Attribute(value=expr, attr=cst.Name(part))
        return expr


def convert_file(path: str, transformer):
    try:
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
    except Exception as e:
        print(f"[ERROR] 处理失败 {path}: {e}")

    try:
        tree = cst.parse_module(source)
        new_tree = tree.visit(transformer)

        with open(path, "w", encoding="utf-8") as f:
            f.write(new_tree.code)

    except Exception as e:
        print(f"[ERROR] 处理失败1 {path}: {e}")


def copy_and_convert(src_root: str, dst_root: str):
    transformer = TorchToMindsporeCST()
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    shutil.copytree(src_root, dst_root)

    for dirpath, _, filenames in os.walk(dst_root):
        for filename in filenames:
            if filename.endswith(".py"):
                filepath = os.path.join(dirpath, filename)
                convert_file(filepath, transformer)
    if transformer.unmapped:
        print("\n以下接口未转换成功，需手动处理：")
        for name in sorted(transformer.unmapped):
            print(" -", name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str, required=True)
    parser.add_argument("--dst_root", type=str, required=True)
    args = parser.parse_args()

    copy_and_convert(args.src_root, args.dst_root)
