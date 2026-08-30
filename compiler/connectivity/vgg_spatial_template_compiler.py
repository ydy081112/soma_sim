#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VGG -> Spatial Pattern Template / Source-Major Weight 编译器
==============================================================

目标
----
把训练好的 PyTorch VGG 网络转换为一种适合 spike-driven 仿真的紧凑表示：

1. 不展开 neuron-to-neuron connection；
2. 对每个 Conv2d：
   - 编译 SpatialPlan：每个输入空间位置只保存 pattern_id + dst_base；
   - 编译 SpatialPatternTemplate：保存相对目标位置和 kernel offset；
   - 权重由 PyTorch 的 [Cout, Cin, Kh, Kw]
     重排为 [Cin, Kh, Kw, Cout]，使一个输入 spike 对所有 Cout 的权重连续；
3. 对每个 Linear：
   - 权重由 [Cout, Cin] 重排为 [Cin, Cout]；
4. 保存 manifest + 每层数组；
5. 提供 Python 参考 runtime，可直接用单个 spike 驱动编译后的 Conv；
6. 提供逐 Conv 层与 PyTorch F.conv2d 的正确性验证。

推荐的 simulator 神经元布局
--------------------------
Conv 输出状态建议按 [Hout * Wout, Cout] 保存，也就是逻辑上的 NHWC / spatial-major：

    neuron_id = ((y * Wout + x) * Cout + cout)

这样一个 spike 触发某个 kernel position 后，所有 cout 的 destination state 是连续的，
对应的 weight W[cin, ky, kx, :] 也是连续的，非常适合 SIMD / 向量化。

使用示例
--------
1) 标准 torchvision VGG16 checkpoint：

    python vgg_spatial_template_compiler.py \
        --arch vgg16 \
        --checkpoint ./vgg16_trained.pth \
        --input-shape 1 3 224 224 \
        --output-dir ./compiled_vgg16 \
        --verify

2) checkpoint 是 {'state_dict': ...} / {'model_state_dict': ...} 都支持。

3) 如果是自定义 VGG 类：
   先在你自己的 Python 文件中提供一个无参数 factory，例如：

       # my_vgg.py
       def build_model():
           model = MyVGG(...)
           return model

   然后：

    python vgg_spatial_template_compiler.py \
        --factory my_vgg:build_model \
        --checkpoint ./best.pth \
        --input-shape 1 3 32 32 \
        --output-dir ./compiled_my_vgg \
        --verify

注意
----
- 当前 SpatialPatternTemplate 对标准 Conv2d groups=1 完整支持，正好覆盖普通 VGG。
- 如果未来要支持 depthwise/group convolution，可在相同框架下增加 group-local Cout block。
- checkpoint 必须来自可信来源；完整 nn.Module pickle 加载需显式 --allow-pickle。
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import shutil
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 基础工具
# ============================================================


def _pair(v: Any) -> Tuple[int, int]:
    """把 int 或长度为 2 的 tuple/list 统一成 (h, w)。"""
    if isinstance(v, (tuple, list)):
        if len(v) != 2:
            raise ValueError(f"期望长度为 2，实际得到: {v}")
        return int(v[0]), int(v[1])
    return int(v), int(v)


def tensor_shape(x: Any) -> Optional[List[int]]:
    """提取 hook 输入/输出中的第一个 Tensor shape。"""
    if isinstance(x, torch.Tensor):
        return list(x.shape)
    if isinstance(x, (tuple, list)):
        for item in x:
            s = tensor_shape(item)
            if s is not None:
                return s
    if isinstance(x, dict):
        for item in x.values():
            s = tensor_shape(item)
            if s is not None:
                return s
    return None


def to_numpy_f32(t: torch.Tensor) -> np.ndarray:
    """Tensor -> CPU contiguous float32 numpy。"""
    return t.detach().cpu().contiguous().to(torch.float32).numpy()


def save_array(path: Path, arr: np.ndarray, save_format: str) -> Dict[str, Any]:
    """
    保存数组，并返回写入 manifest 的描述。

    npy:
        Python 调试方便，可直接 np.load(..., mmap_mode='r')。
    raw:
        直接裸二进制，C/C++ 最容易映射；shape/dtype 写在 manifest。
    """
    arr = np.ascontiguousarray(arr)

    if save_format == "npy":
        out = path.with_suffix(".npy")
        np.save(out, arr, allow_pickle=False)
        return {
            "file": out.name,
            "format": "npy",
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "nbytes": int(arr.nbytes),
        }

    if save_format == "raw":
        out = path.with_suffix(".bin")
        arr.tofile(out)
        return {
            "file": out.name,
            "format": "raw",
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "nbytes": int(arr.nbytes),
        }

    raise ValueError(f"未知 save_format: {save_format}")


def load_saved_array(root: Path, desc: Dict[str, Any], mmap: bool = True) -> np.ndarray:
    """Python 参考 runtime 使用的数组加载器。"""
    p = root / desc["file"]
    shape = tuple(int(x) for x in desc["shape"])
    dtype = np.dtype(desc["dtype"])

    if desc["format"] == "npy":
        return np.load(p, mmap_mode="r" if mmap else None, allow_pickle=False)

    if desc["format"] == "raw":
        if mmap:
            return np.memmap(p, mode="r", dtype=dtype, shape=shape)
        return np.fromfile(p, dtype=dtype).reshape(shape)

    raise ValueError(f"不支持的数组格式: {desc['format']}")


# ============================================================
# Checkpoint / 模型加载
# ============================================================


def extract_state_dict(obj: Any) -> OrderedDict:
    """
    从常见 checkpoint 结构中提取 state_dict。

    支持：
      - 直接保存的 state_dict
      - {'state_dict': state_dict}
      - {'model_state_dict': state_dict}
      - {'model': state_dict}
      - 完整 nn.Module（仅 --allow-pickle 时可能读到）
    """
    if isinstance(obj, nn.Module):
        return OrderedDict(obj.state_dict())

    if isinstance(obj, (dict, OrderedDict)):
        # 最常见的包装字段
        for key in ("state_dict", "model_state_dict", "model", "net", "network"):
            if key in obj:
                value = obj[key]
                if isinstance(value, nn.Module):
                    return OrderedDict(value.state_dict())
                if isinstance(value, (dict, OrderedDict)):
                    # 简单判断是否像参数字典
                    tensor_values = [v for v in value.values() if isinstance(v, torch.Tensor)]
                    if tensor_values:
                        return OrderedDict(value)

        # checkpoint 本身就是 state_dict
        tensor_values = [v for v in obj.values() if isinstance(v, torch.Tensor)]
        if tensor_values:
            return OrderedDict(obj)

    raise ValueError(
        "无法从 checkpoint 中找到 state_dict。"
        "支持 raw state_dict / state_dict / model_state_dict / model / nn.Module。"
    )


def strip_prefix(sd: OrderedDict, prefix: str) -> OrderedDict:
    """如果 key 以 prefix 开头，则去掉 prefix。"""
    out = OrderedDict()
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out


def normalize_state_dict_keys(sd: OrderedDict, model: nn.Module) -> OrderedDict:
    """
    自动处理 DataParallel / 外层 model. / net. 等前缀。
    选择与目标 model.state_dict() key 重合数最多的一组。
    """
    target_keys = set(model.state_dict().keys())

    candidates: List[OrderedDict] = [sd]
    prefixes = ("module.", "model.", "net.", "network.")

    # 单前缀
    for p in prefixes:
        candidates.append(strip_prefix(sd, p))

    # 常见组合，例如 module.model.
    for p1 in prefixes:
        tmp = strip_prefix(sd, p1)
        for p2 in prefixes:
            candidates.append(strip_prefix(tmp, p2))

    best = max(candidates, key=lambda x: len(set(x.keys()) & target_keys))
    overlap = len(set(best.keys()) & target_keys)

    if overlap == 0:
        raise RuntimeError(
            "checkpoint 参数名与模型完全无法匹配。\n"
            f"checkpoint 示例 key: {list(sd.keys())[:8]}\n"
            f"model 示例 key: {list(model.state_dict().keys())[:8]}"
        )

    return best


def infer_num_classes_from_state_dict(sd: OrderedDict) -> Optional[int]:
    """尽量从标准 VGG classifier 最后一层推断类别数。"""
    preferred_suffixes = (
        "classifier.6.weight",
        "classifier.3.weight",
        "classifier.weight",
        "fc.weight",
    )

    for suffix in preferred_suffixes:
        for k, v in sd.items():
            if k.endswith(suffix) and isinstance(v, torch.Tensor) and v.ndim == 2:
                return int(v.shape[0])

    # 兜底：取 key 中含 classifier 的最后一个二维 weight
    candidates = []
    for k, v in sd.items():
        if "classifier" in k and k.endswith("weight") and isinstance(v, torch.Tensor) and v.ndim == 2:
            candidates.append((k, v))
    if candidates:
        return int(candidates[-1][1].shape[0])

    return None


def load_checkpoint_object(path: Path, allow_pickle: bool) -> Any:
    """
    优先 weights_only=True。
    如果 checkpoint 是完整 nn.Module，只有显式 --allow-pickle 才允许回退到 pickle。
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as first_exc:
        if not allow_pickle:
            raise RuntimeError(
                "使用 weights_only=True 加载 checkpoint 失败。\n"
                "如果这个文件是 torch.save(model, ...) 保存的完整模型，"
                "并且你确认文件可信，可以增加 --allow-pickle。\n"
                f"原始错误: {first_exc}"
            ) from first_exc
        return torch.load(path, map_location="cpu", weights_only=False)


def build_standard_vgg(arch: str, num_classes: int) -> nn.Module:
    """构造 torchvision 标准 VGG。"""
    try:
        import torchvision.models as models
    except ImportError as exc:
        raise RuntimeError("需要安装 torchvision 才能使用 --arch。") from exc

    if not hasattr(models, arch):
        raise ValueError(
            f"torchvision.models 中不存在 {arch}。"
            "常用值: vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn"
        )

    factory = getattr(models, arch)
    return factory(weights=None, num_classes=num_classes)


def build_custom_model(factory_spec: str) -> nn.Module:
    """
    从 module:function 构造用户自定义模型。
    约定 function() 无参数并返回 nn.Module。
    """
    if ":" not in factory_spec:
        raise ValueError("--factory 格式必须是 module:function，例如 my_vgg:build_model")

    module_name, fn_name = factory_spec.split(":", 1)
    mod = importlib.import_module(module_name)
    fn = getattr(mod, fn_name)
    model = fn()
    if not isinstance(model, nn.Module):
        raise TypeError(f"{factory_spec} 返回的不是 torch.nn.Module")
    return model


def load_model(
    checkpoint: Path,
    arch: Optional[str],
    factory_spec: Optional[str],
    num_classes: Optional[int],
    allow_pickle: bool,
    strict: bool,
) -> nn.Module:
    """构建模型并加载训练权重。"""
    ckpt_obj = load_checkpoint_object(checkpoint, allow_pickle=allow_pickle)
    raw_sd = extract_state_dict(ckpt_obj)

    if factory_spec:
        model = build_custom_model(factory_spec)
    else:
        if not arch:
            raise ValueError("标准 VGG 请提供 --arch；自定义模型请提供 --factory。")
        if num_classes is None:
            inferred = infer_num_classes_from_state_dict(raw_sd)
            num_classes = inferred if inferred is not None else 1000
        model = build_standard_vgg(arch, num_classes=num_classes)

    sd = normalize_state_dict_keys(raw_sd, model)
    result = model.load_state_dict(sd, strict=strict)

    # strict=False 时把差异打印出来，避免用户误以为权重全部加载成功
    if not strict:
        if result.missing_keys:
            print("[警告] missing_keys:")
            for k in result.missing_keys:
                print("   ", k)
        if result.unexpected_keys:
            print("[警告] unexpected_keys:")
            for k in result.unexpected_keys:
                print("   ", k)

    model.eval()
    model.cpu()
    return model


# ============================================================
# 模型执行顺序 / shape 捕获
# ============================================================


@dataclass
class ModuleCall:
    call_index: int
    name: str
    module: nn.Module
    input_shape: Optional[List[int]]
    output_shape: Optional[List[int]]


def is_leaf_module(module: nn.Module) -> bool:
    """没有子 module 的模块视为 leaf。"""
    return len(list(module.children())) == 0


def capture_module_calls(model: nn.Module, input_shape: Sequence[int]) -> List[ModuleCall]:
    """
    用一次 dummy forward 捕获实际执行顺序和每个 leaf module 的输入输出 shape。
    这样 MaxPool 后的 H/W 不需要手工推导。
    """
    name_of = {id(m): n for n, m in model.named_modules()}
    calls: List[ModuleCall] = []
    handles = []

    def make_hook(module: nn.Module):
        def hook(_m: nn.Module, inputs: Any, output: Any):
            calls.append(
                ModuleCall(
                    call_index=len(calls),
                    name=name_of.get(id(module), f"unnamed_{len(calls)}"),
                    module=module,
                    input_shape=tensor_shape(inputs),
                    output_shape=tensor_shape(output),
                )
            )
        return hook

    for m in model.modules():
        if m is model:
            continue
        if is_leaf_module(m):
            handles.append(m.register_forward_hook(make_hook(m)))

    dummy = torch.zeros(tuple(int(x) for x in input_shape), dtype=torch.float32)
    try:
        with torch.inference_mode():
            _ = model(dummy)
    finally:
        for h in handles:
            h.remove()

    return calls


# ============================================================
# Conv Spatial Pattern Template 编译
# ============================================================


@dataclass
class CompiledConvArrays:
    """编译后的 Conv 数组，便于验证和落盘。"""
    weight: np.ndarray                 # [Cin, Kh, Kw, Cout]，Cout 连续
    bias: Optional[np.ndarray]         # [Cout]

    # 每个输入空间位置 src_spatial = iy * Win + ix：
    plan_pattern_id: np.ndarray        # [Hin*Win], int32
    plan_dst_base: np.ndarray          # [Hin*Win], int32，单位是 output spatial index

    # CSR 风格的 pattern table：
    # pattern p 的 entry 范围是 [pattern_ptr[p], pattern_ptr[p+1])
    pattern_ptr: np.ndarray            # [num_patterns+1], int32
    pattern_dst_offset: np.ndarray     # [total_entries], int32，单位是 output spatial index
    pattern_kernel_index: np.ndarray   # [total_entries], int32，ky*Kw+kx
    pattern_weight_offset: np.ndarray  # [total_entries], int64，单位是 float 元素数，相对单个 cin 的起点

    metadata: Dict[str, Any]


def conv_output_hw(
    hin: int,
    win: int,
    kernel: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
) -> Tuple[int, int]:
    """PyTorch Conv2d 标准输出尺寸公式。"""
    kh, kw = kernel
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    hout = math.floor((hin + 2 * ph - dh * (kh - 1) - 1) / sh + 1)
    wout = math.floor((win + 2 * pw - dw * (kw - 1) - 1) / sw + 1)
    return hout, wout


def enumerate_source_contributions(
    iy: int,
    ix: int,
    hout: int,
    wout: int,
    kh: int,
    kw: int,
    sh: int,
    sw: int,
    ph: int,
    pw: int,
    dh: int,
    dw: int,
) -> List[Tuple[int, int]]:
    """
    对一个输入空间位置 (iy, ix)，在编译期枚举它会贡献到哪些输出位置。

    返回列表元素：
        (dst_spatial, kernel_index)

    其中：
        dst_spatial = oy * Wout + ox
        kernel_index = ky * Kw + kx

    卷积定义：
        iy = oy * sh - ph + ky * dh
        ix = ox * sw - pw + kx * dw

    反推给定 (iy, ix) 时哪些 (oy, ox, ky, kx) 合法。
    这一步只发生在编译阶段，仿真阶段不再做这些整除/边界判断。
    """
    contrib: List[Tuple[int, int]] = []

    for ky in range(kh):
        ny = iy + ph - ky * dh
        if ny % sh != 0:
            continue
        oy = ny // sh
        if oy < 0 or oy >= hout:
            continue

        for kx in range(kw):
            nx = ix + pw - kx * dw
            if nx % sw != 0:
                continue
        
            ox = nx // sw
            if ox < 0 or ox >= wout:
                continue

            dst_spatial = oy * wout + ox
            kernel_index = ky * kw + kx
            contrib.append((dst_spatial, kernel_index))

    # 固定顺序让模板 dedup 稳定，也利于顺序访问 destination
    contrib.sort(key=lambda x: (x[0], x[1]))
    return contrib


def compile_conv2d(
    module: nn.Conv2d,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
) -> CompiledConvArrays:
    """
    把一个 nn.Conv2d 编译为 SpatialPlan + PatternTemplate + source-major weight。
    """
    if module.groups != 1:
        raise NotImplementedError(
            f"当前实现只编译 groups=1，收到 groups={module.groups}。"
            "普通 VGG 的 Conv2d 都是 groups=1。"
        )

    if len(input_shape) != 4 or len(output_shape) != 4:
        raise ValueError(
            f"Conv2d 期望 NCHW 输入输出，实际 input={input_shape}, output={output_shape}"
        )

    _, cin, hin, win = [int(x) for x in input_shape]
    _, cout, hout_hook, wout_hook = [int(x) for x in output_shape]

    kh, kw = _pair(module.kernel_size)
    sh, sw = _pair(module.stride)
    ph, pw = _pair(module.padding)
    dh, dw = _pair(module.dilation)

    hout, wout = conv_output_hw(
        hin, win,
        kernel=(kh, kw),
        stride=(sh, sw),
        padding=(ph, pw),
        dilation=(dh, dw),
    )
    if (hout, wout) != (hout_hook, wout_hook):
        raise RuntimeError(
            f"输出尺寸推导与 PyTorch hook 不一致: 推导 {(hout, wout)}, hook {(hout_hook, wout_hook)}"
        )

    if module.in_channels != cin or module.out_channels != cout:
        raise RuntimeError("Conv module channel 数和 hook shape 不一致。")

    # --------------------------------------------------------
    # 1) 权重重排：OIHW -> IHWO
    # --------------------------------------------------------
    # PyTorch Conv2d.weight: [Cout, Cin, Kh, Kw]
    # 我们希望 source spike 固定 cin/ky/kx 后，Cout 是连续的一整段：
    #     W_source[cin, ky, kx, 0:Cout]
    weight = (
        module.weight.detach()
        .cpu()
        .to(torch.float32)
        .permute(1, 2, 3, 0)
        .contiguous()
        .numpy()
    )

    bias = None
    if module.bias is not None:
        bias = to_numpy_f32(module.bias)

    # --------------------------------------------------------
    # 2) 为每个输入 spatial position 生成 source-driven contribution
    #    然后把相同“相对几何关系”的位置 dedup 成 PatternTemplate
    # --------------------------------------------------------
    num_src_spatial = hin * win
    plan_pattern_id = np.empty(num_src_spatial, dtype=np.int32)
    plan_dst_base = np.empty(num_src_spatial, dtype=np.int32)

    # key: ((relative_dst, kernel_index), ...)
    # value: pattern_id
    pattern_map: Dict[Tuple[Tuple[int, int], ...], int] = {}
    patterns: List[Tuple[Tuple[int, int], ...]] = []

    for iy in range(hin):
        for ix in range(win):
            src_spatial = iy * win + ix
            contrib = enumerate_source_contributions(
                iy=iy,
                ix=ix,
                hout=hout,
                wout=wout,
                kh=kh,
                kw=kw,
                sh=sh,
                sw=sw,
                ph=ph,
                pw=pw,
                dh=dh,
                dw=dw,
            )

            if contrib:
                dst_base = min(dst for dst, _ in contrib)
                signature = tuple((dst - dst_base, kernel_index) for dst, kernel_index in contrib)
            else:
                # 某些 stride/dilation 组合理论上可能让一个输入位置完全不参与输出
                dst_base = 0
                signature = tuple()

            if signature not in pattern_map:
                pattern_map[signature] = len(patterns)
                patterns.append(signature)

            plan_pattern_id[src_spatial] = pattern_map[signature]
            plan_dst_base[src_spatial] = dst_base

    # --------------------------------------------------------
    # 3) 把变长 pattern 编码成 CSR-like 数组
    # --------------------------------------------------------
    ptr = [0]
    dst_offsets: List[int] = []
    kernel_indices: List[int] = []
    weight_offsets: List[int] = []

    # 对固定 cin：
    # weight flat 起点 = cin * Kh * Kw * Cout
    # entry 对应 kernel_index 时：
    # weight pointer = cin_base + kernel_index * Cout
    for signature in patterns:
        for dst_offset, kernel_index in signature:
            dst_offsets.append(int(dst_offset))
            kernel_indices.append(int(kernel_index))
            weight_offsets.append(int(kernel_index) * cout)
        ptr.append(len(dst_offsets))

    pattern_ptr = np.asarray(ptr, dtype=np.int32)
    pattern_dst_offset = np.asarray(dst_offsets, dtype=np.int32)
    pattern_kernel_index = np.asarray(kernel_indices, dtype=np.int32)
    pattern_weight_offset = np.asarray(weight_offsets, dtype=np.int64)

    metadata = {
        "op": "conv2d_spatial_template",
        "input_shape_nchw": [int(x) for x in input_shape],
        "output_shape_nchw": [int(x) for x in output_shape],
        "cin": cin,
        "cout": cout,
        "hin": hin,
        "win": win,
        "hout": hout,
        "wout": wout,
        "kernel_h": kh,
        "kernel_w": kw,
        "stride_h": sh,
        "stride_w": sw,
        "padding_h": ph,
        "padding_w": pw,
        "dilation_h": dh,
        "dilation_w": dw,
        "groups": int(module.groups),
        "num_src_spatial": int(num_src_spatial),
        "num_patterns": int(len(patterns)),
        "num_pattern_entries": int(len(dst_offsets)),
        "weight_layout": "[Cin,Kh,Kw,Cout]",
        "runtime_state_layout": "[Hout*Wout,Cout]",
        "src_spatial_index": "iy*Win+ix",
        "dst_spatial_index": "oy*Wout+ox",
        "neuron_id": "dst_spatial*Cout+cout",
        "has_bias": bias is not None,
    }

    return CompiledConvArrays(
        weight=weight,
        bias=bias,
        plan_pattern_id=plan_pattern_id,
        plan_dst_base=plan_dst_base,
        pattern_ptr=pattern_ptr,
        pattern_dst_offset=pattern_dst_offset,
        pattern_kernel_index=pattern_kernel_index,
        pattern_weight_offset=pattern_weight_offset,
        metadata=metadata,
    )


# ============================================================
# Linear source-major 编译
# ============================================================


@dataclass
class CompiledLinearArrays:
    weight: np.ndarray        # [Cin, Cout]
    bias: Optional[np.ndarray]
    metadata: Dict[str, Any]


def compile_linear(module: nn.Linear, input_shape: Sequence[int], output_shape: Sequence[int]) -> CompiledLinearArrays:
    """
    PyTorch Linear weight [Cout, Cin] -> [Cin, Cout]。
    固定一个输入 spike / source feature 后，所有 output feature 权重连续。
    """
    weight = (
        module.weight.detach()
        .cpu()
        .to(torch.float32)
        .t()
        .contiguous()
        .numpy()
    )
    bias = to_numpy_f32(module.bias) if module.bias is not None else None

    metadata = {
        "op": "linear_source_major",
        "input_shape": [int(x) for x in input_shape],
        "output_shape": [int(x) for x in output_shape],
        "cin": int(module.in_features),
        "cout": int(module.out_features),
        "weight_layout": "[Cin,Cout]",
        "has_bias": bias is not None,
    }
    return CompiledLinearArrays(weight=weight, bias=bias, metadata=metadata)


# ============================================================
# 其他 VGG 常见算子 metadata
# ============================================================


def describe_nonparam_module(module: nn.Module, input_shape: Optional[Sequence[int]], output_shape: Optional[Sequence[int]]) -> Dict[str, Any]:
    """把 VGG 中常见的无连接层记录到 manifest。"""
    base = {
        "input_shape": list(input_shape) if input_shape is not None else None,
        "output_shape": list(output_shape) if output_shape is not None else None,
    }

    if isinstance(module, nn.ReLU):
        return {**base, "op": "relu", "inplace": bool(module.inplace)}

    if isinstance(module, nn.MaxPool2d):
        return {
            **base,
            "op": "maxpool2d",
            "kernel_size": list(_pair(module.kernel_size)),
            "stride": list(_pair(module.stride if module.stride is not None else module.kernel_size)),
            "padding": list(_pair(module.padding)),
            "dilation": list(_pair(module.dilation)),
            "ceil_mode": bool(module.ceil_mode),
        }

    if isinstance(module, nn.AvgPool2d):
        return {
            **base,
            "op": "avgpool2d",
            "kernel_size": list(_pair(module.kernel_size)),
            "stride": list(_pair(module.stride if module.stride is not None else module.kernel_size)),
            "padding": list(_pair(module.padding)),
            "ceil_mode": bool(module.ceil_mode),
            "count_include_pad": bool(module.count_include_pad),
        }

    if isinstance(module, nn.AdaptiveAvgPool2d):
        output_size = module.output_size
        if isinstance(output_size, int):
            output_size = [output_size, output_size]
        else:
            output_size = list(output_size)
        return {**base, "op": "adaptive_avgpool2d", "output_size": output_size}

    if isinstance(module, nn.Dropout):
        return {
            **base,
            "op": "dropout",
            "p": float(module.p),
            "identity_in_eval": True,
        }

    if isinstance(module, nn.BatchNorm2d):
        return {
            **base,
            "op": "batchnorm2d",
            "num_features": int(module.num_features),
            "eps": float(module.eps),
            "momentum": None if module.momentum is None else float(module.momentum),
            "affine": bool(module.affine),
            "track_running_stats": bool(module.track_running_stats),
        }

    return {**base, "op": module.__class__.__name__.lower()}


# ============================================================
# 完整网络编译器
# ============================================================


def compile_model(
    model: nn.Module,
    input_shape: Sequence[int],
    output_dir: Path,
    save_format: str = "npy",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    编译完整 VGG / CNN。

    文件结构示例：
        compiled_vgg16/
          manifest.json
          000_features_0_weight.npy
          000_features_0_bias.npy
          000_features_0_plan_pattern_id.npy
          ...
    """
    output_dir = Path(output_dir)

    if output_dir.exists():
        if not overwrite:
            if any(output_dir.iterdir()):
                raise FileExistsError(
                    f"输出目录非空: {output_dir}。如需覆盖请加 --overwrite。"
                )
        else:
            shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    calls = capture_module_calls(model, input_shape=input_shape)

    manifest: Dict[str, Any] = {
        "ir_version": 1,
        "description": "Spatial Pattern Template + source-major weight IR",
        "input_shape_nchw": [int(x) for x in input_shape],
        "save_format": save_format,
        "preferred_runtime_layout": {
            "conv_state": "[H*W,C]",
            "linear_state": "[C]",
            "reason": "固定 source spike 后 destination Cout block 与 weight Cout block 都连续",
        },
        "layers": [],
    }

    previous_output_shape: Optional[List[int]] = None

    for call in calls:
        module = call.module
        layer_prefix = f"{call.call_index:03d}_{call.name.replace('.', '_')}"

        # torchvision VGG 在 avgpool -> classifier 之间使用 functional torch.flatten，
        # 它不是 leaf module，所以 hook 看不到。这里根据 shape 自动插入虚拟 flatten。
        if (
            previous_output_shape is not None
            and call.input_shape is not None
            and len(previous_output_shape) > 2
            and len(call.input_shape) == 2
            and np.prod(previous_output_shape[1:]) == np.prod(call.input_shape[1:])
        ):
            manifest["layers"].append(
                {
                    "index": len(manifest["layers"]),
                    "name": f"virtual_flatten_before_{call.name}",
                    "op": "flatten",
                    "start_dim": 1,
                    "input_shape": previous_output_shape,
                    "output_shape": call.input_shape,
                }
            )

        if isinstance(module, nn.Conv2d):
            if call.input_shape is None or call.output_shape is None:
                raise RuntimeError(f"无法获得 Conv 层 {call.name} 的 shape")

            compiled = compile_conv2d(module, call.input_shape, call.output_shape)

            arrays = {
                "weight": save_array(output_dir / f"{layer_prefix}_weight", compiled.weight, save_format),
                "plan_pattern_id": save_array(
                    output_dir / f"{layer_prefix}_plan_pattern_id",
                    compiled.plan_pattern_id,
                    save_format,
                ),
                "plan_dst_base": save_array(
                    output_dir / f"{layer_prefix}_plan_dst_base",
                    compiled.plan_dst_base,
                    save_format,
                ),
                "pattern_ptr": save_array(
                    output_dir / f"{layer_prefix}_pattern_ptr",
                    compiled.pattern_ptr,
                    save_format,
                ),
                "pattern_dst_offset": save_array(
                    output_dir / f"{layer_prefix}_pattern_dst_offset",
                    compiled.pattern_dst_offset,
                    save_format,
                ),
                "pattern_kernel_index": save_array(
                    output_dir / f"{layer_prefix}_pattern_kernel_index",
                    compiled.pattern_kernel_index,
                    save_format,
                ),
                "pattern_weight_offset": save_array(
                    output_dir / f"{layer_prefix}_pattern_weight_offset",
                    compiled.pattern_weight_offset,
                    save_format,
                ),
            }
            if compiled.bias is not None:
                arrays["bias"] = save_array(
                    output_dir / f"{layer_prefix}_bias", compiled.bias, save_format
                )

            manifest["layers"].append(
                {
                    "index": len(manifest["layers"]),
                    "call_index": call.call_index,
                    "name": call.name,
                    **compiled.metadata,
                    "arrays": arrays,
                }
            )

        elif isinstance(module, nn.Linear):
            if call.input_shape is None or call.output_shape is None:
                raise RuntimeError(f"无法获得 Linear 层 {call.name} 的 shape")

            compiled = compile_linear(module, call.input_shape, call.output_shape)
            arrays = {
                "weight": save_array(output_dir / f"{layer_prefix}_weight", compiled.weight, save_format),
            }
            if compiled.bias is not None:
                arrays["bias"] = save_array(output_dir / f"{layer_prefix}_bias", compiled.bias, save_format)

            manifest["layers"].append(
                {
                    "index": len(manifest["layers"]),
                    "call_index": call.call_index,
                    "name": call.name,
                    **compiled.metadata,
                    "arrays": arrays,
                }
            )

        elif isinstance(module, nn.BatchNorm2d):
            # BN 参数也保存下来。后续为了仿真更快，建议进一步做 Conv+BN fusion。
            meta = describe_nonparam_module(module, call.input_shape, call.output_shape)
            arrays: Dict[str, Any] = {}

            if module.affine:
                arrays["weight"] = save_array(
                    output_dir / f"{layer_prefix}_weight",
                    to_numpy_f32(module.weight),
                    save_format,
                )
                arrays["bias"] = save_array(
                    output_dir / f"{layer_prefix}_bias",
                    to_numpy_f32(module.bias),
                    save_format,
                )

            if module.track_running_stats:
                arrays["running_mean"] = save_array(
                    output_dir / f"{layer_prefix}_running_mean",
                    to_numpy_f32(module.running_mean),
                    save_format,
                )
                arrays["running_var"] = save_array(
                    output_dir / f"{layer_prefix}_running_var",
                    to_numpy_f32(module.running_var),
                    save_format,
                )

            manifest["layers"].append(
                {
                    "index": len(manifest["layers"]),
                    "call_index": call.call_index,
                    "name": call.name,
                    **meta,
                    "arrays": arrays,
                }
            )

        else:
            meta = describe_nonparam_module(module, call.input_shape, call.output_shape)
            manifest["layers"].append(
                {
                    "index": len(manifest["layers"]),
                    "call_index": call.call_index,
                    "name": call.name,
                    **meta,
                }
            )

        previous_output_shape = call.output_shape

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


def export_consolidated_npz(root: Path, manifest: Dict[str, Any], output_path: Path) -> None:
    """把分层 IR 合并为 C++ WeightStore 可直接读取的无 pickle NPZ。"""
    arrays: Dict[str, np.ndarray] = {}
    for layer in manifest["layers"]:
        descriptions = layer.get("arrays")
        if not descriptions:
            continue
        call_index = int(layer.get("call_index", layer["index"]))
        prefix = f"{call_index:03d}_{str(layer['name']).replace('.', '_')}"
        for name, description in descriptions.items():
            # C++ runtime 不需要 kernel_index，但保留它方便离线检查。
            arrays[f"{prefix}_{name}"] = np.asarray(
                load_saved_array(root, description, mmap=False)
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


# ============================================================
# Python 参考 Runtime：直接执行编译后的 Conv
# ============================================================


class CompiledConvRuntime:
    """
    Python 参考实现。

    真正的 C++ simulator 可以把 propagate_spike() 中的 slice 更新替换成 SIMD 内核，
    数据结构完全相同。
    """

    def __init__(self, root: Path, layer_desc: Dict[str, Any], mmap: bool = True):
        if layer_desc["op"] != "conv2d_spatial_template":
            raise ValueError("CompiledConvRuntime 只能加载 conv2d_spatial_template")

        self.root = Path(root)
        self.desc = layer_desc
        arrays = layer_desc["arrays"]

        self.weight = load_saved_array(self.root, arrays["weight"], mmap=mmap)
        self.weight_flat = self.weight.reshape(-1)
        self.bias = load_saved_array(self.root, arrays["bias"], mmap=mmap) if "bias" in arrays else None

        self.plan_pattern_id = load_saved_array(self.root, arrays["plan_pattern_id"], mmap=mmap)
        self.plan_dst_base = load_saved_array(self.root, arrays["plan_dst_base"], mmap=mmap)
        self.pattern_ptr = load_saved_array(self.root, arrays["pattern_ptr"], mmap=mmap)
        self.pattern_dst_offset = load_saved_array(self.root, arrays["pattern_dst_offset"], mmap=mmap)
        self.pattern_kernel_index = load_saved_array(self.root, arrays["pattern_kernel_index"], mmap=mmap)
        self.pattern_weight_offset = load_saved_array(self.root, arrays["pattern_weight_offset"], mmap=mmap)

        self.cin = int(layer_desc["cin"])
        self.cout = int(layer_desc["cout"])
        self.hin = int(layer_desc["hin"])
        self.win = int(layer_desc["win"])
        self.hout = int(layer_desc["hout"])
        self.wout = int(layer_desc["wout"])
        self.kh = int(layer_desc["kernel_h"])
        self.kw = int(layer_desc["kernel_w"])

    def create_output(self, include_bias: bool = True) -> np.ndarray:
        """
        创建 [Hout*Wout, Cout] 输出状态。
        若要完全复现普通 ANN Conv2d，则 bias 对每个输出位置加一次。
        对 SNN，如果 bias 是每 timestep 注入或不使用，可传 include_bias=False。
        """
        y = np.zeros((self.hout * self.wout, self.cout), dtype=np.float32)
        if include_bias and self.bias is not None:
            y += np.asarray(self.bias, dtype=np.float32)[None, :]
        return y

    def propagate_spike(
        self,
        output: np.ndarray,
        cin: int,
        iy: int,
        ix: int,
        value: float = 1.0,
    ) -> None:
        """
        传播一个 source spike / 非零输入。

        运行时关键路径：
          1. O(1) 取 src_spatial
          2. O(1) 取 pattern_id + dst_base
          3. 遍历很小的 pattern（VGG 3x3 interior 通常 9 entries）
          4. 每个 entry 做一段连续 Cout 向量更新

        没有：
          - connection object
          - padding/stride 判断
          - dst neuron 搜索
          - hash map
          - 逐 cout 的 weight 查找
        """
        if not (0 <= cin < self.cin):
            raise IndexError(f"cin 越界: {cin}")
        if not (0 <= iy < self.hin and 0 <= ix < self.win):
            raise IndexError(f"输入空间坐标越界: ({iy}, {ix})")

        src_spatial = iy * self.win + ix
        pattern_id = int(self.plan_pattern_id[src_spatial])
        dst_base = int(self.plan_dst_base[src_spatial])

        begin = int(self.pattern_ptr[pattern_id])
        end = int(self.pattern_ptr[pattern_id + 1])

        # 固定 cin 后，它在 source-major weight 中的整块起点。
        # shape 是 [Cin, Kh, Kw, Cout]。
        cin_weight_base = cin * self.kh * self.kw * self.cout

        for e in range(begin, end):
            dst_spatial = dst_base + int(self.pattern_dst_offset[e])
            weight_start = cin_weight_base + int(self.pattern_weight_offset[e])

            # W[cin, ky, kx, :] 连续 Cout 个 float。
            w = self.weight_flat[weight_start : weight_start + self.cout]

            # output[dst_spatial, :] 也连续 Cout 个 float。
            # C++ 中这里可直接替换为 AVX/AVX-512 vector FMA / add。
            output[dst_spatial, :] += np.float32(value) * w

    def propagate_sparse_tensor(self, x_nchw: np.ndarray, include_bias: bool = True) -> np.ndarray:
        """
        参考实现：遍历 x 中所有非零元素并调用 propagate_spike。
        只用于验证，不建议作为最终高性能 Python simulator。
        """
        x = np.asarray(x_nchw, dtype=np.float32)
        if x.shape != (1, self.cin, self.hin, self.win):
            raise ValueError(
                f"输入 shape 必须是 {(1, self.cin, self.hin, self.win)}，实际 {x.shape}"
            )

        y = self.create_output(include_bias=include_bias)
        nz = np.argwhere(x[0] != 0)
        for cin, iy, ix in nz:
            self.propagate_spike(
                y,
                cin=int(cin),
                iy=int(iy),
                ix=int(ix),
                value=float(x[0, cin, iy, ix]),
            )

        # simulator 内部建议一直保留 [H*W,C]。
        # 为了和 PyTorch NCHW 对比，这里转换回来。
        return y.reshape(self.hout, self.wout, self.cout).transpose(2, 0, 1)[None, ...]


class CompiledLinearRuntime:
    """Linear 的 source-major 参考执行器。"""

    def __init__(self, root: Path, layer_desc: Dict[str, Any], mmap: bool = True):
        if layer_desc["op"] != "linear_source_major":
            raise ValueError("CompiledLinearRuntime 只能加载 linear_source_major")
        arrays = layer_desc["arrays"]
        self.weight = load_saved_array(root, arrays["weight"], mmap=mmap)
        self.bias = load_saved_array(root, arrays["bias"], mmap=mmap) if "bias" in arrays else None
        self.cin = int(layer_desc["cin"])
        self.cout = int(layer_desc["cout"])

    def create_output(self, include_bias: bool = True) -> np.ndarray:
        y = np.zeros(self.cout, dtype=np.float32)
        if include_bias and self.bias is not None:
            y += np.asarray(self.bias, dtype=np.float32)
        return y

    def propagate_spike(self, output: np.ndarray, src_index: int, value: float = 1.0) -> None:
        # weight[src_index, :] 是连续 Cout 元素
        output[:] += np.float32(value) * self.weight[src_index, :]


# ============================================================
# 正确性验证
# ============================================================


def get_named_module(model: nn.Module, name: str) -> nn.Module:
    """兼容根 module 名为空字符串的 named_modules 查找。"""
    if name == "":
        return model
    table = dict(model.named_modules())
    return table[name]


def verify_compiled_conv_layer(
    model: nn.Module,
    root: Path,
    layer_desc: Dict[str, Any],
    num_spikes: int,
    seed: int,
    atol: float,
    rtol: float,
) -> Tuple[bool, float, float]:
    """
    用一个随机稀疏输入验证：
      编译后的 source-driven event 执行 == PyTorch F.conv2d
    """
    module = get_named_module(model, layer_desc["name"])
    if not isinstance(module, nn.Conv2d):
        raise TypeError(f"manifest {layer_desc['name']} 对应的不是 nn.Conv2d")

    rt = CompiledConvRuntime(root, layer_desc, mmap=True)

    rng = np.random.default_rng(seed)
    x = np.zeros((1, rt.cin, rt.hin, rt.win), dtype=np.float32)

    total = rt.cin * rt.hin * rt.win
    k = min(int(num_spikes), total)
    flat_indices = rng.choice(total, size=k, replace=False)
    values = rng.standard_normal(k).astype(np.float32)

    for flat, value in zip(flat_indices, values):
        cin = int(flat // (rt.hin * rt.win))
        rem = int(flat % (rt.hin * rt.win))
        iy = rem // rt.win
        ix = rem % rt.win
        x[0, cin, iy, ix] = value

    y_compiled = rt.propagate_sparse_tensor(x, include_bias=True)

    xt = torch.from_numpy(x)
    with torch.inference_mode():
        y_ref = F.conv2d(
            xt,
            module.weight.detach().cpu(),
            module.bias.detach().cpu() if module.bias is not None else None,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        ).numpy()

    abs_diff = np.abs(y_compiled - y_ref)
    max_abs = float(abs_diff.max()) if abs_diff.size else 0.0

    denom = np.maximum(np.abs(y_ref), 1e-12)
    max_rel = float((abs_diff / denom).max()) if abs_diff.size else 0.0

    ok = bool(np.allclose(y_compiled, y_ref, atol=atol, rtol=rtol))
    return ok, max_abs, max_rel


def verify_all_convs(
    model: nn.Module,
    root: Path,
    manifest: Dict[str, Any],
    num_spikes: int,
    seed: int,
    atol: float = 2e-5,
    rtol: float = 2e-5,
) -> bool:
    """验证所有编译后的 Conv 层。"""
    all_ok = True
    conv_idx = 0

    print("\n========== 正确性验证：compiled event Conv vs PyTorch Conv ==========")
    for layer in manifest["layers"]:
        if layer.get("op") != "conv2d_spatial_template":
            continue

        ok, max_abs, max_rel = verify_compiled_conv_layer(
            model=model,
            root=root,
            layer_desc=layer,
            num_spikes=num_spikes,
            seed=seed + conv_idx,
            atol=atol,
            rtol=rtol,
        )
        conv_idx += 1
        all_ok &= ok

        status = "PASS" if ok else "FAIL"
        print(
            f"[{status}] {layer['name']:<24} "
            f"patterns={layer['num_patterns']:<3d} "
            f"entries={layer['num_pattern_entries']:<4d} "
            f"max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
        )

    print("===================================================================")
    return all_ok


# ============================================================
# 统计信息
# ============================================================


def estimate_expanded_connection_count(layer: Dict[str, Any]) -> int:
    """
    对 Conv 估算如果完全展开 neuron-to-neuron edge，需要多少 connection。

    对每个 source spatial position，它对应 pattern 中若干 kernel entries；
    每个 input channel 的每个 kernel entry 又连 Cout 个输出 channel。
    """
    if layer.get("op") != "conv2d_spatial_template":
        return 0

    # 需要读取 plan 和 ptr 才能精确；manifest 中不保存每个 pattern 的使用次数。
    return -1


def print_compile_summary(root: Path, manifest: Dict[str, Any]) -> None:
    """打印 IR 统计。"""
    total_weight_bytes = 0
    total_plan_bytes = 0
    total_pattern_bytes = 0

    print("\n================ 编译结果 ================")
    for layer in manifest["layers"]:
        op = layer.get("op")
        if op == "conv2d_spatial_template":
            arrays = layer["arrays"]
            total_weight_bytes += arrays["weight"]["nbytes"]

            for key in ("plan_pattern_id", "plan_dst_base"):
                total_plan_bytes += arrays[key]["nbytes"]
            for key in (
                "pattern_ptr",
                "pattern_dst_offset",
                "pattern_kernel_index",
                "pattern_weight_offset",
            ):
                total_pattern_bytes += arrays[key]["nbytes"]

            print(
                f"Conv {layer['name']:<24} "
                f"{layer['cin']:>4} -> {layer['cout']:<4} "
                f"{layer['hin']}x{layer['win']} -> {layer['hout']}x{layer['wout']} "
                f"K={layer['kernel_h']}x{layer['kernel_w']} "
                f"patterns={layer['num_patterns']} entries={layer['num_pattern_entries']}"
            )

        elif op == "linear_source_major":
            arrays = layer["arrays"]
            total_weight_bytes += arrays["weight"]["nbytes"]
            print(
                f"FC   {layer['name']:<24} "
                f"{layer['cin']} -> {layer['cout']}"
            )

    print("------------------------------------------")
    print(f"Weight bytes : {total_weight_bytes / 1024 / 1024:.2f} MiB")
    print(f"Plan bytes   : {total_plan_bytes / 1024:.2f} KiB")
    print(f"Pattern bytes: {total_pattern_bytes / 1024:.2f} KiB")
    print(f"Manifest     : {root / 'manifest.json'}")
    print("==========================================")


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把训练好的 VGG/CNN 编译成 Spatial Pattern Template + source-major weight IR"
    )

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--arch",
        type=str,
        default=None,
        help="torchvision VGG 名，例如 vgg16 / vgg16_bn / vgg19",
    )
    model_group.add_argument(
        "--factory",
        type=str,
        default=None,
        help="自定义模型 factory，格式 module:function，function() 返回 nn.Module",
    )

    parser.add_argument("--checkpoint", type=Path, required=True, help="训练好的 .pth/.pt checkpoint")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="标准 torchvision VGG 的类别数；不填时尽量从 checkpoint 推断",
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=4,
        metavar=("N", "C", "H", "W"),
        default=[1, 3, 224, 224],
        help="模型输入 NCHW；编译通常 N=1",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="IR 输出目录")
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=None,
        help="可选：额外合并成 C++ WeightStore 使用的 weights.npz",
    )
    parser.add_argument(
        "--save-format",
        choices=("npy", "raw"),
        default="npy",
        help="npy 方便 Python 调试；raw 方便 C/C++ 直接 mmap",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出目录")
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="load_state_dict(strict=False)，仅在你清楚缺失/多余 key 的原因时使用",
    )
    parser.add_argument(
        "--allow-pickle",
        action="store_true",
        help="允许加载 torch.save(model, ...) 的完整 pickle；仅对可信 checkpoint 使用",
    )
    parser.add_argument("--verify", action="store_true", help="编译后逐 Conv 层与 PyTorch 做正确性验证")
    parser.add_argument(
        "--verify-spikes",
        type=int,
        default=128,
        help="每个 Conv 验证时随机放多少个非零输入",
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.input_shape[0] != 1:
        print("[警告] 当前 IR 面向单样本/事件仿真；建议 --input-shape 的 N=1。")

    print("[1/4] 加载模型和训练权重...")
    model = load_model(
        checkpoint=args.checkpoint,
        arch=args.arch,
        factory_spec=args.factory,
        num_classes=args.num_classes,
        allow_pickle=args.allow_pickle,
        strict=not args.non_strict,
    )

    print("[2/4] 捕获 VGG 各层 shape，并编译 connectivity template...")
    manifest = compile_model(
        model=model,
        input_shape=args.input_shape,
        output_dir=args.output_dir,
        save_format=args.save_format,
        overwrite=args.overwrite,
    )
    if args.output_npz is not None:
        export_consolidated_npz(args.output_dir, manifest, args.output_npz)

    print("[3/4] 输出统计...")
    print_compile_summary(args.output_dir, manifest)

    if args.verify:
        print("[4/4] 验证编译后事件执行结果...")
        ok = verify_all_convs(
            model=model,
            root=args.output_dir,
            manifest=manifest,
            num_spikes=args.verify_spikes,
            seed=args.seed,
        )
        if not ok:
            print("\n验证失败：至少一个 Conv 与 PyTorch 结果不一致。", file=sys.stderr)
            return 2
    else:
        print("[4/4] 跳过验证（如需验证请加 --verify）。")

    print("\n完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
