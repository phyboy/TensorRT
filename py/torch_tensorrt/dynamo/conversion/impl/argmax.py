from typing import Optional, cast

import numpy as np
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion.impl.shape import get_shape_with_dynamic_shape
from torch_tensorrt.fx.converters.converter_utils import (
    get_positive_dim,
    has_dynamic_shape,
    to_numpy,
)
from torch_tensorrt.fx.types import Shape, TRTNetwork, TRTTensor

import tensorrt as trt


def argmax(
    network: TRTNetwork,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    dim: int = 0,
    keep_dim: bool = False,
) -> TRTTensor:
    if not isinstance(input, TRTTensor):
        raise RuntimeError(
            f"argmax received input {input} that is not part "
            "of the TensorRT region!"
        )
    if dim < 0:
        dim = len(tuple(input.shape)) + dim
    reduce_mask = 1 << dim
    topk_layer = network.add_topk(input, trt.TopKOperation.MAX, 1, reduce_mask)

    set_layer_name(topk_layer, target, name)

    return topk_layer.get_output(1)
    
    
