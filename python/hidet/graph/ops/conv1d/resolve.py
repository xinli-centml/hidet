from typing import List, Optional
from hidet.graph.operator import Operator, Tensor
from hidet.graph.transforms import ResolveRule, register_resolve_rule
from hidet.graph import ops
from hidet.ir.expr import is_constant

from .conv1d import Conv1dOp


@register_resolve_rule(Conv1dOp)
class Conv1dResolveRule(ResolveRule):
    def __init__(self, enable_winograd=False):
        self.enable_winograd = enable_winograd

    def resolve(self, op: Operator) -> Optional[List[Tensor]]:
        assert isinstance(op, Conv1dOp)
        (stride,) = ops.utils.normalize_stride(op.attrs['stride'], dim=1)
        groups = op.attrs['groups']
        (dilations,) = op.attrs['dilations']
        channels = op.inputs[1].shape[0]

        if is_constant(channels) and groups == channels:
            return None  # use depthwise schedule in the default Task

        data, weight = op.inputs
        # implicit gemm algorithm
        out = ops.conv1d_gemm(data, weight, stride, dilations, groups)
        return [out]
