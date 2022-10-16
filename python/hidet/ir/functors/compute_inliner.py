from hidet.ir.compute import TensorNode, GridCompute
from hidet.ir.expr import TensorElement
from hidet.utils import prod, same_list
from .base import ExprRewriter
from .util_functors import rewrite


class ComputeInlineRewriter(ExprRewriter):
    def __init__(self, reduce_limit=0):
        super().__init__()
        self.reduce_limit = reduce_limit

    def visit_TensorElement(self, e: TensorElement):
        base = self(e.base)
        indices = [self(index) for index in e.indices]
        if isinstance(base, TensorNode) and base.tensor_compute:
            if isinstance(base.tensor_compute, GridCompute):
                grid_compute = base.tensor_compute
                input_scalars = grid_compute.input_scalars
                cnt = sum(prod(input_scalar.scalar_compute.const_shape()) for input_scalar in input_scalars if input_scalar.scalar_compute)
                if cnt == 0 or cnt <= self.reduce_limit:
                    return rewrite(grid_compute.value, {axis: index for axis, index in zip(grid_compute.axes, indices)})
        return ExprRewriter.visit_TensorElement(self, e)


def inline_compute(expr: TensorNode, reduce_limit=0) -> TensorNode:
    """
    Inline the computation.

    GridCompute(axes => value)[indices] => value (axes replaced by indices).

    Parameters
    ----------
    expr: TensorNode
        the computation node to inline.
    reduce_limit: int, default 0
        reduce_limit < 0: Do not allow reducing compute in the computation, raise an Exception when encountered.
        reduce_limit == 0: Allow reducing compute in the computation, but do not expand it.
        reduce_limit > 0: Allow reducing compute, and expand it when its extent is equal or greater than reduce_limit.

    Returns
    -------
    ret: TensorNode
        Inlined compute.
    """
    inliner = ComputeInlineRewriter(reduce_limit)
    return inliner(expr)
