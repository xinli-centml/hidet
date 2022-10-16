from typing import Union, List, Dict, Sequence, Tuple, Set, Optional

from hidet.ir.type import uint8, int32
from hidet.ir.expr import TensorElement, Call, Expr, Var, scalar_var, convert, cast, var
from hidet.ir.stmt import Stmt, AssignStmt, SeqStmt, ForStmt, DeclareStmt, BufferStoreStmt, ForTaskStmt, EvaluateStmt
from hidet.ir.task import Task
from hidet.ir.func import IRModule, Function
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.functors import ExprRewriter, ExprVisitor, inline_compute, collect, rewrite, infer_type, simplify_to_int
from hidet.ir.dialects.compute import ComputeNode, ScalarNode, TensorNode, GridCompute, ReduceCompute, ArgReduceCompute, TensorCompute
from hidet.ir.dialects.lowlevel import TensorPointerType, tensor_pointer_var, void_pointer, VoidType
from hidet.ir.primitives.runtime import request_cuda_workspace, request_cpu_workspace
from hidet.utils import prod, DirectedGraph
from hidet.utils.namer import Namer


class ScalarComputeFound(Exception):
    pass


class GridComputeInlineChecker(ExprVisitor):
    def check(self, gc: GridCompute) -> bool:
        """Check whether the grid compute can be inlined.

        A grid compute can be inlined if and only if it only directly accesses TensorNode but not ScalarNode.

        Parameters
        ----------
        gc: GridCompute
            The grid compute to be checked.

        Returns
        -------
        ret: bool
            True if the grid can be inlined, else False.
        """
        try:
            self.visit(gc.value)
        except ScalarComputeFound:
            return False
        return True

    def visit_TensorNode(self, e: TensorNode):
        return

    def visit_ScalarNode(self, e: ScalarNode):
        if e.scalar_compute is not None:
            raise ScalarComputeFound()


def can_inline_grid_compute(gc: GridCompute) -> bool:
    return GridComputeInlineChecker().check(gc)


class GridComputeInliner(ExprRewriter):
    def __init__(self):
        super().__init__()

    def visit_TensorElement(self, e: TensorElement):
        base = self(e.base)
        indices = [self(index) for index in e.indices]
        if isinstance(base, TensorNode) and isinstance(base.tensor_compute, GridCompute):
            gc = base.tensor_compute
            if can_inline_grid_compute(gc):
                return rewrite(gc.value, {axis: index for axis, index in zip(gc.axes, indices)})
        return ExprRewriter.visit_TensorElement(self, e)


def inline_grid_compute(nodes: List[TensorNode]) -> List[TensorNode]:
    """Inline the grid compute that does not contain reduce.

    If a grid compute only does not contain scalar compute (e.g., reduce and arg_reduce), the grid will be
    inlined.

    For example:

    .. code-block:: python

         from hidet.ir.dialects.compute import tensor_input, compute
         a = tensor_input('a', 'float32', [10])
         b = compute('b', [10], lambda i: a[i] + 1)
         c = compute('c', [10], lambda i: b[i] * 2)
         d = inline_grid_compute(c)  # d is equivalent to
         e = compute('e', [10], lambda i: (a[i] + 1) * 2)

    Parameters
    ----------
    nodes: List[TensorNode]
        The compute nodes.

    Returns
    -------
    ret: List[TensorNode]
        The nodes after inlining.
    """
    inliner = GridComputeInliner()
    return [inliner(node) for node in nodes]


class AutoScheduler:
    def __init__(self):
        super().__init__()
        self.ir_module: IRModule = IRModule()

    @staticmethod
    def get_accessed_nodes(node: TensorNode) -> List[TensorNode]:
        tc = node.tensor_compute
        if tc is None:
            return []
        elif isinstance(tc, GridCompute):
            e = tc.value
        else:
            raise NotImplementedError()
        accessed_nodes: List[TensorNode] = collect(e, TensorNode, stop_when_found=True)
        accessed_nodes = list(set(accessed_nodes))  # remove duplicated ones
        return accessed_nodes

    @staticmethod
    def build_dag(outputs: Sequence[TensorNode]) -> DirectedGraph:
        dag = DirectedGraph()
        remain_nodes: List[TensorNode] = list(outputs)
        while len(remain_nodes) > 0:
            node = remain_nodes.pop()
            dag.add_node(node)
            accessed_nodes: List[TensorNode] = AutoScheduler.get_accessed_nodes(node)
            for accessed_node in accessed_nodes:
                if accessed_node not in dag:
                    remain_nodes.append(accessed_node)
                dag.add_edge(accessed_node, node)
        return dag

    @staticmethod
    def plan_memory(dag: DirectedGraph, order: Sequence[TensorNode], require_allocate: Set[TensorNode]) -> Tuple[int, Dict[TensorNode, int]]:
        from hidet.ir.functors import simplify_to_int
        # dag has not been used in this simple plan.
        alignment_bytes: int = 128  # make sure each buffer aligns with 128 bytes
        allocated_bytes: int = 0
        buffer_offset: Dict[TensorNode, int] = {}
        for tensor in order:
            if tensor not in require_allocate:
                continue
            buffer_offset[tensor] = allocated_bytes
            allocated_bytes += simplify_to_int(tensor.data_type.storage_bytes())
            allocated_bytes = (allocated_bytes + alignment_bytes - 1) // alignment_bytes * alignment_bytes
        return allocated_bytes, buffer_offset

    @staticmethod
    def allocate_tensors(fb: FunctionBuilder, device: str, buffer_bytes: int, buffer_offset: Dict[TensorNode, int], node_map: Dict[TensorNode, Var]):
        if buffer_bytes > 0:
            buffer = Var('buffer', TensorPointerType(scope='unspecified', dtype='uint8', shape=[buffer_bytes]))
            if device == 'cuda':
                space_ptr: Expr = request_cuda_workspace(nbytes=buffer_bytes, require_clean=False)
            elif device == 'cpu':
                space_ptr: Expr = request_cpu_workspace(nbytes=buffer_bytes, require_clean=False)
            else:
                raise ValueError()
            fb += DeclareStmt(buffer, init=cast(space_ptr, ~uint8))
        else:
            buffer = None
        for node in buffer_offset:
            if node in node_map:
                # this node is either an input or output tensor
                continue
            else:
                assert buffer is not None
                v = Var(node.name, TensorPointerType.from_tensor_type(node.data_type))
                node_map[node] = v
                fb += DeclareStmt(v, init=cast(~buffer[buffer_offset[node]], ~v.type.tensor_type.scalar_type))

    def schedule_task(self, task: Task, device: str) -> IRModule:
        # absorb the prologue and epilogue into a single task
        task = task.task_graph.absorb()

        self.ir_module.task = task

        # Inline the grid compute that does not contain reduce
        outputs: List[TensorNode] = inline_grid_compute(task.outputs)

        # Taking the TensorNode as node to construct the computation directed-acyclic-graph (DAG)
        # In the DAG, each node is a TensorNode and each edge (src, dst) indicates src is accessed by dst.
        dag = self.build_dag(outputs)

        # Get a topological order of the tensor nodes in the DAG
        order: List[TensorNode] = dag.topological_order()

        # Plan the memory for intermediate tensors
        require_allocate = set(node for node in order if node not in task.inputs and node not in outputs)
        buffer_bytes, buffer_offset = self.plan_memory(dag, order, require_allocate)

        # Allocate the memory for intermediate tensors, get the mapping from node to tensor var or tensor pointer var
        with FunctionBuilder(name=task.name, kind='packed_func') as fb:
            # packed function arguments, packed_func(num_args: int32, arg_types: *int32, args: **void)
            num_args = scalar_var('num_args', 'int32')
            arg_types = Var('arg_types', ~int32)
            args = Var('args', ~void_pointer())
            fb.extend_params([num_args, arg_types, args])

            # extract the actual arguments from packed arguments
            params: List[Var] = []
            for idx, task_param in enumerate(task.inputs + task.outputs):
                param = Var(task_param.name, ~task_param.data_type.scalar_type)
                params.append(param)
                fb += DeclareStmt(param, init=cast(args[idx], param.type))

            # allocate memory space for intermediate tensors
            node_map: Dict[TensorNode, Var] = {a: b for a, b in zip(task.inputs + outputs, params)}
            self.allocate_tensors(fb, device, buffer_bytes, buffer_offset, node_map)

            # schedule each tensor computation
            for tensor in order:
                if tensor.tensor_compute is None:
                    # input tensor does not need scheduling, skip
                    continue
                fb += self.schedule_tensor_node(tensor, node_map)
        func = fb.get()
        self.ir_module.add(func.name, func)

        return self.ir_module

    def schedule_tensor_node(self, node: TensorNode, node_map: Dict[TensorNode, Expr]) -> Stmt:
        tc = node.tensor_compute
        if isinstance(tc, GridCompute):
            return self.schedule_grid_compute(tc, node, node_map)
        else:
            raise ValueError('Cannot recognize tensor compute node: {}.'.format(type(tc).__name__))

    def add_function(self, func: Function) -> Var:
        """Add a function to current ir module.

        This function is used to add a function to current ir module, which allows the
        schedule_grid_compute method calls a function to implement the given computation.

        Parameters
        ----------
        func: Function
            The function to be added.

        Returns
        -------
        ret: Var
            The variable points to the added function.
        """
        name = Namer.unique_name_among(func.name, self.ir_module.functions.keys())
        func.name = name
        self.ir_module.add(func.name, func)
        return self.ir_module.lookup_var(func.name)

    def schedule_grid_compute(self, gc: GridCompute, node: TensorNode, node_map: Dict[TensorNode, Expr]) -> Stmt:
        raise NotImplementedError()


class ComputeExprLower(ExprRewriter):
    def __init__(self, expr: Expr, param_map: Dict[Union[TensorNode, ScalarNode], Expr]):
        super().__init__()
        self.sb: StmtBuilder = StmtBuilder()
        self.compute_expr: Expr = expr
        self.param_map: Dict[Union[TensorNode, ScalarNode], Expr] = param_map

    def lower(self) -> Tuple[List[Stmt], Expr]:
        result = self.visit(self.compute_expr)
        assert len(self.sb.scope_stack) == 1, "some scope has not been exited?"
        return self.sb.scope_stack[0], result

    def visit_TensorNode(self, e: TensorNode):
        if e in self.param_map:
            return self.param_map[e]

        if e.tensor_compute is None:
            raise ValueError('Expect tensor input in param_map.')

        tc = e.tensor_compute
        if isinstance(tc, GridCompute):
            grid_compute: GridCompute = tc
            # declare intermediate tensor buffer
            buf = Var(e.name, e.data_type)

            shape, axes, value = grid_compute.shape, grid_compute.axes, grid_compute.value
            # tensor compute loops
            for i in range(len(shape)):
                self.sb.enter_body(ForStmt(axes[i], shape[i]))

            # at the innermost loop body
            expr = self.visit(grid_compute.value)
            self.sb.append(BufferStoreStmt(buf, axes, expr))

            # exit loop scope
            for i in range(len(shape)):
                self.sb.exit_body()
            return buf
        else:
            raise NotImplementedError('Compute pattern {}'.format(type(tc).__name__))

    def visit_ScalarNode(self, e: ScalarNode):
        if e in self.param_map:
            return self.param_map[e]

        if e.scalar_compute is None:
            raise ValueError('Expect scalar input in param_map.')

        sc = e.scalar_compute
        if isinstance(sc, ReduceCompute):
            shape, axes, value = sc.shape, sc.axes, sc.value
            # declare accumulator
            acc = scalar_var(e.name, infer_type(value))
            self.sb += DeclareStmt(acc, init=sc.reduce_operation.initial_value(e.data_type.name))

            # reduction loops
            for i in range(len(shape)):
                self.sb.enter_body(ForStmt(axes[i], shape[i]))

            # at the innermost loop body
            expr = self.visit(value)
            self.sb += AssignStmt(acc, sc.reduce_operation.combine(acc, expr))

            # exit loop scope
            for i in range(len(shape)):
                self.sb.exit_body()

            # finalize
            acc = sc.reduce_operation.finalize(acc, prod(shape))

            return acc
        elif isinstance(sc, ArgReduceCompute):
            extent, axis, value = sc.extent, sc.axis, sc.value
            value_dtype = infer_type(value)
            # declare index accumulator
            acc_index = scalar_var(e.name + '_idx', sc.index_dtype)
            acc_value = scalar_var(e.name + '_val', value_dtype)

            # init accumulator
            self.sb += DeclareStmt(acc_index, init=convert(0))
            self.sb += DeclareStmt(acc_value, init=sc.reduce_operation.initial_value(value_dtype))
            self.sb += AssignStmt(acc_index, 0)

            # reduction loops
            self.sb.enter_body(ForStmt(axis, extent))

            # compare and update index
            expr = self.visit(value)
            with self.sb.if_then(sc.reduce_operation.arg_combine(lhs_value=expr, rhs_value=acc_value)):
                self.sb += AssignStmt(acc_value, expr)
                self.sb += AssignStmt(acc_index, axis)

            # exit loop
            self.sb.exit_body()

            return acc_index
        else:
            raise NotImplementedError('Compute pattern {}'.format(type(sc).__name__))