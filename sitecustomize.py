
import sys
import types


def _install_dgl_graphbolt_stub():
    if "dgl.graphbolt" in sys.modules:
        return

    gb = types.ModuleType("dgl.graphbolt")
    gb.FusedCSCSamplingGraph = type("FusedCSCSamplingGraph", (), {})
    gb.load_from_shared_memory = lambda *a, **k: None
    gb.expand_indptr = lambda *a, **k: None
    gb.etype_tuple_to_str = lambda *a, **k: None
    gb.fused_csc_sampling_graph = lambda *a, **k: None
    gb.load_graphbolt = lambda *a, **k: None
    sys.modules["dgl.graphbolt"] = gb


def _install_dgl_sparse_stub():
    if "dgl.sparse" in sys.modules:
        return

    import torch

    class SparseMatrix:
        def __init__(self, indices, val=None, shape=None, **kwargs):
            num_rows, num_cols = shape if shape is not None else (0, 0)
            dense = torch.zeros(num_rows, num_cols, dtype=torch.float32)
            if indices is not None and indices.numel() > 0:
                if val is None:
                    val = torch.ones(indices.shape[1], dtype=torch.float32)
                dense.index_put_(
                    (indices[0].long(), indices[1].long()),
                    val.to(dtype=torch.float32),
                )
            self._dense = dense

        def to_dense(self):
            return self._dense

        def __repr__(self):
            return f"SparseMatrix(stub, shape={tuple(self._dense.shape)})"

    sparse = types.ModuleType("dgl.sparse")
    sparse.spmatrix = SparseMatrix
    sparse.SparseMatrix = SparseMatrix
    sys.modules["dgl.sparse"] = sparse


_install_dgl_graphbolt_stub()
_install_dgl_sparse_stub()
