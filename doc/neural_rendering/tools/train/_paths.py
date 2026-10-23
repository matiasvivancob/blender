# SPDX-License-Identifier: Apache-2.0
"""Import side effect: make `nnm` (tools/nnm) and the numpy prototype (experiments/neural_material_train) importable.

Every script in this directory starts with `import _paths  # noqa: F401`.
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)                                    # doc/neural_rendering/tools
DOC = os.path.dirname(TOOLS)                                     # doc/neural_rendering
PROTO_DIR = os.path.join(DOC, "experiments", "neural_material_train")

for _p in (HERE, TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_proto(name="proto_train"):
    """Import experiments/neural_material_train/train.py (numpy trainer) under a unique module name.

    The prototype does `import material as M`, so its directory is appended to sys.path (at the end, so
    nothing in this directory is shadowed). Returns the module; `module.M` is the analytic material.
    """
    if name in sys.modules:
        return sys.modules[name]
    if PROTO_DIR not in sys.path:
        sys.path.append(PROTO_DIR)
    spec = importlib.util.spec_from_file_location(name, os.path.join(PROTO_DIR, "train.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_material():
    """experiments/neural_material_train/material.py (analytic layered BRDF + procedural texture)."""
    if PROTO_DIR not in sys.path:
        sys.path.append(PROTO_DIR)
    import material
    return material
