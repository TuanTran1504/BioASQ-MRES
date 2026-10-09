"""Explicit, recorded compilation policy for CUDA recovery runs."""

import os


def prepare_execution(mode):
    if mode not in {"default", "eager"}:
        raise ValueError(f"Unsupported execution mode: {mode}")
    if mode == "eager":
        # Set before importing Unsloth, which patches/compiles model forwards.
        os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
    elif os.environ.get("UNSLOTH_COMPILE_DISABLE", "0") != "0":
        raise ValueError("Default execution requested with compilation disabled in the environment")


def activate_execution(mode, torch):
    if mode == "eager":
        setter = getattr(getattr(torch, "compiler", None), "set_stance", None)
        if not callable(setter):
            raise RuntimeError("Eager recovery requires torch.compiler.set_stance")
        # Also bypass torch.compile decorators in the patched LoRA forward path.
        setter("force_eager")
