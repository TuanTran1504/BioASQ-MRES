"""Stable text generation for quantised Ministral 3 on Gadi V100 GPUs."""


def configure_ministral_generation(model):
    """Keep PEFT hooks, using Transformers generation with a dynamic FP16 cache.

    FastModel's generation wrapper forces a static cache even when a caller
    requests dynamic caching. On the observed V100 adapter reload this allocated
    Float cache values for Half attention values. Use its saved original generate
    method on the underlying backbone, retaining the outer PEFT generate method.
    """
    backbone = model.get_base_model() if callable(getattr(model, "get_base_model", None)) else model
    config = getattr(backbone, "config", None)
    text_config = getattr(config, "text_config", None)
    types = {getattr(config, "model_type", None), getattr(text_config, "model_type", None)}
    if not types.intersection({"mistral3", "ministral3"}):
        return model
    if getattr(backbone, "_bioasq_dynamic_generation", False):
        return model
    import torch
    original = getattr(backbone, "_old_generate", None)
    if not callable(original):
        original = backbone.generate

    def generate(*args, **kwargs):
        kwargs["cache_implementation"] = "dynamic"
        kwargs["disable_compile"] = True
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            return original(*args, **kwargs)

    backbone.generate = generate
    backbone._bioasq_dynamic_generation = True
    model._bioasq_generation_backend = "transformers-dynamic-fp16-v100"
    print("Ministral generation: Transformers dynamic cache, FP16 autocast, decode compilation disabled", flush=True)
    return model
