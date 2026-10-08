"""Resolve the tokenizer used by the portable bundle's text-only pipelines."""

from typing import Any


def text_only_tokenizer(processing_object: Any) -> Any:
    """Unwrap a multimodal processor without replacing its native chat template.

    ProcessorMixin.apply_chat_template may require typed content blocks. Our
    question/evidence messages contain strings, and the text trainer also needs
    tokenizer padding, encoding and saving APIs directly.
    """
    tokenizer = getattr(processing_object, "tokenizer", None)
    if tokenizer is None:
        return processing_object
    if not getattr(tokenizer, "chat_template", None):
        template = getattr(processing_object, "chat_template", None)
        if not template:
            raise ValueError("Multimodal processor has no native text chat template")
        tokenizer.chat_template = template
    return tokenizer
