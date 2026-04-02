"""Shared content block types for multimodal message storage.

Messages with images use a list[ContentBlock] instead of a plain str.
ImageRefBlock stores a disk path (not base64) to keep memory low.
Conversion to Anthropic API format happens at request assembly time.
"""

from typing import Literal, TypedDict


class TextBlock(TypedDict):
    type: Literal["text"]
    text: str


class ImageRefBlock(TypedDict):
    type: Literal["image_ref"]
    path: str  # disk path, e.g. "storage/image_cache/ab/ab3f7c.jpg"
    media_type: str  # e.g. "image/jpeg"


ContentBlock = TextBlock | ImageRefBlock

# Messages store content as str (text-only, backward compat) or list of blocks (multimodal).
Content = str | list[ContentBlock]
