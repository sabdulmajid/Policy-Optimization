from __future__ import annotations

from typing import Iterable

from PIL import Image, ImageDraw


def _bbox_xyxy(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = box
    left = max(0, min(int(round(x)), width - 1))
    top = max(0, min(int(round(y)), height - 1))
    right = max(left + 1, min(int(round(x + w)), width))
    bottom = max(top + 1, min(int(round(y + h)), height))
    return left, top, right, bottom


def mask_entities(
    image: Image.Image,
    entity_boxes: Iterable[list[float]],
    *,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    masked = image.copy()
    draw = ImageDraw.Draw(masked)
    for box in entity_boxes:
        draw.rectangle(_bbox_xyxy(box, image.width, image.height), fill=fill)
    return masked
