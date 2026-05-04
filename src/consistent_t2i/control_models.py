from __future__ import annotations

CONTROL_MODEL_CONSISTENT_T2I = "consistent_t2i"
CONTROL_MODEL_TIFF_STORYBOARD = "novel_conti_tiff"
CONTROL_MODEL_CONSISTENCY_KEEPER = "consistency_keeper"

_CONTROL_MODEL_ALIASES = {
    CONTROL_MODEL_CONSISTENT_T2I: CONTROL_MODEL_CONSISTENT_T2I,
    "conisstent_t2i": CONTROL_MODEL_CONSISTENT_T2I,
    "image": CONTROL_MODEL_CONSISTENT_T2I,
    "image_generation": CONTROL_MODEL_CONSISTENT_T2I,
    CONTROL_MODEL_TIFF_STORYBOARD: CONTROL_MODEL_TIFF_STORYBOARD,
    "tiff": CONTROL_MODEL_TIFF_STORYBOARD,
    "tiff_storyboard": CONTROL_MODEL_TIFF_STORYBOARD,
    "conti_tiff": CONTROL_MODEL_TIFF_STORYBOARD,
    "storyboard_tiff": CONTROL_MODEL_TIFF_STORYBOARD,
    CONTROL_MODEL_CONSISTENCY_KEEPER: CONTROL_MODEL_CONSISTENCY_KEEPER,
    "consistency": CONTROL_MODEL_CONSISTENCY_KEEPER,
    "consistency_model": CONTROL_MODEL_CONSISTENCY_KEEPER,
    "visual_bible": CONTROL_MODEL_CONSISTENCY_KEEPER,
}


def normalize_control_model(requested: str, *, build_visual_bible: bool = False) -> str:
    if build_visual_bible:
        return CONTROL_MODEL_CONSISTENCY_KEEPER
    normalized = requested.strip().lower().replace("-", "_")
    try:
        return _CONTROL_MODEL_ALIASES[normalized]
    except KeyError as exc:
        supported = ", ".join(
            (
                CONTROL_MODEL_CONSISTENT_T2I,
                CONTROL_MODEL_TIFF_STORYBOARD,
                CONTROL_MODEL_CONSISTENCY_KEEPER,
            )
        )
        raise ValueError(
            f"Unknown control model '{requested}'. Supported values: {supported}."
        ) from exc
