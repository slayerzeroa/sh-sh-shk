from __future__ import annotations

from .memory import ContinuityState
from .models import CharacterBible, PanelSpec, StyleBible, TraitConstraint


def build_demo_style() -> StyleBible:
    return StyleBible(
        style_id="painterly-webtoon-v1",
        title="Painterly Webtoon",
        positive_traits=(
            "clean ink lines",
            "limited but bold color palette",
            "cinematic Korean webtoon paneling",
            "consistent rim light",
        ),
        negative_traits=("photorealism", "3d render", "western superhero anatomy"),
        render_rules=(
            "preserve face proportions across panels",
            "preserve costume silhouette unless script says otherwise",
            "preserve brush texture and line weight",
        ),
    )


def build_demo_characters() -> dict[str, CharacterBible]:
    return {
        "haeun": CharacterBible(
            character_id="haeun",
            display_name="Yoon Haeun",
            core_prompt="18-year-old heroine, calm eyes, reserved body language",
            immutable_traits=(
                TraitConstraint("hair_color", "black"),
                TraitConstraint("hair_style", "long straight hair with blunt bangs"),
                TraitConstraint("eye_color", "dark brown"),
                TraitConstraint("signature_item", "silver hairpin shaped like a crescent moon"),
            ),
            mutable_traits=(TraitConstraint("expression", "can vary by scene", level="soft"),),
            negative_traits=("short hair", "blue eyes", "missing silver hairpin"),
            reference_asset_ids=("canon-haeun-front", "canon-haeun-profile"),
        ),
        "dohyun": CharacterBible(
            character_id="dohyun",
            display_name="Kang Dohyun",
            core_prompt="19-year-old male lead, tall frame, restrained expression",
            immutable_traits=(
                TraitConstraint("hair_color", "dark ash brown"),
                TraitConstraint("hair_style", "slightly tousled short hair"),
                TraitConstraint("eye_color", "gray brown"),
                TraitConstraint("signature_item", "school blazer with gold crest"),
            ),
            negative_traits=("red hair", "hoodie", "different school uniform crest"),
            reference_asset_ids=("canon-dohyun-front",),
        ),
    }


def build_demo_panel() -> PanelSpec:
    return PanelSpec(
        panel_id="ep01-sc01-p01",
        narrative="Haeun meets Dohyun in the rain-soaked school corridor for the first time.",
        characters=("haeun", "dohyun"),
        location="high school corridor at dusk, rain outside the windows",
        shot_type="medium two-shot",
        emotion="awkward tension",
        action="they stop mid-step and make eye contact",
        props=("wet umbrella", "school lockers"),
        dialogue=("Haeun: ...", "Dohyun: You dropped this."),
    )


def build_demo_state() -> ContinuityState:
    return ContinuityState(project_id="novel-to-webtoon-demo", style_id="painterly-webtoon-v1")
