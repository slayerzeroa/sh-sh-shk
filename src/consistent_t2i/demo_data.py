from __future__ import annotations

from .memory import ContinuityState
from .models import (
    CharacterBible,
    ModelEndpointConfig,
    PanelSpec,
    StyleBible,
    TraitConstraint,
    WebNovelInput,
)


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
            aliases=("haeun", "yoon haeun", "hae-un", "해은", "윤해은"),
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
            aliases=("dohyun", "kang dohyun", "do hyun", "도현", "강도현"),
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


def build_demo_novel() -> WebNovelInput:
    return WebNovelInput(
        episode_id="ep01",
        title="Rain Corridor Encounter",
        source_text=(
            "A late spring rain drenches the high school building as Haeun hurries down the corridor, "
            "hugging a notebook against her chest. She nearly collides with Dohyun near the lockers, "
            "and both of them freeze for a breath too long.\n\n"
            "Water drips from the umbrella in Dohyun's hand. Haeun notices the silver hairpin has slipped, "
            "but before she can speak he bends down first. Dohyun: You dropped this.\n\n"
            "Their eyes meet in the dim window light, and the whole hallway feels quieter than it should."
        ),
        genre="romance drama",
        tone="awkward tension",
        default_location="high school corridor at dusk",
        primary_characters=("haeun", "dohyun"),
        target_panel_count=3,
        adaptation_rules=(
            "Keep early panels readable for vertical-scroll webtoon pacing.",
            "Favor romantic tension over exposition.",
        ),
    )


def build_demo_image_model_config() -> ModelEndpointConfig:
    return ModelEndpointConfig(
        stage="refine",
        provider="openai",
        model="gpt-image-1.5",
        api_key="",
        api_key_env="OPENAI_API_KEY",
        size="1024x1024",
        quality="low",
        output_format="png",
        output_dir="outputs/demo",
        notes=(
            "API key intentionally left blank until real deployment.",
            "Uses the latest GPT Image model family by default.",
            "Defaults are tuned for lower image-generation cost.",
        ),
    )


def build_demo_gemini_image_model_config() -> ModelEndpointConfig:
    return ModelEndpointConfig(
        stage="refine",
        provider="gemini",
        model="gemini-2.5-flash-image",
        api_key="",
        api_key_env="GEMINI_API_KEY",
        size="1024x1024",
        quality="standard",
        output_format="png",
        output_dir="outputs/gemini",
        notes=(
            "Native Gemini image generation model.",
            "Configured to load the API key from .env or the environment.",
        ),
    )


def build_demo_bubble_model_config() -> ModelEndpointConfig:
    return ModelEndpointConfig(
        stage="bubble",
        provider="local-llm",
        model="gemma4",
        api_key="",
        max_output_tokens=120,
        notes=(
            "Low-token bubble fill stage.",
            "Can be swapped for other compact dialogue models later.",
        ),
    )
