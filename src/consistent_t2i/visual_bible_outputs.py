from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .models import WebNovelInput
from .visual_bible_engine import VisualBibleCompiler
from .visual_bible_schema import (
    CATEGORY_LABELS,
    FIXED_APPEARANCE_CATEGORIES,
    VisualBibleBuildResult,
    VisualBibleDocument,
)


def build_visual_bible_workspace(
    source: WebNovelInput,
    *,
    workspace_dir: str | Path = "visual_bible",
    max_characters: int = 10,
    evidence_limit: int = 4,
) -> tuple[VisualBibleDocument, VisualBibleBuildResult]:
    compiler = VisualBibleCompiler(max_characters=max_characters, evidence_limit=evidence_limit)
    document = compiler.compile(source)
    workspace_path = Path(workspace_dir)
    input_dir = workspace_path / "input"
    output_dir = workspace_path / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = _slugify(source.title or source.episode_id)
    source_copy_path = input_dir / f"{slug}.txt"
    json_path = output_dir / f"{slug}.visual-bible.json"
    world_path = output_dir / f"{slug}.world.md"
    characters_path = output_dir / f"{slug}.characters.md"
    appearance_path = output_dir / f"{slug}.appearance-locks.md"
    scene_state_path = output_dir / f"{slug}.scene-state.md"
    prompt_pack_path = output_dir / f"{slug}.prompt-pack.json"
    conflict_path = output_dir / f"{slug}.conflicts.md"

    source_copy_path.write_text(source.source_text.strip() + "\n", encoding="utf-8")
    json_path.write_text(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(), "document": asdict(document)}, indent=2, ensure_ascii=False), encoding="utf-8")
    world_path.write_text(render_world_markdown(document), encoding="utf-8")
    characters_path.write_text(render_characters_markdown(document), encoding="utf-8")
    appearance_path.write_text(render_appearance_locks_markdown(document), encoding="utf-8")
    scene_state_path.write_text(render_scene_states_markdown(document), encoding="utf-8")
    prompt_pack_path.write_text(json.dumps(asdict(document.prompt_pack), indent=2, ensure_ascii=False), encoding="utf-8")
    conflict_path.write_text(render_conflicts_markdown(document), encoding="utf-8")

    result = VisualBibleBuildResult(
        episode_id=source.episode_id,
        title=source.title,
        workspace_dir=str(workspace_path.resolve()),
        source_copy_path=str(source_copy_path.resolve()),
        json_path=str(json_path.resolve()),
        world_path=str(world_path.resolve()),
        characters_path=str(characters_path.resolve()),
        appearance_path=str(appearance_path.resolve()),
        scene_state_path=str(scene_state_path.resolve()),
        prompt_pack_path=str(prompt_pack_path.resolve()),
        conflict_path=str(conflict_path.resolve()),
        character_count=len(document.characters),
        scene_count=len(document.scene_states),
        world_section_count=sum(1 for section in document.world_sections if section.top_terms or section.evidence),
        conflict_count=len(document.conflicts),
    )
    return document, result


def render_world_markdown(document: VisualBibleDocument) -> str:
    lines = [f"# {document.title} 세계관 정리", "", "## 작품 개요", document.synopsis or "(요약 가능한 문장이 부족합니다.)", "", "## 일관성 운용 원칙"]
    lines.extend(f"- {rule}" for rule in document.consistency_rules)
    for section in document.world_sections:
        lines.extend(["", f"## {section.title}", f"- 추출 키워드: {', '.join(section.top_terms) if section.top_terms else '(근거 부족)'}", "- 근거 문장:"])
        lines.extend(f"  - {sentence}" for sentence in section.evidence) if section.evidence else lines.append("  - 추출된 근거 문장이 없습니다.")
    return "\n".join(lines) + "\n"


def render_characters_markdown(document: VisualBibleDocument) -> str:
    lines = [f"# {document.title} 캐릭터 canon", "", f"- 감지된 캐릭터 수: {len(document.characters)}", "- 고정 canon과 가변 scene state 후보를 분리합니다.", ""]
    if not document.characters:
        return "\n".join(lines + ["자동 감지된 주요 캐릭터가 없습니다."]) + "\n"
    for character in document.characters:
        lines.extend([f"## {character.name}", f"- 중요도 점수: {character.importance_score}", f"- 첫 등장/대표 문장: {character.first_seen or '(근거 부족)'}", f"- 별칭/호칭 추적: {', '.join(character.aliases) if character.aliases else '(없음)'}", f"- 역할/정체성: {' | '.join(character.role_cues) if character.role_cues else '(근거 부족)'}", "### 고정 canon"])
        for category in FIXED_APPEARANCE_CATEGORIES:
            evidences = character.fixed_appearance.get(category, ())
            confidence = character.confidence_by_category.get(category, 0.0)
            lines.append(f"- {CATEGORY_LABELS[category]} (confidence {confidence:.2f}): {' | '.join(e.text for e in evidences) if evidences else '(근거 부족)'}")
        lines.extend(["### 가변 scene state", f"- 의상: {' | '.join(e.text for e in character.variable_appearance.get('outfit', ())) if character.variable_appearance.get('outfit') else '(근거 부족)'}", f"- 행동/감정 큐: {' | '.join(character.behavior_cues) if character.behavior_cues else '(근거 부족)'}", f"- 시그니처 소품: {', '.join(character.signature_props) if character.signature_props else '(근거 부족)'}", f"- conflict: {' | '.join(character.conflict_notes) if character.conflict_notes else '없음'}", f"- prompt lock: {' | '.join(character.prompt_lock)}", ""])
    return "\n".join(lines).rstrip() + "\n"


def render_appearance_locks_markdown(document: VisualBibleDocument) -> str:
    lines = [f"# {document.title} 외형 고정 카드", "", "이미지 모델에는 고정 canon을 우선하고, 의상/표정/손소품은 scene state에서만 바꾸세요.", ""]
    if not document.characters:
        return "\n".join(lines + ["자동 감지된 캐릭터가 없어 외형 고정 카드를 만들지 못했습니다."]) + "\n"
    for character in document.characters:
        lines.extend([f"## {character.name}", "### 고정할 것"])
        lines.extend(f"- {line}" for line in character.prompt_lock)
        lines.extend(["### 가변값", f"- 의상: {' | '.join(e.text for e in character.variable_appearance.get('outfit', ())) if character.variable_appearance.get('outfit') else '장면 전환 시 바뀔 수 있음'}", f"- 표정/행동: {' | '.join(character.behavior_cues) if character.behavior_cues else '감정선에 따라 변경 가능'}", ""])
    return "\n".join(lines).rstrip() + "\n"


def render_scene_states_markdown(document: VisualBibleDocument) -> str:
    lines = [f"# {document.title} scene state", "", "scene state는 canon을 깨지 않으면서 장면별 의상, 표정, 소품 상태만 덮어씁니다.", ""]
    if not document.scene_states:
        return "\n".join(lines + ["추출된 scene state가 없습니다."]) + "\n"
    for scene in document.scene_states:
        lines.extend([f"## {scene.scene_id}", f"- 요약: {scene.summary}", f"- 장소 키워드: {', '.join(scene.location_terms) if scene.location_terms else '(근거 부족)'}", f"- 분위기 키워드: {', '.join(scene.mood_terms) if scene.mood_terms else '(근거 부족)'}", f"- 활성 캐릭터: {', '.join(scene.active_characters) if scene.active_characters else '(없음)'}", f"- 연속성 노트: {' | '.join(scene.continuity_notes)}", "- 캐릭터 상태:"])
        if scene.character_states:
            for state in scene.character_states:
                lines.extend([f"  - {state.name}", f"    outfit: {' | '.join(state.outfit_cues) if state.outfit_cues else '(canon baseline)'}", f"    expression: {' | '.join(state.expression_cues) if state.expression_cues else '(근거 부족)'}", f"    props: {', '.join(state.prop_cues) if state.prop_cues else '(근거 부족)'}"])
        else:
            lines.append("  - 장면 중심 환경 묘사")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_conflicts_markdown(document: VisualBibleDocument) -> str:
    lines = [f"# {document.title} conflict report", "", "고정 canon 안에서 서로 충돌하는 외형 신호를 감지하면 여기 기록합니다.", ""]
    if not document.conflicts:
        return "\n".join(lines + ["감지된 외형 충돌이 없습니다."]) + "\n"
    for conflict in document.conflicts:
        lines.extend([f"## {conflict.character_name} / {CATEGORY_LABELS.get(conflict.category, conflict.category)} / {conflict.dimension}", f"- variants: {', '.join(conflict.variants)}", f"- note: {conflict.note}", ""])
    return "\n".join(lines).rstrip() + "\n"


def format_visual_bible_summary(result: VisualBibleBuildResult) -> str:
    return "\n".join(
        [
            f"Visual bible: {result.title} ({result.episode_id})",
            f"Workspace: {result.workspace_dir}",
            f"Source copy: {result.source_copy_path}",
            f"JSON bible: {result.json_path}",
            f"World guide: {result.world_path}",
            f"Character canon: {result.characters_path}",
            f"Appearance locks: {result.appearance_path}",
            f"Scene states: {result.scene_state_path}",
            f"Prompt pack: {result.prompt_pack_path}",
            f"Conflict report: {result.conflict_path}",
            f"Detected characters: {result.character_count}",
            f"Detected scenes: {result.scene_count}",
            f"World sections with evidence: {result.world_section_count}",
            f"Conflicts: {result.conflict_count}",
        ]
    )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9가-힣_-]+", "-", value).strip("-")
    return slug or "visual-bible"
