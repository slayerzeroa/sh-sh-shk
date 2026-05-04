from __future__ import annotations

import re
from collections import Counter, defaultdict

from .models import WebNovelInput
from .visual_bible_schema import (
    APPEARANCE_CATEGORY_KEYWORDS,
    BEHAVIOR_KEYWORDS,
    CATEGORY_LABELS,
    CONFLICT_VARIANT_GROUPS,
    ENGLISH_STOPWORDS,
    FIXED_APPEARANCE_CATEGORIES,
    KOREAN_STOPWORDS,
    PROP_KEYWORDS,
    PRONOUN_AND_REFERENCE_CUES,
    ROLE_KEYWORDS,
    SCENE_SHIFT_KEYWORDS,
    WORLD_SECTION_KEYWORDS,
    CharacterPromptCard,
    CharacterRelationship,
    CharacterSceneState,
    CharacterVisualProfile,
    SceneState,
    TraitConflict,
    TraitEvidence,
    VisualBibleDocument,
    VisualBibleSection,
    VisualPromptPack,
)


class VisualBibleCompiler:
    def __init__(self, *, max_characters: int = 10, evidence_limit: int = 4) -> None:
        self.max_characters = max_characters
        self.evidence_limit = evidence_limit

    def compile(self, source: WebNovelInput) -> VisualBibleDocument:
        sentences = _split_sentences(source.source_text)
        character_scores = self._score_character_candidates(sentences)
        character_names = tuple(name for name, score in character_scores.most_common(self.max_characters) if score >= 2)
        explicit_mentions = self._build_explicit_mentions(sentences, character_names)
        aliases = self._build_aliases(sentences, explicit_mentions)
        mention_indices, mention_sources = self._build_mentions(sentences, character_names, explicit_mentions, aliases)
        relationships = self._build_relationships(mention_indices)
        characters = tuple(
            self._build_character_profile(
                name=name,
                sentences=sentences,
                mention_indices=mention_indices.get(name, ()),
                mention_sources=mention_sources.get(name, {}),
                aliases=aliases.get(name, ()),
                relationships=relationships.get(name, Counter()),
                all_mentions=mention_indices,
            )
            for name in character_names
        )
        world_sections = tuple(
            VisualBibleSection(
                title=title,
                top_terms=_collect_normalized_terms(sentences, keyword_map, limit=8),
                evidence=_filter_sentences_by_keywords(sentences, tuple(keyword_map.keys()), limit=self.evidence_limit),
            )
            for title, keyword_map in WORLD_SECTION_KEYWORDS
        )
        scene_states = self._build_scene_states(sentences, mention_indices)
        prompt_pack = self._build_prompt_pack(world_sections, characters, scene_states)
        conflicts = tuple(conflict for character in characters for conflict in character.conflicts)
        return VisualBibleDocument(
            episode_id=source.episode_id,
            title=source.title,
            synopsis=_summarize_text(source.source_text, limit=220),
            consistency_rules=(
                "머리, 눈매, 체형, 시그니처 액세서리는 canon prompt 기준으로 고정합니다.",
                "의상, 표정, 포즈, 손에 든 소품은 scene state에서만 덮어씁니다.",
                "충돌 신호와 confidence를 먼저 보고, 근거가 약한 값은 사람이 확정합니다.",
            ),
            world_sections=world_sections,
            characters=characters,
            scene_states=scene_states,
            prompt_pack=prompt_pack,
            conflicts=conflicts,
        )

    def _score_character_candidates(self, sentences: tuple[str, ...]) -> Counter[str]:
        scores: Counter[str] = Counter()
        for sentence in sentences:
            for raw_name in re.findall(r"(?:(?<=^)|(?<=[\s\"“'(\[]))([A-Za-z][A-Za-z' -]{1,24}|[가-힣]{2,8})\s*:", sentence):
                name = _normalize_name(raw_name)
                if _is_character_name(name):
                    scores[name] += 5
            for raw_name in re.findall(r"(?<![가-힣])([가-힣]{2,4})(?=(?:은|는|이|가|을|를|와|과|의|에게|한테|께서|도)\b)", sentence):
                name = _normalize_name(raw_name)
                if _is_character_name(name):
                    scores[name] += 1
            for raw_name in re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", sentence):
                name = _normalize_name(raw_name)
                if _is_character_name(name):
                    scores[name] += 1
        return scores

    def _build_explicit_mentions(self, sentences: tuple[str, ...], names: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
        return {name: tuple(index for index, sentence in enumerate(sentences) if _mentions(sentence, name)) for name in names}

    def _build_aliases(self, sentences: tuple[str, ...], explicit_mentions: dict[str, tuple[int, ...]]) -> dict[str, tuple[str, ...]]:
        aliases: dict[str, list[str]] = {name: [name] for name in explicit_mentions}
        for name, indices in explicit_mentions.items():
            if re.search(r"[A-Za-z]", name):
                aliases[name].append(name.split()[0])
            for index in indices:
                sentence = sentences[index]
                for role in ROLE_KEYWORDS:
                    if role.lower() in sentence.lower():
                        aliases[name].append(role)
        return {name: tuple(_dedupe_keep_order(items)) for name, items in aliases.items()}

    def _build_mentions(
        self,
        sentences: tuple[str, ...],
        names: tuple[str, ...],
        explicit_mentions: dict[str, tuple[int, ...]],
        aliases: dict[str, tuple[str, ...]],
    ) -> tuple[dict[str, tuple[int, ...]], dict[str, dict[int, str]]]:
        mention_sets = {name: set(indices) for name, indices in explicit_mentions.items()}
        mention_sources = {name: {index: "direct" for index in indices} for name, indices in explicit_mentions.items()}
        for index, sentence in enumerate(sentences):
            explicit = [name for name in names if index in explicit_mentions.get(name, ())]
            alias_hits = [name for name in names if name not in explicit and any(alias != name and _contains_alias(sentence, alias) for alias in aliases.get(name, ()))]
            if explicit:
                continue
            if len(alias_hits) == 1:
                mention_sets[alias_hits[0]].add(index)
                mention_sources.setdefault(alias_hits[0], {})[index] = "alias"
                continue
            if not _looks_character_focused(sentence):
                continue
            contextual = self._resolve_contextual(index, sentences, names, explicit_mentions, aliases)
            if len(contextual) == 1:
                mention_sets[contextual[0]].add(index)
                mention_sources.setdefault(contextual[0], {})[index] = "inferred_reference"
        return ({name: tuple(sorted(indices)) for name, indices in mention_sets.items()}, mention_sources)

    def _resolve_contextual(
        self,
        index: int,
        sentences: tuple[str, ...],
        names: tuple[str, ...],
        explicit_mentions: dict[str, tuple[int, ...]],
        aliases: dict[str, tuple[str, ...]],
    ) -> tuple[str, ...]:
        sentence = sentences[index]
        scores: Counter[str] = Counter()
        for neighbor in range(max(0, index - 2), min(len(sentences), index + 3)):
            if neighbor == index:
                continue
            neighbor_names = [name for name in names if neighbor in explicit_mentions.get(name, ())]
            if len(neighbor_names) == 1:
                scores[neighbor_names[0]] += max(1, 4 - abs(neighbor - index))
        if any(token.lower() in sentence.lower() for token in PRONOUN_AND_REFERENCE_CUES):
            for neighbor in range(max(0, index - 2), index):
                neighbor_names = [name for name in names if neighbor in explicit_mentions.get(name, ())]
                if len(neighbor_names) == 1:
                    scores[neighbor_names[0]] += 2
        for name in names:
            if any(alias != name and _contains_alias(sentence, alias) for alias in aliases.get(name, ())):
                scores[name] += 2
        top = scores.most_common(2)
        if not top:
            return ()
        if len(top) == 1 or top[0][1] > top[1][1]:
            return (top[0][0],)
        return ()

    def _build_relationships(self, mentions: dict[str, tuple[int, ...]]) -> dict[str, Counter[str]]:
        relationships: dict[str, Counter[str]] = defaultdict(Counter)
        sentence_map: dict[int, list[str]] = defaultdict(list)
        for name, indices in mentions.items():
            for index in indices:
                sentence_map[index].append(name)
        for names in sentence_map.values():
            unique = _dedupe_keep_order(names)
            for name in unique:
                for other in unique:
                    if name != other:
                        relationships[name][other] += 1
        return relationships

    def _build_character_profile(
        self,
        *,
        name: str,
        sentences: tuple[str, ...],
        mention_indices: tuple[int, ...],
        mention_sources: dict[int, str],
        aliases: tuple[str, ...],
        relationships: Counter[str],
        all_mentions: dict[str, tuple[int, ...]],
    ) -> CharacterVisualProfile:
        context_indices = _expand_context(name, sentences, mention_indices, all_mentions)
        fixed = {category: _collect_trait_evidence(sentences, context_indices, APPEARANCE_CATEGORY_KEYWORDS[category], mention_sources, self.evidence_limit) for category in FIXED_APPEARANCE_CATEGORIES}
        variable = {"outfit": _collect_trait_evidence(sentences, context_indices, APPEARANCE_CATEGORY_KEYWORDS["outfit"], mention_sources, self.evidence_limit)}
        appearance = {category: tuple(e.text for e in evidences) for category, evidences in {**fixed, **variable}.items()}
        confidence = {category: round(_average_confidence(evidences), 2) for category, evidences in {**fixed, **variable}.items()}
        evidence = tuple(_summarize_text(sentences[index], limit=180) for index in mention_indices[: self.evidence_limit])
        role_cues = _filter_sentences_by_keywords(tuple(sentences[index] for index in context_indices), ROLE_KEYWORDS, limit=self.evidence_limit) or evidence[:1]
        behavior_cues = _filter_sentences_by_keywords(tuple(sentences[index] for index in context_indices), BEHAVIOR_KEYWORDS, limit=self.evidence_limit) or evidence
        props = _collect_normalized_terms(tuple(sentences[index] for index in context_indices), PROP_KEYWORDS, limit=5)
        conflicts = _detect_conflicts(name, fixed)
        prompt_lock = _build_prompt_lock(fixed, props, conflicts)
        dynamic_aliases = [
            token
            for token in ("선배", "후배", "전학생", "학생회장", "소년", "소녀")
            if any(token in sentence for sentence in tuple(sentences[index] for index in context_indices))
        ]
        merged_aliases = tuple(alias for alias in _dedupe_keep_order(list(aliases) + dynamic_aliases) if alias != name)
        return CharacterVisualProfile(
            name=name,
            aliases=merged_aliases,
            importance_score=len(mention_indices),
            first_seen=evidence[0] if evidence else "",
            role_cues=role_cues,
            appearance=appearance,
            fixed_appearance=fixed,
            variable_appearance=variable,
            signature_props=props,
            behavior_cues=behavior_cues,
            relationships=tuple(CharacterRelationship(other_name=other, shared_scene_count=count, relation_strength=_relation_strength(count)) for other, count in relationships.most_common(4)),
            prompt_lock=prompt_lock,
            confidence_by_category=confidence,
            conflict_notes=tuple(conflict.note for conflict in conflicts),
            conflicts=conflicts,
            evidence=evidence,
        )

    def _build_scene_states(self, sentences: tuple[str, ...], mentions: dict[str, tuple[int, ...]]) -> tuple[SceneState, ...]:
        location_keywords = WORLD_SECTION_KEYWORDS[0][1]
        mood_keywords = WORLD_SECTION_KEYWORDS[-1][1]
        scene_states: list[SceneState] = []
        for scene_number, block in enumerate(_split_scene_blocks(sentences), start=1):
            block_sentences = tuple(sentences[index] for index in block)
            active = tuple(name for name in mentions if any(index in mentions[name] for index in block))
            states = []
            for name in active:
                own_sentences = tuple(sentences[index] for index in block if index in mentions[name]) or block_sentences
                states.append(
                    CharacterSceneState(
                        name=name,
                        expression_cues=_filter_sentences_by_keywords(own_sentences, BEHAVIOR_KEYWORDS, limit=2),
                        outfit_cues=_filter_sentences_by_keywords(own_sentences, APPEARANCE_CATEGORY_KEYWORDS["outfit"], limit=2),
                        prop_cues=_collect_normalized_terms(own_sentences, PROP_KEYWORDS, limit=3),
                        pose_cues=tuple(_summarize_text(sentence, limit=120) for sentence in own_sentences[:2]),
                        evidence=tuple(_summarize_text(sentence, limit=160) for sentence in own_sentences[:2]),
                    )
                )
            scene_states.append(
                SceneState(
                    scene_id=f"scene-{scene_number:02d}",
                    summary=_summarize_text(" ".join(block_sentences), limit=180),
                    location_terms=_collect_normalized_terms(block_sentences, location_keywords, limit=4),
                    mood_terms=_collect_normalized_terms(block_sentences, mood_keywords, limit=4),
                    active_characters=active,
                    character_states=tuple(states),
                    continuity_notes=tuple(_build_scene_notes(states)),
                    evidence=tuple(_summarize_text(sentence, limit=180) for sentence in block_sentences[: self.evidence_limit]),
                )
            )
        return tuple(scene_states)

    def _build_prompt_pack(
        self,
        world_sections: tuple[VisualBibleSection, ...],
        characters: tuple[CharacterVisualProfile, ...],
        scene_states: tuple[SceneState, ...],
    ) -> VisualPromptPack:
        motifs: list[str] = []
        for section in world_sections:
            if section.title in {"핵심 배경/장소", "반복 오브젝트/상징", "분위기/시각 모티프"}:
                motifs.extend(section.top_terms[:3])
        character_cards = []
        for character in characters:
            fixed_traits = []
            for category, evidences in character.fixed_appearance.items():
                if evidences:
                    fixed_traits.append(f"{CATEGORY_LABELS[category]}: {evidences[0].text}")
            if character.signature_props:
                fixed_traits.append(f"시그니처 소품: {', '.join(character.signature_props)}")
            scene_vars = tuple(e.text for e in character.variable_appearance.get("outfit", ())[:2])
            character_cards.append(CharacterPromptCard(name=character.name, positive_prompt=f"{character.name}: preserve {'; '.join(fixed_traits) if fixed_traits else 'canon baseline'}.", negative_prompt=f"Do not alter {character.name}'s face, hair silhouette, eye identity, body frame, or signature props.", fixed_traits=tuple(fixed_traits), scene_variables=scene_vars))
        scene_prompts = {scene.scene_id: f"Scene summary: {scene.summary} Location: {', '.join(scene.location_terms) if scene.location_terms else 'unspecified'}. Mood motifs: {', '.join(scene.mood_terms) if scene.mood_terms else 'carry global mood'}. Active characters: {', '.join(scene.active_characters) if scene.active_characters else 'environment only'}." for scene in scene_states}
        return VisualPromptPack(
            global_positive_prompt=f"Continuity-first webtoon rendering. Preserve canonical face structure, hair silhouette, body frame, and signature accessories. Reuse motifs: {', '.join(_dedupe_keep_order(motifs)) or 'grounded continuity motifs'}.",
            global_negative_prompt="Do not redesign faces between scenes. Do not change hair color, eye identity, body silhouette, or signature accessories unless the canon records a story event.",
            character_cards=tuple(character_cards),
            scene_prompts=scene_prompts,
        )


def _collect_trait_evidence(sentences: tuple[str, ...], indices: tuple[int, ...], keywords: tuple[str, ...], mention_sources: dict[int, str], limit: int) -> tuple[TraitEvidence, ...]:
    evidences: list[TraitEvidence] = []
    seen: set[str] = set()
    for index in indices:
        sentence = sentences[index]
        if not any(keyword.lower() in sentence.lower() for keyword in keywords):
            continue
        compact = _summarize_text(sentence, limit=180)
        if compact in seen:
            continue
        seen.add(compact)
        source_kind = mention_sources.get(index, "contextual")
        evidences.append(TraitEvidence(text=compact, confidence=_confidence_for_source(source_kind), source_kind=source_kind, sentence_index=index))
        if len(evidences) >= limit:
            break
    return tuple(evidences)


def _detect_conflicts(name: str, fixed_appearance: dict[str, tuple[TraitEvidence, ...]]) -> tuple[TraitConflict, ...]:
    conflicts: list[TraitConflict] = []
    for category, feature_groups in CONFLICT_VARIANT_GROUPS.items():
        texts = tuple(e.text for e in fixed_appearance.get(category, ()))
        for dimension, keyword_map in feature_groups.items():
            variants = _collect_detected_variants(texts, keyword_map)
            if len(variants) > 1:
                conflicts.append(TraitConflict(character_name=name, category=category, dimension=dimension, variants=variants, note=f"{name}의 {CATEGORY_LABELS.get(category, category)}에서 {', '.join(variants)}가 함께 감지되었습니다. 설정 변경인지 자동 추출 충돌인지 확인이 필요합니다."))
    return tuple(conflicts)


def _build_prompt_lock(fixed_appearance: dict[str, tuple[TraitEvidence, ...]], signature_props: tuple[str, ...], conflicts: tuple[TraitConflict, ...]) -> tuple[str, ...]:
    lines = [f"{CATEGORY_LABELS[category]}: {evidences[0].text}" for category, evidences in fixed_appearance.items() if evidences]
    if signature_props:
        lines.append(f"시그니처 소품: {', '.join(signature_props)}")
    if conflicts:
        lines.append("충돌 신호가 있으니 수동 검토 후 canon 확정")
    lines.append("머리/눈/실루엣/액세서리는 명시적 설정 변경 전까지 유지")
    return tuple(lines or ("원문 근거가 얕으므로 사람이 직접 고정 외형을 확정해야 합니다.",))


def _build_scene_notes(states: list[CharacterSceneState]) -> list[str]:
    notes = ["Character canon stays fixed while scene state updates outfit, expression, and handheld props."]
    for state in states:
        notes.append(f"{state.name}: {'scene outfit ' + state.outfit_cues[0] if state.outfit_cues else 'canon baseline look'}")
    return notes


def _expand_context(name: str, sentences: tuple[str, ...], mention_indices: tuple[int, ...], all_mentions: dict[str, tuple[int, ...]]) -> tuple[int, ...]:
    relevant = tuple(keyword for keywords in APPEARANCE_CATEGORY_KEYWORDS.values() for keyword in keywords) + BEHAVIOR_KEYWORDS + ROLE_KEYWORDS + tuple(PROP_KEYWORDS.keys())
    selected: list[int] = []
    for index in mention_indices:
        selected.append(index)
        for candidate in (index - 1, index + 1):
            if candidate < 0 or candidate >= len(sentences):
                continue
            sentence = sentences[candidate]
            if any(keyword.lower() in sentence.lower() for keyword in relevant) and not any(other != name and candidate in indices for other, indices in all_mentions.items()):
                selected.append(candidate)
    return tuple(int(value) for value in _dedupe_keep_order([str(index) for index in selected]))


def _split_sentences(text: str) -> tuple[str, ...]:
    return tuple(re.sub(r"\s+", " ", part).strip() for part in re.split(r"(?<=[.!?。！？])\s+|\n+", text.replace("\r\n", "\n").strip()) if part and part.strip())


def _split_scene_blocks(sentences: tuple[str, ...]) -> tuple[tuple[int, ...], ...]:
    blocks: list[list[int]] = []
    current: list[int] = []
    for index, sentence in enumerate(sentences):
        if current and (len(current) >= 3 or any(keyword.lower() in sentence.lower() for keyword in SCENE_SHIFT_KEYWORDS)):
            blocks.append(current)
            current = []
        current.append(index)
    if current:
        blocks.append(current)
    return tuple(tuple(block) for block in blocks)


def _collect_normalized_terms(sentences: tuple[str, ...], keyword_map: dict[str, str], limit: int) -> tuple[str, ...]:
    counts: Counter[str] = Counter()
    for sentence in sentences:
        for keyword, normalized in keyword_map.items():
            if keyword.lower() in sentence.lower():
                counts[normalized] += 1
    return tuple(term for term, _ in counts.most_common(limit))


def _collect_detected_variants(texts: tuple[str, ...], keyword_map: dict[str, str]) -> tuple[str, ...]:
    detected: list[str] = []
    for text in texts:
        for keyword, normalized in keyword_map.items():
            if keyword.lower() in text.lower() and normalized not in detected:
                detected.append(normalized)
    return tuple(detected)


def _filter_sentences_by_keywords(sentences: tuple[str, ...], keywords: tuple[str, ...], limit: int) -> tuple[str, ...]:
    matched: list[str] = []
    for sentence in sentences:
        if any(keyword.lower() in sentence.lower() for keyword in keywords):
            compact = _summarize_text(sentence, limit=180)
            if compact not in matched:
                matched.append(compact)
        if len(matched) >= limit:
            break
    return tuple(matched)


def _looks_character_focused(sentence: str) -> bool:
    relevant = tuple(keyword for keywords in APPEARANCE_CATEGORY_KEYWORDS.values() for keyword in keywords) + BEHAVIOR_KEYWORDS + ROLE_KEYWORDS + tuple(PROP_KEYWORDS.keys()) + PRONOUN_AND_REFERENCE_CUES
    return any(keyword.lower() in sentence.lower() for keyword in relevant)


def _mentions(sentence: str, name: str) -> bool:
    if re.search(r"[가-힣]", name):
        return name in sentence
    return re.search(rf"\b{re.escape(name)}\b", sentence) is not None


def _contains_alias(sentence: str, alias: str) -> bool:
    return _mentions(sentence, alias)


def _normalize_name(name: str) -> str:
    compact = re.sub(r"\s+", " ", name).strip(" \"'“”‘’.,!?()[]{}")
    if re.search(r"[가-힣]", compact):
        return compact
    return " ".join(part.capitalize() for part in compact.split())


def _is_character_name(name: str) -> bool:
    if not name:
        return False
    if re.search(r"[가-힣]", name):
        return name not in KOREAN_STOPWORDS
    return name not in ENGLISH_STOPWORDS


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _summarize_text(text: str, limit: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact if len(compact) <= limit else compact[: limit - 3].rstrip() + "..."


def _confidence_for_source(source_kind: str) -> float:
    return {"direct": 0.95, "alias": 0.8, "contextual": 0.72, "inferred_reference": 0.6}.get(source_kind, 0.55)


def _average_confidence(evidences: tuple[TraitEvidence, ...]) -> float:
    return sum(e.confidence for e in evidences) / len(evidences) if evidences else 0.0


def _relation_strength(count: int) -> str:
    if count >= 3:
        return "high"
    if count == 2:
        return "medium"
    return "supporting"
