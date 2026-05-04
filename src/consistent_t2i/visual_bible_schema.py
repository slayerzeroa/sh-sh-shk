from __future__ import annotations

from dataclasses import dataclass

KOREAN_STOPWORDS = {
    "그리고",
    "하지만",
    "그러나",
    "그녀",
    "그는",
    "그녀는",
    "그들",
    "사람",
    "남자",
    "여자",
    "학생",
    "학교",
    "교실",
    "복도",
    "옥상",
    "거리",
    "도시",
    "마을",
    "왕국",
    "오늘",
    "이번",
    "그때",
    "정말",
    "모두",
    "우리",
}

ENGLISH_STOPWORDS = {
    "The",
    "A",
    "An",
    "He",
    "She",
    "They",
    "We",
    "I",
    "You",
    "It",
    "This",
    "That",
    "These",
    "Those",
    "Today",
    "Tonight",
    "Morning",
    "Evening",
    "School",
    "Hallway",
    "Classroom",
    "Street",
    "Rain",
}

APPEARANCE_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "hair": (
        "머리",
        "머리카락",
        "앞머리",
        "단발",
        "장발",
        "긴 생머리",
        "짧은 머리",
        "생머리",
        "웨이브",
        "포니테일",
        "흑발",
        "금발",
        "은발",
        "갈색 머리",
        "검은 머리",
        "black hair",
        "blonde",
        "silver hair",
        "bangs",
        "ponytail",
        "braid",
        "bob cut",
        "long hair",
        "short hair",
    ),
    "face_eyes": (
        "눈",
        "눈동자",
        "눈매",
        "속눈썹",
        "얼굴",
        "입술",
        "미소",
        "미간",
        "피부",
        "갈색 눈",
        "회색 눈",
        "푸른 눈",
        "brown eyes",
        "blue eyes",
        "grey eyes",
        "gray eyes",
        "freckles",
    ),
    "body": (
        "키",
        "체격",
        "체형",
        "어깨",
        "손",
        "왜소",
        "마른",
        "탄탄",
        "근육",
        "큰 키",
        "작은 키",
        "tall",
        "short",
        "slim",
        "athletic",
        "broad shoulders",
    ),
    "outfit": (
        "교복",
        "셔츠",
        "재킷",
        "자켓",
        "코트",
        "가디건",
        "치마",
        "바지",
        "부츠",
        "구두",
        "후드",
        "로브",
        "robe",
        "uniform",
        "coat",
        "jacket",
        "dress",
        "skirt",
        "shirt",
        "cardigan",
    ),
    "accessories": (
        "반지",
        "목걸이",
        "귀걸이",
        "머리핀",
        "안경",
        "시계",
        "리본",
        "장갑",
        "bracelet",
        "glasses",
        "hairpin",
        "earring",
        "ribbon",
    ),
}

FIXED_APPEARANCE_CATEGORIES = ("hair", "face_eyes", "body", "accessories")
VARIABLE_APPEARANCE_CATEGORIES = ("outfit",)

ROLE_KEYWORDS = (
    "학생",
    "전학생",
    "선배",
    "후배",
    "주인공",
    "왕자",
    "공주",
    "기사",
    "마법사",
    "의사",
    "형사",
    "교사",
    "학생회",
    "학생회장",
    "반장",
    "captain",
    "student",
    "prince",
    "princess",
    "knight",
    "mage",
    "detective",
)

BEHAVIOR_KEYWORDS = (
    "웃",
    "울",
    "노려",
    "바라",
    "주저",
    "망설",
    "움찔",
    "긴장",
    "당황",
    "분노",
    "미소",
    "한숨",
    "손을 떨",
    "stare",
    "smile",
    "laugh",
    "cry",
    "hesitate",
    "flinch",
    "tense",
    "anger",
)

PROP_KEYWORDS: dict[str, str] = {
    "우산": "umbrella",
    "검": "sword",
    "책": "book",
    "머리핀": "hairpin",
    "안경": "glasses",
    "목걸이": "necklace",
    "반지": "ring",
    "편지": "letter",
    "스마트폰": "smartphone",
    "핸드폰": "smartphone",
    "phone": "smartphone",
    "sword": "sword",
    "book": "book",
    "ring": "ring",
    "letter": "letter",
    "umbrella": "umbrella",
}

WORLD_SECTION_KEYWORDS: tuple[tuple[str, dict[str, str]], ...] = (
    ("핵심 배경/장소", {"학교": "school", "교실": "classroom", "복도": "corridor", "옥상": "rooftop", "거리": "street", "도시": "city", "마을": "village", "왕국": "kingdom", "궁": "palace", "숲": "forest", "기숙사": "dormitory", "병원": "hospital", "카페": "cafe", "school": "school", "classroom": "classroom", "corridor": "corridor", "rooftop": "rooftop", "street": "street", "city": "city"}),
    ("소속/집단/직함", {"학생회": "student council", "가문": "family house", "기사단": "knight order", "길드": "guild", "동아리": "club", "반": "class group", "팀": "team", "회사": "company", "왕실": "royalty", "student council": "student council", "guild": "guild", "academy": "academy", "company": "company", "team": "team"}),
    ("세계 규칙/능력", {"마법": "magic", "능력": "special ability", "저주": "curse", "축복": "blessing", "계약": "contract", "회귀": "regression", "시스템": "system", "각성": "awakening", "검기": "sword aura", "magic": "magic", "curse": "curse", "contract": "contract", "system": "system", "awakening": "awakening"}),
    ("반복 오브젝트/상징", {"우산": "umbrella", "검": "sword", "책": "book", "머리핀": "hairpin", "안경": "glasses", "열쇠": "key", "편지": "letter", "반지": "ring", "clock": "clock", "umbrella": "umbrella", "sword": "sword", "book": "book", "ring": "ring", "letter": "letter"}),
    ("분위기/시각 모티프", {"비": "rain", "눈": "snow or eyes motif", "밤": "night", "새벽": "dawn", "안개": "fog", "네온": "neon", "붉": "red accent", "푸른": "blue accent", "빛": "glow", "rain": "rain", "night": "night", "fog": "fog", "neon": "neon", "glow": "glow"}),
)

PRONOUN_AND_REFERENCE_CUES = (
    "그녀",
    "그는",
    "그녀는",
    "그 애",
    "그 남자",
    "그 여자",
    "소년",
    "소녀",
    "선배",
    "후배",
    "전학생",
    "학생회장",
    "he",
    "she",
    "his",
    "her",
)

SCENE_SHIFT_KEYWORDS = ("다음 날", "잠시 후", "그날 밤", "다음 순간", "그 후", "한편", "Meanwhile", "Later", "The next day")

CONFLICT_VARIANT_GROUPS: dict[str, dict[str, dict[str, str]]] = {
    "hair": {
        "hair_color": {"흑발": "black hair", "검은 머리": "black hair", "검은": "black hair", "금발": "blonde hair", "은발": "silver hair", "갈색 머리": "brown hair", "black hair": "black hair", "blonde": "blonde hair", "silver hair": "silver hair", "brown hair": "brown hair"},
        "hair_length": {"장발": "long hair", "긴 생머리": "long hair", "긴 머리": "long hair", "단발": "short bob", "짧은 머리": "short hair", "long hair": "long hair", "bob cut": "short bob", "short hair": "short hair"},
    },
    "face_eyes": {"eye_color": {"회색 눈": "grey eyes", "갈색 눈": "brown eyes", "푸른 눈": "blue eyes", "grey eyes": "grey eyes", "gray eyes": "grey eyes", "brown eyes": "brown eyes", "blue eyes": "blue eyes"}},
    "body": {"body_frame": {"큰 키": "tall", "작은 키": "short", "왜소": "small frame", "마른": "slim", "탄탄": "athletic", "근육": "muscular", "tall": "tall", "short": "short", "slim": "slim", "athletic": "athletic"}},
}

CATEGORY_LABELS = {"hair": "머리", "face_eyes": "얼굴/눈", "body": "체형", "outfit": "의상", "accessories": "액세서리"}


@dataclass(frozen=True)
class TraitEvidence:
    text: str
    confidence: float
    source_kind: str
    sentence_index: int


@dataclass(frozen=True)
class TraitConflict:
    character_name: str
    category: str
    dimension: str
    variants: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class CharacterRelationship:
    other_name: str
    shared_scene_count: int
    relation_strength: str = "supporting"


@dataclass(frozen=True)
class CharacterSceneState:
    name: str
    expression_cues: tuple[str, ...]
    outfit_cues: tuple[str, ...]
    prop_cues: tuple[str, ...]
    pose_cues: tuple[str, ...]
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class SceneState:
    scene_id: str
    summary: str
    location_terms: tuple[str, ...]
    mood_terms: tuple[str, ...]
    active_characters: tuple[str, ...]
    character_states: tuple[CharacterSceneState, ...]
    continuity_notes: tuple[str, ...]
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class CharacterPromptCard:
    name: str
    positive_prompt: str
    negative_prompt: str
    fixed_traits: tuple[str, ...]
    scene_variables: tuple[str, ...]


@dataclass(frozen=True)
class VisualPromptPack:
    global_positive_prompt: str
    global_negative_prompt: str
    character_cards: tuple[CharacterPromptCard, ...]
    scene_prompts: dict[str, str]


@dataclass(frozen=True)
class CharacterVisualProfile:
    name: str
    aliases: tuple[str, ...]
    importance_score: int
    first_seen: str
    role_cues: tuple[str, ...]
    appearance: dict[str, tuple[str, ...]]
    fixed_appearance: dict[str, tuple[TraitEvidence, ...]]
    variable_appearance: dict[str, tuple[TraitEvidence, ...]]
    signature_props: tuple[str, ...]
    behavior_cues: tuple[str, ...]
    relationships: tuple[CharacterRelationship, ...]
    prompt_lock: tuple[str, ...]
    confidence_by_category: dict[str, float]
    conflict_notes: tuple[str, ...]
    conflicts: tuple[TraitConflict, ...]
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class VisualBibleSection:
    title: str
    top_terms: tuple[str, ...]
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class VisualBibleDocument:
    episode_id: str
    title: str
    synopsis: str
    consistency_rules: tuple[str, ...]
    world_sections: tuple[VisualBibleSection, ...]
    characters: tuple[CharacterVisualProfile, ...]
    scene_states: tuple[SceneState, ...]
    prompt_pack: VisualPromptPack
    conflicts: tuple[TraitConflict, ...]


@dataclass(frozen=True)
class VisualBibleBuildResult:
    episode_id: str
    title: str
    workspace_dir: str
    source_copy_path: str
    json_path: str
    world_path: str
    characters_path: str
    appearance_path: str
    scene_state_path: str
    prompt_pack_path: str
    conflict_path: str
    character_count: int
    scene_count: int
    world_section_count: int
    conflict_count: int
