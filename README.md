# sh-sh-shk

웹소설을 웹툰으로 바꾸는 과정에서 가장 큰 문제는 "같은 인물인데 매 컷마다 다른 사람처럼 보이는 현상"입니다.
이 저장소는 그 문제를 줄이기 위해, 단순 prompt 문자열이 아니라 `연속성 메모리 + 캐릭터/스타일 바이블 + 결정적 요청 조립 + 드리프트 검증 + 화자 추론 + SVG 후처리 말풍선 합성`을 중심으로 한 Python 코어를 제공합니다.
실시간 운영을 염두에 두고, `hot path`와 `cold path`를 분리한 경량 구조를 기본으로 둡니다.
이제 여기에 `웹소설 -> 텍스트 콘티 -> 졸라맨 그림 콘티 -> 실제 그림 고도화 -> 저토큰 말풍선 채우기 -> SVG 후처리 말풍선 합성` 파이프라인이 추가되어, 웹툰 제작 초안을 단계별로 조립할 수 있습니다.

## 목표

- 이전에 생성된 그림체가 다음 컷에서 무너지지 않게 유지
- 웹소설 원문을 먼저 패널 단위 콘티로 구조화
- 텍스트 콘티를 졸라맨 레벨의 그림 콘티로 바꿔 카메라/구도/말풍선 위치를 선고정
- 그림 콘티를 바탕으로 실제 그림 생성 요청을 안정적으로 고도화
- Gemma4 같은 경량 모델로 낮은 토큰 비용의 말풍선 채우기 요청을 따로 분리
- 이미지 모델이 글자를 직접 그리지 않도록 하고, 화자 추론 후 후처리 SVG 오버레이로 안정적으로 대사를 합성
- 캐릭터의 얼굴, 머리색, 복장 실루엣, 시그니처 소품 같은 핵심 속성을 고정
- 웹소설 장면 설명을 패널 단위 생성 요청으로 바꿀 때 항상 같은 규칙으로 prompt/reference/seed를 조립
- 생성 후 관측된 속성으로 외형 드리프트를 감지하고 재생성 여부를 판단

## 구조

- `src/consistent_t2i/models.py`
  핵심 도메인 모델. 스타일 바이블, 캐릭터 바이블, 패널 스펙, 생성 요청, 드리프트 리포트를 정의합니다.
- `src/consistent_t2i/memory.py`
  승인된 생성 결과를 이후 컷의 참조 이미지로 재사용하는 연속성 상태 저장소입니다.
- `src/consistent_t2i/planner.py`
  스타일/캐릭터 고정 규칙을 합쳐 결정적인 생성 요청을 만듭니다.
- `src/consistent_t2i/pipeline.py`
  1단계 웹소설 콘티화, 2단계 졸라맨 그림 콘티화, 3단계 그림 고도화 요청 생성, 4단계 저토큰 말풍선 요청 생성, 화자 추론, 5단계 말풍선 후처리 연결을 담당합니다.
- `src/consistent_t2i/bubble_renderer.py`
  Stage 5 SVG 오버레이 렌더러입니다. 생성된 이미지 위에 말풍선과 대사를 후처리로 합성합니다.
- `src/consistent_t2i/providers.py`
  실제 provider payload를 조립하며, 숨겨진 system prompt가 항상 포함되도록 강제합니다. 이미지 모델용 payload와 말풍선 모델용 payload도 분리되어 있습니다.
- `src/consistent_t2i/paths.py`
  실시간 생성 경로(`hot path`)와 검토/복구 판단 경로(`cold path`)를 분리합니다.
- `src/consistent_t2i/drift.py`
  관측된 외형 속성을 캐릭터 바이블과 비교해 외형 붕괴를 점수화합니다.
- `src/consistent_t2i/engine.py`
  계획, 생성, 승인, 재사용을 묶는 얇은 오케스트레이션 계층입니다.

## 핵심 아이디어

1. `Stage 1 - Storyboard`
   웹소설 원문을 패널 단위 텍스트 콘티(`StoryboardPlan`)로 바꿉니다.
2. `Stage 2 - Sketch Storyboard`
   텍스트 콘티를 졸라맨 구도 가이드(`SketchStoryboard`)로 바꿉니다.
3. `Stage 3 - Refine`
   졸라맨 콘티와 continuity rule을 합쳐 실제 그림 생성 요청(`GenerationRequest`)을 만듭니다.
4. `Stage 4 - Bubble Fill`
   Gemma4 같은 경량 모델용 저토큰 말풍선 요청(`BubbleFillRequest`)을 만듭니다.
5. `Speaker Attribution`
   대사 speaker label을 캐릭터 alias와 패널 문맥으로 정규화해, 어떤 캐릭터에게 말풍선을 붙일지 결정합니다.
6. `Stage 5 - Bubble Overlay`
   생성된 이미지에 SVG 기반 말풍선과 대사를 후처리로 안정적으로 합성합니다.
6. `Hot Path`
   패널 요청 조립, 이미지 생성, 승인된 참조 재사용만 담당합니다.
7. `Cold Path`
   드리프트 판정, 복구 필요 여부 판단, canon 갱신 보류 같은 무거운 결정을 담당합니다.
8. `Style Bible`
   작화 질감, 선 두께, 색감, 연출 규칙을 고정합니다.
9. `Character Bible`
   절대 변하면 안 되는 속성과, 장면에 따라 바뀌어도 되는 속성을 분리합니다.
10. `Reference Selection`
   최신 승인 컷과 정본(canon) 레퍼런스를 함께 사용합니다.
11. `Deterministic Request Building`
   prompt 순서, negative prompt, seed bundle을 항상 같은 방식으로 조립합니다.
12. `Drift Detection`
   생성 결과를 바로 canon에 편입하지 않고, 먼저 외형 붕괴 여부를 검사합니다.
13. `Hidden System Prompt Guardrail`
   provider에 전달되는 숨겨진 system prompt로 그림체 변경, 얼굴 구조 변경, 핵심 외형 변경 요청을 기본적으로 거부합니다.

## 빠른 실행

```powershell
$env:PYTHONPATH='src'; python -m consistent_t2i
```

직접 텍스트를 붙여 넣어 테스트하려면:

```powershell
$env:PYTHONPATH='src'
python -m consistent_t2i --interactive --title "내 테스트" --episode-id "ep-custom" --panel-count 4
```

실행 후 웹소설 텍스트를 붙여 넣고, 마지막 줄에 `END`만 입력하면 됩니다.

한 줄/짧은 텍스트는 바로 인자로도 넣을 수 있습니다.

```powershell
$env:PYTHONPATH='src'
python -m consistent_t2i --text "비 오는 복도에서 두 사람이 마주친다. 도현: 이거 떨어뜨렸어." --format summary
```

파일이나 파이프로도 테스트할 수 있습니다.

```powershell
$env:PYTHONPATH='src'
python -m consistent_t2i --text-file .\sample.txt --format summary

Get-Content .\sample.txt -Raw | python -m consistent_t2i --stdin --format summary
```

전체 소설 텍스트를 받아 그림 모델용 일관성 자산으로 정리하려면:

```powershell
$env:PYTHONPATH='src'
python -m consistent_t2i `
  --build-visual-bible `
  --text-file .\novel.txt `
  --title "내 소설" `
  --episode-id "novel-full" `
  --visual-bible-dir .\visual_bible `
  --format summary
```

이 모드는 루트의 `visual_bible` 워크스페이스에 원문 사본과 함께 아래 산출물을 씁니다.

- `*.world.md`
  세계관, 장소, 집단, 규칙, 반복 오브젝트, 시각 모티프 정리
- `*.characters.md`
  캐릭터별 `character canon` 정리. 고정 외형, 가변값, 별칭/호칭, confidence, conflict 포함
- `*.appearance-locks.md`
  그림 모델에 직접 넘기기 좋은 외형 고정 카드
- `*.scene-state.md`
  장면별 의상, 표정, 소품 같은 가변 state 정리
- `*.prompt-pack.json`
  그림 모델에 바로 연결하기 쉬운 global prompt, character card, scene prompt 묶음
- `*.conflicts.md`
  고정 외형끼리 충돌하는 신호를 따로 모아 검수할 수 있는 리포트
- `*.visual-bible.json`
  후속 파이프라인이 읽기 쉬운 구조화 JSON

러프 검토용으로 2컷을 1장으로 묶는 draft mode:

```powershell
$env:PYTHONPATH='src'
python -m consistent_t2i --provider mock --image-model gemini-2.5-flash-image --cheap-image --draft-mode --draft-panels-per-image 2 --text "비 오는날 복도에서 두 사람이 마주친다. 도현: 이거 떨어뜨렸어." --panel-count 2 --image-output-dir .\outputs\drafts --format summary
```

Gemini Batch API와 함께 쓰려면:

```powershell
$env:PYTHONPATH='src'
python -m consistent_t2i --provider gemini --image-model gemini-2.5-flash-image --cheap-image --draft-mode --draft-panels-per-image 2 --batch-mode --text "비 오는날 복도에서 두 사람이 마주친다. 도현: 이거 떨어뜨렸어." --panel-count 4 --image-output-dir .\outputs\drafts --format summary
```

배치 완료까지 기다리고 inline 결과를 내려받으려면 `--batch-wait`를 추가하면 됩니다.

실제 OpenAI 이미지 생성까지 하려면:

```powershell
$env:PYTHONPATH='src'
$env:OPENAI_API_KEY='sk-...'
python -m consistent_t2i `
  --interactive `
  --provider openai `
  --title "실전 테스트" `
  --episode-id "ep-real" `
  --panel-count 3 `
  --image-model gpt-image-1.5 `
  --image-size 1024x1536 `
  --image-quality medium `
  --image-output-dir .\outputs\ep-real `
  --format summary
```

이 경우 3단계에서 실제 이미지를 생성해서 `--image-output-dir` 아래에 패널별 PNG 파일로 저장하고, Stage 5가 `*.overlay.svg`와 `*.final.svg`를 함께 만들어 안정적으로 대사를 합성합니다. 요약 출력에는 패널별 비용과 최종 오버레이 경로가 함께 표시됩니다.
말풍선 텍스트는 기본적으로 Stage 5 후처리 오버레이로 들어가며, `--render-bubble-text`는 그 안정적 후처리 경로를 명시적으로 켜는 용도입니다.
이미지 비용을 최대한 아끼려면 `--cheap-image`를 같이 붙이면 됩니다.

Gemini 2.5 Flash Image를 쓰려면 `.env`에 `GEMINI_API_KEY`를 두고 아래처럼 실행하면 됩니다.

```powershell
$env:PYTHONPATH='src'
python -m consistent_t2i `
  --interactive `
  --provider gemini `
  --title "Gemini 실전 테스트" `
  --episode-id "ep-gemini" `
  --panel-count 3 `
  --image-model gemini-2.5-flash-image `
  --cheap-image `
  --render-bubble-text `
  --image-output-dir .\outputs\ep-gemini `
  --format summary
```

Gemini 경로는 `generateContent` REST API를 사용하며, 현재 구현은 `1024x1024` 기본 비율 기준 비용 추정과 실제 응답 `usageMetadata`를 함께 사용합니다. 테스트 코드는 실제 Gemini 이미지를 생성하지 않도록 네트워크 호출 없이 payload와 비용 계산만 검증합니다.

샘플 실행은 다음을 출력합니다.

- 웹소설에서 잘린 패널 단위 텍스트 콘티
- 각 패널의 졸라맨 그림 콘티 가이드
- 그림 고도화용 요청과 mock 이미지 결과
- Gemma4 같은 모델에 넘길 저토큰 말풍선 요청
- Stage 5 SVG 오버레이와 최종 합성 SVG 경로
- Draft spread 요약과 Batch manifest/job 정보
- 패널별 생성 요청
- mock provider가 만든 가짜 이미지 결과
- 캐릭터 외형이 유지되었는지에 대한 드리프트 리포트
- cold path가 승인 차단/복구 필요 여부를 어떻게 판단하는지에 대한 리뷰 결과
- provider에 전달되는 hidden system prompt
- 실제 provider 사용 시 저장된 이미지 파일 경로
- 단계 3 패널별 비용과 총비용

## 테스트

```powershell
$env:PYTHONPATH='src'; python -m unittest discover -s tests -v
```

## 실제 모델에 붙이는 방법

현재 코드는 외부 의존성 없이 돌아가는 코어입니다. 실제 서비스에 붙일 때는 아래 순서가 안전합니다.

1. `pipeline.py`의 `NovelToWebtoonPipeline`으로 1~4단계 초안을 먼저 만듭니다.
2. 3단계 이미지 모델은 `ModelEndpointConfig(stage="refine", model="gpt-image-1.5", api_key_env="OPENAI_API_KEY")`처럼 설정하고, API 키는 배포 시점에만 주입합니다.
   Gemini를 쓸 때는 `ModelEndpointConfig(stage="refine", provider="gemini", model="gemini-2.5-flash-image", api_key_env="GEMINI_API_KEY")`를 사용하면 됩니다.
3. 실제 OpenAI 호출은 `OpenAIImageProvider`가 수행하며, 기본 URL은 `https://api.openai.com`이고 `v1/images`를 먼저 시도한 뒤 필요하면 `v1/images/generations`로 fallback합니다.
4. 실제 Gemini 호출은 `GeminiImageProvider`가 수행하며, `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent` 형식으로 호출합니다.
5. GPT Image API는 별도 `system` 필드가 없으므로, 현재 구현은 continuity guardrail + positive prompt + negative prompt를 하나의 최종 prompt로 합쳐 보냅니다.
6. Gemini도 별도 system 필드 대신 하나의 text prompt로 보내며, aspect ratio는 내부적으로 `size -> imageConfig.aspectRatio`로 변환합니다.
7. 비용 계산은 응답 `usage`/`usageMetadata`가 있으면 가능한 범위에서 반영하고, 없으면 공식 per-image 가격표와 prompt 길이 기반의 text input 추정치로 계산합니다.
8. 4단계 말풍선 모델은 `ModelEndpointConfig(stage="bubble", model="gemma4", max_output_tokens=120)`처럼 낮은 토큰 예산으로 따로 둡니다.
9. Stage 3 이미지 생성은 텍스트를 직접 그리지 않고, 빈 말풍선/빈 bubble lane만 남기도록 유도합니다.
10. Stage 5는 `bubble_renderer.py`에서 SVG 오버레이와 최종 합성 SVG를 생성합니다.
11. Draft mode에서는 `draft_mode.py`가 인접 컷을 2-up spread로 묶고, Stage 5 overlay도 spread 단위로 오프셋 합성합니다.
12. Gemini Batch API를 사용할 경우 `batch_api.py`가 inline batch body, manifest, job polling, 결과 materialization을 담당합니다.
13. 생성 결과에서 얼굴/복장 속성을 추출하는 관측 계층을 붙입니다.
14. `ColdPathController`에서 승인 차단과 복구 정책을 서비스 규칙으로 확장합니다.
15. 운영에서는 `generate -> review_and_approve_generation` 순서를 기본 경로로 사용합니다.
16. 에피소드/씬 단위로 canon reference를 별도 저장소에 보관합니다.

## 비용 기준

2026년 4월 17일 기준 OpenAI 공식 문서에서는 `gpt-image-1.5`를 최신 GPT Image 모델로 안내합니다. 이 저장소는 다음 기준을 반영합니다.

- `gpt-image-1.5`
  - `1024x1024`: low `$0.009`, medium `$0.034`, high `$0.133`
  - `1024x1536` / `1536x1024`: medium `$0.05`
- `gpt-image-1`
  - `1024x1024`: low `$0.011`, medium `$0.042`, high `$0.167`
- `gemini-2.5-flash-image`
  - standard: `$0.039` per image
  - input: `$0.30 / 1M tokens`

여기에 prompt text input 비용이 소량 추가됩니다. 실제 비용은 OpenAI 공식 문서를 기준으로 수시 변동될 수 있으니 배포 전 한 번 더 확인하는 것이 안전합니다.

## 설계 판단

- 새 라이브러리를 추가하지 않았습니다.
- 실제 이미지 모델 API보다 먼저 "일관성 유지 코어"를 구현했습니다.
- 실시간 경로에는 생성에 꼭 필요한 로직만 남기고, 복구 판단은 cold path로 분리했습니다.
- provider 계층에서 숨겨진 system prompt를 항상 포함하도록 구조를 보강했습니다.
- 실제 OpenAI 호출은 명시적으로 `--provider openai`일 때만 발생하게 해 무의식적인 과금을 막았습니다.
- Gemini도 명시적으로 `--provider gemini`일 때만 실제 호출되게 해, 테스트 중 임의 과금이 발생하지 않도록 했습니다.
- 비용은 가능한 경우 API usage 기준으로 계산하고, 그렇지 않으면 공식 가격표 기반 추정으로 표시합니다.
- 말풍선 텍스트는 이미지 모델에게 직접 맡기지 않고, 기본적으로 Stage 5 SVG 후처리 오버레이로 넣어 안정성을 높였습니다.
- 생성 품질보다 `재현 가능성`, `검증 가능성`, `레퍼런스 재사용성`을 우선했습니다.
