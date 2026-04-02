# sh-sh-shk

웹소설을 웹툰으로 바꾸는 과정에서 가장 큰 문제는 "같은 인물인데 매 컷마다 다른 사람처럼 보이는 현상"입니다.
이 저장소는 그 문제를 줄이기 위해, 단순 prompt 문자열이 아니라 `연속성 메모리 + 캐릭터/스타일 바이블 + 결정적 요청 조립 + 드리프트 검증`을 중심으로 한 Python 코어를 제공합니다.
실시간 운영을 염두에 두고, `hot path`와 `cold path`를 분리한 경량 구조를 기본으로 둡니다.

## 목표

- 이전에 생성된 그림체가 다음 컷에서 무너지지 않게 유지
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
- `src/consistent_t2i/paths.py`
  실시간 생성 경로(`hot path`)와 검토/복구 판단 경로(`cold path`)를 분리합니다.
- `src/consistent_t2i/drift.py`
  관측된 외형 속성을 캐릭터 바이블과 비교해 외형 붕괴를 점수화합니다.
- `src/consistent_t2i/engine.py`
  계획, 생성, 승인, 재사용을 묶는 얇은 오케스트레이션 계층입니다.
- `src/consistent_t2i/providers.py`
  실제 이미지 생성 모델 연동을 붙이기 위한 추상 인터페이스와 mock provider입니다.

## 핵심 아이디어

1. `Hot Path`
   패널 요청 조립, 이미지 생성, 승인된 참조 재사용만 담당합니다.
2. `Cold Path`
   드리프트 판정, 복구 필요 여부 판단, canon 갱신 보류 같은 무거운 결정을 담당합니다.
3. `Style Bible`
   작화 질감, 선 두께, 색감, 연출 규칙을 고정합니다.
4. `Character Bible`
   절대 변하면 안 되는 속성과, 장면에 따라 바뀌어도 되는 속성을 분리합니다.
5. `Reference Selection`
   최신 승인 컷과 정본(canon) 레퍼런스를 함께 사용합니다.
6. `Deterministic Request Building`
   prompt 순서, negative prompt, seed bundle을 항상 같은 방식으로 조립합니다.
7. `Drift Detection`
   생성 결과를 바로 canon에 편입하지 않고, 먼저 외형 붕괴 여부를 검사합니다.

## 빠른 실행

```powershell
$env:PYTHONPATH='src'; python -m consistent_t2i
```

샘플 실행은 다음을 출력합니다.

- 패널별 생성 요청
- mock provider가 만든 가짜 이미지 결과
- 캐릭터 외형이 유지되었는지에 대한 드리프트 리포트
- cold path가 승인 차단/복구 필요 여부를 어떻게 판단하는지에 대한 리뷰 결과

## 테스트

```powershell
$env:PYTHONPATH='src'; python -m unittest discover -s tests -v
```

## 실제 모델에 붙이는 방법

현재 코드는 외부 의존성 없이 돌아가는 코어입니다. 실제 서비스에 붙일 때는 아래 순서가 안전합니다.

1. `providers.py`에 실제 모델 어댑터를 추가합니다.
2. 생성 결과에서 얼굴/복장 속성을 추출하는 관측 계층을 붙입니다.
3. `ColdPathController`에서 승인 차단과 복구 정책을 서비스 규칙으로 확장합니다.
4. 운영에서는 `generate -> review_and_approve_generation` 순서를 기본 경로로 사용합니다.
5. 에피소드/씬 단위로 canon reference를 별도 저장소에 보관합니다.

## 설계 판단

- 새 라이브러리를 추가하지 않았습니다.
- 실제 이미지 모델 API보다 먼저 "일관성 유지 코어"를 구현했습니다.
- 실시간 경로에는 생성에 꼭 필요한 로직만 남기고, 복구 판단은 cold path로 분리했습니다.
- 생성 품질보다 `재현 가능성`, `검증 가능성`, `레퍼런스 재사용성`을 우선했습니다.
