# visual_bible

전체 소설 텍스트를 받아 그림 모델용 일관성 자산으로 정리하는 워크스페이스입니다.

## 목적

- 세계관, 장소, 소속, 반복 오브젝트를 근거 문장과 함께 정리
- 주요 캐릭터의 역할, 외형, 소품, 관계를 한곳에 모아 drift를 줄이기
- `character canon`과 `scene state`를 분리해, 고정 외형과 장면별 변경값을 따로 관리
- 이미지 모델에 바로 넘길 수 있는 `외형 고정 카드`와 `prompt pack`을 자동 생성

## 폴더 구조

- `input/`
  원문 텍스트 사본이 저장됩니다.
- `output/`
  자동 생성된 `*.world.md`, `*.characters.md`, `*.appearance-locks.md`, `*.scene-state.md`, `*.prompt-pack.json`, `*.conflicts.md`, `*.visual-bible.json`이 저장됩니다.

## 사용 예시

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

자동 생성 결과는 초안 역할을 하므로, 실제 운영 전에 `output/` 문서를 사람이 한 번 검수해 고정 외형 값을 확정하는 것이 안전합니다.
