# novel_conti_tiff

루트에서 바로 실행할 수 있는 초경량 웹소설 콘티 TIFF 생성기입니다.

핵심 목표는 두 가지입니다.

1. 웹소설 텍스트를 패널 단위 콘티로 잘라서 시트 형태로 보기 쉽게 정리합니다.
2. 각 페이지를 `1비트` 흑백 TIFF로 저장하고, 기본적으로 페이지별 저장 방식을 써서 피크 메모리를 아주 작게 유지합니다.

## 특징

- `Pillow`의 `mode="1"` 비트맵 캔버스를 사용해 페이지를 직접 그립니다.
- 저장은 `TIFF Group 4` 압축을 사용해 흑백 라인아트/텍스트에 유리합니다.
- 기본 동작은 페이지별 TIFF 저장이라 큰 멀티페이지 이미지 버퍼를 오래 들고 있지 않습니다.
- 패널 분할은 기존 저장소의 콘티 분할 감각을 닮은 규칙 기반 휴리스틱으로 처리합니다.

## 실행

```powershell
python -m novel_conti_tiff `
  --text-file .\sample.txt `
  --title "비 오는 복도" `
  --episode-id "hallway-01" `
  --panel-count 8 `
  --output-dir .\outputs\novel_conti_tiff
```

표준 입력으로도 받을 수 있습니다.

```powershell
Get-Content .\sample.txt -Raw | python -m novel_conti_tiff --stdin --title "stdin 테스트"
```

필요하면 마지막에 멀티페이지 TIFF도 같이 만들 수 있습니다.

```powershell
python -m novel_conti_tiff --text-file .\sample.txt --bundle
```

## 출력물

- `source.txt`
  입력 원문 사본
- `manifest.json`
  생성된 페이지 정보와 파일 크기
- `page-001.tiff`, `page-002.tiff`, ...
  페이지별 1비트 TIFF
- `storyboard.tiff`
  `--bundle` 사용 시 생성되는 멀티페이지 TIFF
