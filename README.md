# 🎵 호시마치 스이세이 자막 생성기 (v1.2 Enhanced)

일본어 노래 가사와 MP3 파일을 입력받아 **정확한 타임스탬프가 부여된 LRC 자막 파일**을 자동 생성하는 배치 처리 도구입니다.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![GPU](https://img.shields.io/badge/GPU-CUDA_Required-green.svg)
![Version](https://img.shields.io/badge/version-1.2-brightgreen.svg)

## ✨ 주요 특징

- 🎯 **최고 품질**: Whisper large-v3 모델로 ±0.2~0.3초 정확도
- ⚡ **GPU 가속**: RTX 3070 Ti CUDA 가속으로 3분 곡 기준 15초 처리
- 🔄 **배치 처리**: 여러 곡 자동 처리
- 🛡️ **견고한 실행**: 개별 곡 실패 시에도 나머지 계속 처리
- 📊 **실시간 피드백**: 진행 상황 및 처리 결과 요약 리포트
- 🇯🇵 **일본어 전용**: UTF-8 BOM 처리 및 일본어 최적화

### 🆕 v1.2 신규 기능
- 🎤 **Enhanced LRC 옵션**: 단어별 타임스탬프 (카라오케용)
- 🚀 **모델 선택**: large-v3 vs large-v3-turbo (속도 6배)
- 📈 **진행률 바**: tqdm 지원으로 시각적 피드백
- 📄 **요약 로그 저장**: 처리 결과를 파일로 자동 저장
- ⚙️ **간편한 설정**: 스크립트 상단의 상수로 쉽게 조정

## 📋 목차

- [시스템 요구사항](#-시스템-요구사항)
- [설치](#-설치)
- [사용법](#-사용법)
- [파일 구조](#-파일-구조)
- [예제](#-예제)
- [트러블슈팅](#-트러블슈팅)
- [기술 스택](#-기술-스택)
- [성능](#-성능)
- [라이선스](#-라이선스)

## 🖥️ 시스템 요구사항

### 필수 요구사항
- **Python**: 3.10 이상
- **GPU**: NVIDIA GPU (8GB VRAM 이상 권장)
  - RTX 3070 Ti 또는 동급
  - CUDA 지원 필수
- **디스크 공간**: 최소 5GB (모델 다운로드용)
- **메모리**: 8GB RAM 이상

### 지원 플랫폼
- Linux (Ubuntu 20.04+)
- Windows 10/11
- macOS (CUDA 지원 시)

## 🚀 설치

### 1. Python 환경 확인

```bash
python --version
# Python 3.10 이상이어야 함
```

### 2. GPU 드라이버 설치 확인

```bash
nvidia-smi
# GPU 정보가 출력되어야 함
```

### 3. 필수 라이브러리 설치

```bash
# PyTorch (CUDA 12.4)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# stable-whisper
pip install stable-ts
```

### 4. 저장소 클론

```bash
git clone https://github.com/dusaud8887-svg/mp3.git
cd mp3
```

### 5. 폴더 구조 확인

```bash
mkdir -p songs lyrics output
ls -la
```

## 📖 사용법

### 기본 사용법

1. **MP3 파일 준비**
   - MP3 파일들을 `songs/` 폴더에 배치
   - 파일명: `stellar_stellar.mp3`, `template.mp3` 등

2. **가사 파일 준비**
   - 일본어 가사 파일(.txt)을 `lyrics/` 폴더에 배치
   - 파일명: `stellar_stellar.txt`, `template.txt` 등
   - **중요**: MP3 파일명과 정확히 일치해야 함
   - **인코딩**: UTF-8 (BOM 있어도 자동 처리)

3. **스크립트 실행**

```bash
python sync_suisei.py
```

4. **결과 확인**
   - 생성된 LRC 파일은 `output/` 폴더에 저장됨
   - `output/stellar_stellar.lrc` 등

### 가사 파일 형식

✅ **올바른 형식** (일본어 원문, UTF-8):
```text
行こう　この声に導かれ
今日もまた一歩ずつ
夢見た場所へ
輝く未来を信じて
```

❌ **잘못된 형식**:
- 로마자 표기 (`iko kono koe ni...`) - 사용 금지
- 번역된 텍스트 (한글, 영어) - 사용 금지
- 빈 파일 또는 공백만 있는 파일

### 인코딩 확인

```bash
# 파일 인코딩 확인
file -i lyrics/stellar_stellar.txt
# 출력: text/plain; charset=utf-8

# UTF-8로 변환 (필요시)
iconv -f EUC-KR -t UTF-8 old.txt > lyrics/new.txt
```

## ⚙️ 고급 설정

v1.2부터 스크립트 상단의 상수를 변경하여 동작을 쉽게 커스터마이징할 수 있습니다.

### 설정 가능한 옵션

`sync_suisei.py` 파일 상단을 열고 원하는 값으로 변경하세요:

```python
# ============================================================
# 설정 (사용자 수정 가능)
# ============================================================

# 폴더 경로
SONGS_DIR = 'songs'      # MP3 파일 폴더
LYRICS_DIR = 'lyrics'    # 가사 파일 폴더
OUTPUT_DIR = 'output'    # LRC 출력 폴더

# 모델 선택 (속도 vs 품질)
MODEL_NAME = 'large-v3'          # 최고 품질 (±0.2초), 느림
# MODEL_NAME = 'large-v3-turbo'  # 6배 빠름, large-v2급 품질 (±0.3초)

# 언어
LANGUAGE = 'ja'  # 일본어

# Enhanced LRC 옵션 (단어별 타임스탬프 - 카라오케용)
WORD_LEVEL_LRC = False   # 일반 LRC (라인별)
# WORD_LEVEL_LRC = True  # Enhanced LRC (단어별 - 더 정밀)

# 요약 로그 저장
SAVE_SUMMARY_LOG = True          # 활성화
SUMMARY_LOG_FILE = 'summary.txt' # 로그 파일명
```

### 설정 조합 예시

**1. 최고 품질 모드 (기본)**
```python
MODEL_NAME = 'large-v3'
WORD_LEVEL_LRC = False
```
- 타임스탬프 정확도: ±0.2초
- 처리 속도: 3분 곡 기준 12-15초
- 용도: 일반 자막

**2. 카라오케 모드**
```python
MODEL_NAME = 'large-v3'
WORD_LEVEL_LRC = True  # 단어별 타임스탬프
```
- 타임스탬프 정확도: ±0.2초 (단어별)
- 처리 속도: 3분 곡 기준 12-15초
- 용도: 카라오케 앱, 단어별 하이라이트

**3. 고속 처리 모드**
```python
MODEL_NAME = 'large-v3-turbo'  # 6배 빠름
WORD_LEVEL_LRC = False
```
- 타임스탬프 정확도: ±0.3초
- 처리 속도: 3분 곡 기준 2-3초
- 용도: 대량 배치 처리

**4. 고속 카라오케 모드**
```python
MODEL_NAME = 'large-v3-turbo'
WORD_LEVEL_LRC = True
```
- 타임스탬프 정확도: ±0.3초 (단어별)
- 처리 속도: 3분 곡 기준 2-3초
- 용도: 빠른 카라오케 생성

## 📂 파일 구조

```
mp3/
├── songs/                    # 입력: MP3 파일
│   ├── stellar_stellar.mp3
│   ├── template.mp3
│   └── ghost.mp3
├── lyrics/                   # 입력: 일본어 가사 (UTF-8)
│   ├── stellar_stellar.txt
│   ├── template.txt
│   └── ghost.txt
├── output/                   # 출력: LRC 자막 파일
│   ├── stellar_stellar.lrc
│   ├── template.lrc
│   └── ghost.lrc
├── sync_suisei.py           # 메인 스크립트
├── README.md                # 이 파일
├── SPEC.md                  # 프로그램 기획서
├── progress.md              # 상세 구현 스펙
└── CLAUDE.md                # 프로젝트 가이드라인
```

## 📝 예제

### 실행 예시

```bash
$ python sync_suisei.py

============================================================
🎵 호시마치 스이세이 가사 싱크 시작
============================================================

✅ GPU 감지: NVIDIA GeForce RTX 3070 Ti
✅ VRAM: 8.0GB

------------------------------------------------------------
📂 파일 검증 중...
------------------------------------------------------------
✅ 매칭 완료: stellar_stellar
✅ 매칭 완료: template
⚠️ 가사 누락: ghost (MP3만 존재)

총 처리 대상: 2곡

🔄 large-v3 모델 로딩 중...
   (첫 실행시 2.9GB 다운로드됩니다)

✅ 모델 로딩 완료!

[1/2] 처리 중: stellar_stellar
------------------------------------------------------------
📝 가사 라인: 48개
⏳ 정렬 중... (GPU)
✅ 완료: output/stellar_stellar.lrc
⏱️ 소요시간: 12.3초
📊 LRC 크기: 2.1 KB

[2/2] 처리 중: template
------------------------------------------------------------
📝 가사 라인: 52개
⏳ 정렬 중... (GPU)
✅ 완료: output/template.lrc
⏱️ 소요시간: 13.1초
📊 LRC 크기: 2.3 KB

============================================================
✅ 전체 처리 완료!
============================================================
총 곡 수: 2곡
성공: 2곡
실패: 0곡

총 소요시간: 25.4초 (0.4분)
평균 처리 시간: 12.7초/곡
============================================================
```

### 생성된 LRC 파일 예시

```lrc
[00:15.23] 行こう　この声に導かれ
[00:19.45] 今日もまた一歩ずつ
[00:23.67] 夢見た場所へ
[00:27.89] 輝く未来を信じて
```

### LRC 파일 재생 테스트

```bash
# VLC, foobar2000, MusicBee 등 LRC 지원 플레이어 사용
vlc songs/stellar_stellar.mp3
# output/stellar_stellar.lrc 자동 로드됨
```

## 🔧 트러블슈팅

### 문제 1: CUDA를 사용할 수 없습니다

**증상**:
```
❌ 오류: CUDA를 사용할 수 없습니다!
```

**해결책**:
1. GPU 드라이버 설치 확인
   ```bash
   nvidia-smi
   ```

2. PyTorch CUDA 버전 재설치
   ```bash
   pip uninstall torch
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```

3. CUDA 버전 확인
   ```bash
   nvcc --version
   ```

### 문제 2: 인코딩 오류

**증상**:
```
❌ 오류: 인코딩 오류 (UTF-8 필요)
```

**해결책**:
1. 파일 인코딩 확인
   ```bash
   file -i lyrics/song.txt
   ```

2. UTF-8로 변환
   ```bash
   iconv -f EUC-KR -t UTF-8 lyrics/song.txt > lyrics/song_utf8.txt
   mv lyrics/song_utf8.txt lyrics/song.txt
   ```

3. BOM 제거 (필요시)
   ```bash
   sed -i '1s/^\xEF\xBB\xBF//' lyrics/*.txt
   ```

### 문제 3: 모델 다운로드 실패

**증상**:
```
❌ 모델 로드 실패
```

**해결책**:
1. 인터넷 연결 확인
2. stable-ts 버전 확인
   ```bash
   pip install --upgrade stable-ts
   ```
3. 수동 다운로드 후 재시도

### 문제 4: VRAM 부족

**증상**:
```
RuntimeError: CUDA out of memory
```

**해결책**:
1. 다른 GPU 프로세스 종료
   ```bash
   nvidia-smi
   kill <PID>
   ```

2. GPU 메모리 확인
   ```bash
   nvidia-smi
   ```

3. 8GB 미만 GPU인 경우: medium 모델 사용 (코드 수정 필요)

### 문제 5: 타임스탬프 부정확

**원인**:
- 가사 텍스트가 실제 노래와 다름
- 음원 버전 불일치 (리믹스, 라이브 등)

**해결책**:
1. 공식 가사 확인 (일본어 원문)
2. 오리지널 버전 음원 사용
3. 가사와 음원이 정확히 일치하는지 확인

## 🛠️ 기술 스택

| 항목 | 기술 | 버전 | 용도 |
|------|------|------|------|
| **언어** | Python | 3.10+ | 스크립트 실행 환경 |
| **핵심 엔진** | stable-whisper | 2.19.1+ | 오디오-텍스트 정렬 (Forced Alignment) |
| **ML 모델** | OpenAI Whisper large-v3 | 1.55B params | 최고 정확도 음성 인식 |
| **딥러닝** | PyTorch | 2.5.1+ | CUDA GPU 가속 |
| **CUDA** | CUDA Toolkit | 12.4+ | GPU 연산 |

## ⚡ 성능

### RTX 3070 Ti 기준

| 곡 길이 | 예상 처리 시간 | 정확도 |
|---------|---------------|--------|
| 3분 | 10-15초 | ±0.2초 |
| 4분 | 15-20초 | ±0.2초 |
| 5분 | 20-25초 | ±0.3초 |

### 리소스 사용량
- **VRAM**: ~4-5GB / 8GB
- **GPU 사용률**: ~80-90%
- **모델 크기**: 2.9GB (첫 실행시 자동 다운로드)
- **10곡 배치 처리**: 약 2-3분

## 🎯 주요 개선 사항 (v1.2 Enhanced)

### 신규 기능 (v1.2)
- ✅ **Enhanced LRC 옵션**: 단어별 타임스탬프 카라오케 모드
- ✅ **모델 선택**: large-v3-turbo 추가 (6배 빠름)
- ✅ **진행률 바**: tqdm 지원 (선택적)
- ✅ **요약 로그 저장**: summary.txt 자동 생성
- ✅ **성공한 곡 상세**: 각 곡별 처리 시간 및 라인 수 표시
- ✅ **설정 집중화**: 스크립트 상단에서 모든 설정 관리

### 버그 수정 (v1.1)
- ✅ UTF-8 BOM 자동 처리 추가
- ✅ 안전한 평균 시간 계산 (`time` 키 누락 처리)
- ✅ 더 상세한 에러 메시지 (파일명 포함)

### 기능 개선 (v1.1)
- ✅ 라이브러리 import 에러 명확한 메시지
- ✅ KeyboardInterrupt 처리 (Ctrl+C 중단 시 요약 출력)
- ✅ 빈 라인 자동 제거 (가사 파일 정리)
- ✅ 모델 로드 에러 시 해결 방법 제시
- ✅ 인코딩 에러 시 파일 경로 출력

### 견고성 향상 (v1.1)
- ✅ UTF-8-sig 우선 시도 → UTF-8 폴백
- ✅ BOM 강제 제거 (`\ufeff`)
- ✅ RuntimeError 별도 처리 (GPU 메모리 부족)
- ✅ 안전한 딕셔너리 접근 (`.get()` 사용)

## 📚 참고 문서

### 프로젝트 문서
- [SPEC.md](SPEC.md) - 프로그램 기획서 (아키텍처, 요구사항)
- [progress.md](progress.md) - 상세 구현 스펙 (코드 예제)
- [CLAUDE.md](CLAUDE.md) - 프로젝트 가이드라인

### 외부 문서
- [stable-ts GitHub](https://github.com/jianfch/stable-ts)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [PyTorch 설치 가이드](https://pytorch.org/get-started/locally/)

## ❓ FAQ

### Q1. GPU 없이 사용 가능한가요?
**A**: 이 프로젝트는 GPU 필수입니다. CPU로는 처리 시간이 10배 이상 느려지며 (3분 곡 기준 2-3분), CLAUDE.md에서 GPU 환경을 전제로 하고 있습니다.

### Q2. 다른 언어(영어, 한국어)도 지원하나요?
**A**: 현재는 일본어 전용입니다. `LANGUAGE = 'ja'`로 고정되어 있으며, 다른 언어는 코드 수정이 필요합니다.

### Q3. 단어별 LRC (Enhanced LRC)를 생성할 수 있나요?
**A**: v1.2부터 지원합니다! 스크립트 상단의 `WORD_LEVEL_LRC` 설정을 변경하세요:
```python
# sync_suisei.py 상단
WORD_LEVEL_LRC = True  # Enhanced LRC 활성화 (카라오케용)
```

### Q4. 이미 생성된 LRC 파일은 어떻게 되나요?
**A**: 현재는 항상 덮어씁니다. 스킵 기능이 필요하면 `verify_files()` 함수에 스킵 로직을 추가할 수 있습니다 (progress.md 참고).

### Q5. 여러 GPU가 있을 때 특정 GPU를 선택할 수 있나요?
**A**: 환경 변수로 지정 가능합니다:
```bash
CUDA_VISIBLE_DEVICES=0 python sync_suisei.py  # GPU 0 사용
CUDA_VISIBLE_DEVICES=1 python sync_suisei.py  # GPU 1 사용
```

### Q6. 처리 속도를 더 빠르게 할 수 있나요?
**A**: `MODEL_NAME`을 `'large-v3-turbo'`로 변경하면 6배 빠릅니다 (품질은 large-v2급):
```python
# sync_suisei.py 상단
MODEL_NAME = 'large-v3-turbo'  # 6배 빠름
```

### Q7. 진행률 바가 표시되지 않아요
**A**: tqdm 라이브러리를 설치하세요:
```bash
pip install tqdm
```
tqdm이 없어도 정상 동작하지만, 진행률 바는 표시되지 않습니다.

### Q8. 요약 로그 파일은 어디에 저장되나요?
**A**: 프로젝트 루트 디렉토리에 `summary.txt`로 저장됩니다. 비활성화하려면:
```python
# sync_suisei.py 상단
SAVE_SUMMARY_LOG = False
```

## 🤝 기여

이 프로젝트는 1회용 도구로 개발되었으며, 현재는 기여를 받지 않습니다.

개선 제안이 있다면 Issue를 열어주세요.

## 📄 라이선스

MIT License

Copyright (c) 2025 dusaud8887-svg

## 🙏 감사의 말

- [OpenAI Whisper](https://github.com/openai/whisper) - 최고 품질의 음성 인식 모델
- [stable-ts](https://github.com/jianfch/stable-ts) - 안정적인 타임스탬프 생성
- [호시마치 스이세이](https://www.youtube.com/@HoshimachiSuisei) - 멋진 노래들

## 📞 문의

- GitHub Issues: [https://github.com/dusaud8887-svg/mp3/issues](https://github.com/dusaud8887-svg/mp3/issues)

---

**Made with ❤️ for Hoshimachi Suisei fans**

*Last updated: 2025-12-28*
