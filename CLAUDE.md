# CLAUDE.md

## 역할 및 전문성

당신은 **오디오-텍스트 정렬 전문가**입니다. 이 프로젝트에서 당신의 목적은 다음과 같습니다:

- **단일 목적 실행**: 일본어 가사와 MP3 오디오를 정렬하여 정확한 타임스탬프를 가진 LRC/SRT 파일 생성
- **최고 품질 우선**: large-v3 모델을 활용한 최고 정확도 (±0.2~0.3초)
- **GPU 최적화**: RTX 3070 Ti CUDA 가속으로 10배 빠른 처리
- **오류 없는 실행**: 1회용 도구이므로 복잡한 구조보다 안정적인 동작이 핵심

---

## Project Overview

**호시마치 스이세이 가사 싱크**

- 목적: 호시마치 스이세이 노래의 일본어 가사에 정확한 타임스탬프 부여
- 도구: `stable-whisper` (OpenAI Whisper 기반 오디오-텍스트 정렬)
- 모델: `large-v3` (2.9GB, 최고 정확도)
- GPU: RTX 3070 Ti (CUDA 지원, 8GB VRAM)
- 용도: 학습 앱 데이터 생성, 1회성 도구

---

## 핵심 개발 원칙

### 1. 단순성 (Simplicity First)
- 이 프로젝트는 **1회용 도구**입니다
- 복잡한 아키텍처, 테스트, 타입 시스템 불필요
- 목표: **오류 없이 한 번에 잘 동작하는 스크립트**

### 2. 품질 우선 (Quality Over Speed)
- 처리 시간보다 **정확도**가 중요
- large-v3 모델 사용 (medium, small 사용 금지)
- GPU 가속으로 품질 타협 없이 속도 확보

### 3. 실패 방지 (Fail-Safe)
- 모든 파일 존재 여부 사전 확인
- 인코딩 문제 방지 (UTF-8 강제)
- 명확한 에러 메시지 출력
- 개별 곡 실패 시에도 나머지 계속 처리

### 4. 검증 가능성 (Verifiable Output)
- 각 곡 처리 후 소요 시간 출력
- 생성된 LRC 파일 라인 수 확인
- 전체 처리 완료 후 요약 리포트

---

## Tech Stack

| 영역 | 기술 | 버전/사양 |
|------|------|----------|
| Language | Python | 3.10+ |
| Package Manager | uv (권장) / pip | 최신 |
| Core Library | stable-whisper | 최신 |
| ML Framework | PyTorch | CUDA 지원 |
| Model | Whisper large-v3 | 2.9GB |
| GPU | RTX 3070 Ti | 8GB VRAM |
| 언어 설정 | Japanese (`ja`) | - |

---

## 파일 구조

```
suisei_lyrics/
├── songs/                      # MP3 파일
│   ├── stellar_stellar.mp3
│   ├── template.mp3
│   ├── ghost.mp3
│   └── ...
├── lyrics/                     # 일본어 가사 (UTF-8)
│   ├── stellar_stellar.txt
│   ├── template.txt
│   ├── ghost.txt
│   └── ...
├── output/                     # 생성된 LRC 파일
│   ├── stellar_stellar.lrc
│   └── ...
├── sync_suisei.py              # 메인 스크립트
└── CLAUDE.md                   # 이 파일
```

**파일명 규칙**:
- MP3와 가사 파일명 **정확히 일치** (확장자만 다름)
- 공백 대신 언더스코어 (`_`) 사용
- 소문자 권장

---

## Commands

```bash
# 환경 설정 (uv 권장 - 빠르고 안정적)
uv pip install stable-ts torch

# 또는 기존 pip 사용
pip install stable-ts torch

# GPU 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# 단일 곡 테스트
python sync_one.py stellar_stellar

# 전체 배치 처리
python sync_suisei.py

# 결과 확인
ls -la output/
```

---

## 핵심 코드 패턴

### 모델 로드 (GPU 필수)
```python
import stable_whisper
import torch

# GPU 확인 필수
assert torch.cuda.is_available(), "CUDA 필수! GPU를 확인하세요."

# large-v3 모델 로드 (최고 품질)
model = stable_whisper.load_model('large-v3', device='cuda')
```

### 가사 정렬 (일본어)
```python
# 가사 파일 읽기 (UTF-8 필수)
with open('lyrics/song.txt', 'r', encoding='utf-8') as f:
    lyrics = f.read().strip()

# 오디오-텍스트 정렬 (일본어 명시)
result = model.align(
    'songs/song.mp3',
    lyrics,
    language='ja'  # 반드시 일본어 지정!
)

# LRC 저장 (라인 단위)
result.to_srt_vtt('output/song.lrc', word_level=False)
```

### 에러 처리 패턴
```python
from pathlib import Path

def process_song(song_name: str) -> bool:
    mp3_path = Path(f'songs/{song_name}.mp3')
    lyrics_path = Path(f'lyrics/{song_name}.txt')
    output_path = Path(f'output/{song_name}.lrc')
    
    # 파일 존재 확인
    if not mp3_path.exists():
        print(f"❌ MP3 없음: {mp3_path}")
        return False
    if not lyrics_path.exists():
        print(f"❌ 가사 없음: {lyrics_path}")
        return False
    
    try:
        # 처리 로직...
        return True
    except Exception as e:
        print(f"❌ 처리 실패 [{song_name}]: {e}")
        return False
```

---

## 가사 파일 형식

### ✅ 올바른 형식 (UTF-8, 일본어 원문)
```text
行こう　この声に導かれ
今日もまた一歩ずつ
夢見た場所へ
輝く未来を信じて
```

### ❌ 잘못된 형식
```text
# 로마자 표기 - 절대 금지!
iko kono koe ni michibikarete

# 번역 - 절대 금지!
Let's go, guided by this voice

# BOM 포함 - 문제 발생 가능
```

### 인코딩 확인
```bash
# 파일 인코딩 확인
file -i lyrics/stellar_stellar.txt
# 출력: text/plain; charset=utf-8  ← 이게 나와야 함

# BOM 제거 (필요시)
sed -i '1s/^\xEF\xBB\xBF//' lyrics/*.txt
```

---

## 예상 성능

### RTX 3070 Ti 처리 속도

| 곡 길이 | 예상 시간 | 정확도 |
|---------|----------|--------|
| 3분 | 10~15초 | ±0.2초 |
| 4분 | 15~20초 | ±0.2초 |
| 5분 | 20~25초 | ±0.3초 |

### 리소스 사용량
- VRAM: ~4~5GB / 8GB
- GPU 사용률: ~80~90%
- 첫 실행: +3분 (모델 다운로드 2.9GB)

---

## 트러블슈팅

### CUDA 관련
```bash
# CUDA 미인식 (uv 사용 시)
uv pip uninstall torch
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 미인식 (pip 사용 시)
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121

# VRAM 부족 (8GB면 괜찮음)
# → medium 모델로 대체 (품질 저하 감수)
model = stable_whisper.load_model('medium', device='cuda')
```

### 인코딩 관련
```python
# 인코딩 강제 (문제 발생 시)
with open(path, 'r', encoding='utf-8-sig') as f:  # BOM 처리
    lyrics = f.read()
```

### 정렬 품질 문제
```python
# 가사가 노래와 다를 때 → 원본 가사 확인
# 빈 라인이 많을 때 → 빈 라인 제거
lyrics = '\n'.join(line for line in lyrics.split('\n') if line.strip())
```

---

## 품질 체크리스트

### 실행 전
- [ ] GPU 인식 확인 (`torch.cuda.is_available()`)
- [ ] stable-ts 설치 완료
- [ ] 폴더 구조 생성 (songs/, lyrics/, output/)
- [ ] 모든 가사 파일 UTF-8 인코딩 확인
- [ ] MP3-가사 파일명 매칭 확인

### 실행 후
- [ ] LRC 파일 생성 확인
- [ ] 음악 플레이어에서 싱크 테스트
- [ ] 타임스탬프 정확도 확인 (±0.3초 이내)
- [ ] 일본어 텍스트 깨짐 없음

---

## 예상 출력 (LRC 샘플)

```lrc
[00:15.23] 行こう　この声に導かれ
[00:19.45] 今日もまた一歩ずつ
[00:23.67] 夢見た場所へ
[00:27.89] 輝く未来を信じて
```

---

## 주의사항

### ⚠️ 절대 하지 말 것
- ❌ `medium` 또는 `small` 모델 사용 (품질 저하)
- ❌ `language` 파라미터 생략 (자동 감지 불안정)
- ❌ CPU 모드 실행 (10배 느림)
- ❌ 로마자/번역 가사 사용
- ❌ 복잡한 에러 핸들링 추가 (1회용이므로)

### ✅ 반드시 할 것
- ✅ `large-v3` 모델만 사용
- ✅ `language='ja'` 명시
- ✅ `device='cuda'` 명시
- ✅ UTF-8 인코딩 강제
- ✅ 각 곡 처리 결과 즉시 확인

---

## Out of Scope

이 프로젝트에서 **하지 않을 것**:

- ❌ 웹 UI / CLI 인터페이스
- ❌ 테스트 코드
- ❌ 타입 힌트 (간단한 힌트는 OK)
- ❌ 패키징 / 배포
- ❌ 로깅 시스템
- ❌ 설정 파일 (YAML, JSON 등)
- ❌ 다국어 지원 (일본어만)
- ❌ 실시간 처리

**목표는 단 하나**: MP3 + 가사 → LRC 변환, 최고 품질로, 오류 없이.

---

## 실행 계획

1. **환경 확인** (2분)
   ```bash
   # uv 설치 (권장 - 빠르고 안정적)
   pip install uv

   # 의존성 설치 (uv 사용)
   uv pip install stable-ts torch

   # 또는 기존 pip 사용
   pip install stable-ts torch

   # GPU 확인
   python -c "import torch; print(torch.cuda.get_device_name(0))"
   ```

2. **폴더 생성** (1분)
   ```bash
   mkdir -p songs lyrics output
   ```

3. **파일 배치** (5분)
   - MP3 → `songs/`
   - 가사 → `lyrics/` (UTF-8, 일본어 원문)

4. **단일 테스트** (2분)
   ```bash
   python sync_one.py stellar_stellar
   cat output/stellar_stellar.lrc
   ```

5. **배치 실행** (10곡 기준 3분)
   ```bash
   python sync_suisei.py
   ```

6. **검증** (5분)
   - 음악 플레이어에서 LRC 테스트
   - 타임스탬프 정확도 확인

**총 소요시간**: ~20분 (10곡 기준)
