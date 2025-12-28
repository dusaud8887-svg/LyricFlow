# 🎵 노래 가사 싱크 프로젝트 - 기술 최신 정보 종합 (2025년 12월)

## 📋 목차
1. [stable-ts (stable-whisper)](#1-stable-ts-stable-whisper)
2. [Whisper 모델 계열](#2-whisper-모델-계열)
3. [faster-whisper](#3-faster-whisper)
4. [PyTorch & CUDA](#4-pytorch--cuda)
5. [Demucs (보컬 분리)](#5-demucs-보컬-분리)
6. [LRC 파일 포맷](#6-lrc-파일-포맷)
7. [일본어 음성 인식 팁](#7-일본어-음성-인식-팁)
8. [커뮤니티 베스트 프랙티스](#8-커뮤니티-베스트-프랙티스)
9. [권장 설정 및 워크플로우](#9-권장-설정-및-워크플로우)

---

## 1. stable-ts (stable-whisper)

### 📌 개요
OpenAI Whisper를 수정하여 **더 안정적인 타임스탬프**를 생성하는 라이브러리. 가사 싱크의 핵심 도구.

### 🔄 최신 버전 (2025년 12월 기준)
| 버전 | 출시일 | 주요 변경사항 |
|------|--------|--------------|
| **2.19.1** | 2025.08 | 최신 안정 버전 |
| 2.19.0 | 2025.03 | 주요 업데이트 |
| 2.18.3 | 2025.01 | 버그 수정 |
| 2.18.0 | 2024.12 | 기능 개선 |

### ⚙️ 핵심 기능

#### 1) Forced Alignment (강제 정렬) - 가사 싱크의 핵심!
```python
import stable_whisper

model = stable_whisper.load_model('large-v3', device='cuda')

# 텍스트와 오디오 정렬 (가사 싱크)
result = model.align(
    'audio.mp3', 
    lyrics_text,  # 원본 가사 텍스트
    language='ja'  # 일본어
)
```

#### 2) 고급 전처리 옵션
```python
result = model.transcribe(
    'audio.mp3',
    language='ja',
    
    # 🔇 침묵 억제 (VAD)
    suppress_silence=True,
    vad=True,  # Silero VAD 사용
    vad_threshold=0.35,  # 음성 감지 임계값 (0.35 권장)
    
    # 🎵 노이즈 제거
    denoiser='demucs',  # Demucs로 보컬 분리
    only_voice_freq=True,  # 200-5000Hz만 사용 (음성 주파수)
    
    # 📊 세그먼트 재그룹화
    regroup=True,  # 자연스러운 경계로 재그룹화
    
    # 🔄 기타 옵션
    mel_first=True,  # 긴 오디오시 메모리 절약
    word_timestamps=True,  # 단어별 타임스탬프
)
```

#### 3) 출력 포맷
```python
# SRT/VTT 출력
result.to_srt_vtt('output.srt', word_level=False)  # 라인별
result.to_srt_vtt('output_word.srt', word_level=True)  # 단어별

# ASS 출력 (카라오케 스타일)
result.to_ass('output.ass')

# JSON 저장 (재처리용)
result.save_as_json('output.json')
```

#### 4) 세그먼트 조작 메서드
```python
# 구두점으로 분할
result.split_by_punctuation([('.', ' '), '。', '?', '？', ',', '，'])

# 침묵 구간으로 분할
result.split_by_gap(0.5)  # 0.5초 이상 gap에서 분할

# 짧은 세그먼트 병합
result.merge_by_gap(0.15, max_words=3)
```

### 📦 설치
```bash
pip install stable-ts

# 또는 최신 개발 버전
pip install git+https://github.com/jianfch/stable-ts.git
```

### ⚠️ 주의사항
- `word_timestamps=False` 사용 금지 - 세그먼트 타임스탬프 보정에 필요
- 긴 오디오에서 이상 동작 시 `mel_first=True` 사용

---

## 2. Whisper 모델 계열

### 🏆 모델 비교 (2025년 12월 기준)

| 모델 | 파라미터 | VRAM | 속도 | 정확도 | 비고 |
|------|----------|------|------|--------|------|
| **large-v3** | 1.55B | ~10GB | 기준 | 최고 | 🏆 최고 품질 |
| **large-v3-turbo** | 809M | ~6GB | 6배↑ | large-v2급 | ⚡ 속도/품질 균형 |
| distil-large-v3 | 756M | ~4GB | 6배↑ | v3의 99% | 영어 전용 |
| medium | 769M | ~5GB | 빠름 | 좋음 | 다국어 지원 |
| small | 244M | ~2GB | 매우 빠름 | 보통 | 저사양용 |

### 🆕 large-v3-turbo (2024년 10월 출시)

**핵심 특징:**
- 디코더 레이어 32개 → 4개로 축소 (Distil-Whisper 영감)
- large-v3 대비 **6배 빠른 추론 속도**
- large-v2와 동등한 정확도 유지
- 번역 성능은 하락 (transcription 데이터로만 fine-tune)

**일본어 성능:**
- large-v2와 동등한 일본어 정확도 유지
- 태국어, 광둥어 등 일부 언어에서만 정확도 하락

```python
# large-v3-turbo 사용
model = stable_whisper.load_model('large-v3-turbo', device='cuda')
```

### 📊 WER (Word Error Rate) 벤치마크
| 모델 | 전체 WER | 일본어 WER |
|------|----------|-----------|
| large-v3 | 7.88% | ~5-6% |
| large-v3-turbo | 7.75% | large-v2급 |
| large-v2 | ~8% | ~6% |

---

## 3. faster-whisper

### 📌 개요
CTranslate2 기반 Whisper 재구현. **원본 대비 4배 빠르고 메모리 효율적**.

### 🔄 최신 버전
| 버전 | 출시일 | 비고 |
|------|--------|------|
| **1.2.1** | 2025.10 | 최신 |
| 1.2.0 | 2025.08 | |
| 1.1.1 | 2025.01 | |
| 1.1.0 | 2024.11 | turbo 지원 |

### ⚡ 성능 벤치마크 (13분 오디오, RTX 3070 Ti)

| 구현체 | 정밀도 | 시간 | GPU 메모리 | CPU 메모리 |
|--------|--------|------|-----------|-----------|
| faster-whisper large-v3 | fp16 | 52초 | 4521MB | 901MB |
| faster-whisper large-v3 | int8 | 53초 | 2953MB | 2261MB |
| faster-large-v3-turbo | fp16 | **19초** | - | - |
| faster-distil-large-v3 | fp16 | 26초 | 2409MB | 900MB |

### 💻 사용법
```python
from faster_whisper import WhisperModel

# 모델 로드
model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16"  # 또는 "int8" (메모리 절약)
)

# 전사
segments, info = model.transcribe(
    "audio.mp3",
    language="ja",
    beam_size=5,
    vad_filter=True,  # VAD 필터 활성화
    vad_parameters=dict(
        min_silence_duration_ms=500,
        speech_pad_ms=400
    )
)

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

### 🔧 양자화 옵션
| compute_type | 메모리 | 속도 | 정확도 | 권장 환경 |
|--------------|--------|------|--------|----------|
| float32 | 높음 | 느림 | 최고 | CPU |
| float16 | 중간 | 빠름 | 높음 | GPU (권장) |
| int8 | 낮음 | 빠름 | 좋음 | 저사양 GPU |
| int8_float16 | 낮음 | 빠름 | 좋음 | GPU 메모리 부족시 |

### 📦 설치
```bash
pip install faster-whisper

# CUDA 12 필요 (최신 버전)
# cuDNN 9 필요
```

---

## 4. PyTorch & CUDA

### 🔄 최신 버전 (2025년 12월)

| PyTorch | CUDA 지원 | 출시일 |
|---------|-----------|--------|
| **2.9.0** | 12.6, 12.8, 13.0 | 최신 |
| 2.8.0 | 12.6, 12.8, 12.9 | |
| 2.7.1 | 12.4, 12.6 | |
| 2.5.1 | 11.8, 12.1, 12.4 | 안정 |

### 💻 RTX 3070 Ti 권장 설정

```bash
# CUDA 12.4 + PyTorch 2.5.1 (안정적인 조합)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# 또는 최신 버전
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126
```

### ✅ GPU 확인
```python
import torch

print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"CUDA 버전: {torch.version.cuda}")
print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
```

### ⚠️ 일반적인 문제 해결

| 문제 | 해결책 |
|------|--------|
| `torch.cuda.is_available()` = False | PyTorch CUDA 버전과 시스템 CUDA 버전 확인 |
| CUDA out of memory | `compute_type="int8"` 또는 더 작은 모델 사용 |
| 드라이버 호환성 | `nvidia-smi`로 드라이버 버전 확인 후 업데이트 |

### 🔧 CUDA 버전 호환성
- **RTX 3070 Ti**: CUDA 11.0 이상 필요
- **RTX 40xx 시리즈**: CUDA 11.8 이상 권장
- **RTX 50xx 시리즈**: CUDA 12.8 이상 필요 (Blackwell)

---

## 5. Demucs (보컬 분리)

### 📌 개요
Meta/Kyutai의 **최첨단 음원 분리 모델**. 노래에서 보컬을 분리하여 가사 인식 정확도 향상.

### 🏗️ 아키텍처 (v4 - Hybrid Transformer Demucs)

```
                    ┌─────────────────────────────────┐
                    │      Cross-Domain Transformer    │
                    │   (Self-Attention + Cross-Attn)  │
                    └──────────────┬──────────────────┘
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         │                                                   │
┌────────┴────────┐                               ┌──────────┴────────┐
│  Time Domain    │                               │  Frequency Domain │
│    U-Net        │                               │      U-Net        │
│ (Waveform)      │                               │  (Spectrogram)    │
└─────────────────┘                               └───────────────────┘
```

### 🎯 사전 학습 모델

| 모델 | 설명 | 품질 | 속도 |
|------|------|------|------|
| **htdemucs_ft** | Fine-tuned HT Demucs | 최고 🏆 | 4배 느림 |
| **htdemucs** | 기본 HT Demucs | 매우 좋음 | 빠름 |
| htdemucs_6s | 6소스 (피아노, 기타 추가) | 실험적 | 보통 |
| hdemucs_mmi | Hybrid Demucs v3 | 좋음 | 빠름 |

### 📊 벤치마크 (MUSDB HQ)

| 모델 | SDR (dB) | 설명 |
|------|----------|------|
| HT Demucs f.t. | **9.20** | 최고 성능 |
| HT Demucs | 9.00 | 기본 |
| Hybrid Demucs v3 | 8.5+ | 이전 버전 |

*SDR: Signal-to-Distortion Ratio, 높을수록 좋음*

### 💻 사용법

```python
import demucs.separate

# CLI 사용
# demucs -n htdemucs_ft --two-stems=vocals audio.mp3

# Python에서 사용
from demucs import pretrained
from demucs.apply import apply_model
import torch
import torchaudio

# 모델 로드
model = pretrained.get_model('htdemucs_ft')
model.cuda()
model.eval()

# 오디오 로드
wav, sr = torchaudio.load('audio.mp3')
wav = wav.cuda()

# 분리 실행
with torch.no_grad():
    sources = apply_model(model, wav[None], device='cuda')[0]

# sources: [drums, bass, other, vocals]
vocals = sources[3]  # 보컬 추출
```

### 🔧 stable-ts와 통합
```python
result = model.transcribe(
    'audio.mp3',
    denoiser='demucs',  # Demucs로 보컬 분리 후 처리
    language='ja'
)
```

### 📦 설치
```bash
pip install demucs

# 또는
pip install -U demucs
```

### ⚠️ 참고사항
- 저자(Alexandre Défossez)가 Meta 퇴사 후 Kyutai로 이동
- 공식 저장소: `github.com/adefossez/demucs` (새 위치)
- 적극적인 개발은 중단되었으나 버그 수정은 진행

---

## 6. LRC 파일 포맷

### 📝 기본 LRC (Simple LRC)
```lrc
[ar:아티스트명]
[ti:곡 제목]
[al:앨범명]
[length:3:45]

[00:12.00]첫 번째 가사 라인
[00:17.20]두 번째 가사 라인
[00:21.10]세 번째 가사 라인
```

**타임스탬프 형식:** `[mm:ss.xx]` (분:초.밀리초)

### 🌟 Enhanced LRC (단어별 타임스탬프)

카라오케 스타일로 **단어 단위** 하이라이트 가능:

```lrc
[ar:Artist Name]
[ti:Song Title]
[la:ja]
[re:stable-ts]

[00:12.34]<00:12.34>行こう <00:12.89>この <00:13.45>声に <00:13.78>導かれ
[00:18.50]<00:18.50>今日も <00:18.95>また <00:19.40>一歩 <00:19.85>ずつ
```

**Enhanced 형식:**
- 라인 시작: `[mm:ss.xx]`
- 단어 시작: `<mm:ss.xx>`

### 🔄 stable-ts에서 LRC 출력

```python
# 라인별 LRC
result.to_srt_vtt('output.lrc', word_level=False)

# 단어별 LRC (Enhanced)
result.to_srt_vtt('output_word.lrc', word_level=True)
```

### 📊 출력 포맷 비교

| 포맷 | 용도 | 단어 타임스탬프 | 호환성 |
|------|------|----------------|--------|
| LRC (Simple) | 일반 플레이어 | ❌ | 높음 |
| LRC (Enhanced) | 카라오케 | ✅ | 중간 |
| SRT | 자막 | ❌ | 높음 |
| VTT | 웹 자막 | ✅ | 높음 |
| ASS | 고급 자막/카라오케 | ✅ | 중간 |

---

## 7. 일본어 음성 인식 팁

### 🎯 최적 설정

```python
result = model.transcribe(
    'audio.mp3',
    language='ja',  # 필수: 일본어 명시
    
    # 일본어 최적화 옵션
    initial_prompt="以下は日本語の歌詞です。",  # 일본어 가사임을 명시
    
    # VAD 사용 (배경 음악 구간 제거)
    vad=True,
    vad_threshold=0.35,
    
    # 세그먼트 조정
    regroup=True,
)
```

### ⚠️ 일반적인 문제와 해결책

| 문제 | 원인 | 해결책 |
|------|------|--------|
| 가사 누락 | 배경 음악이 보컬 덮음 | Demucs로 보컬 분리 |
| 환각(Hallucination) | 침묵 구간에서 발생 | VAD 필터 사용 |
| 타임스탬프 오차 | 30초 청크 경계 문제 | VAD + 세그먼트 재그룹화 |
| 반복되는 텍스트 | 모델 환각 | `suppress_ts_tokens=True` |

### 🎤 보컬 분리 권장

노래 가사 인식 시 **Demucs 보컬 분리 필수**:

```python
# 방법 1: stable-ts 내장
result = model.transcribe('audio.mp3', denoiser='demucs', language='ja')

# 방법 2: 별도 전처리
# 1. Demucs로 vocals.wav 추출
# 2. vocals.wav를 stable-ts에 입력
```

### 📊 일본어 WER (Word Error Rate)

Whisper large-v3 기준:
- 깨끗한 음성: **~5%**
- 노래 (보컬 분리 후): **~10-15%**
- 노래 (원본): **~25-35%**

---

## 8. 커뮤니티 베스트 프랙티스

### 🔥 가사 싱크 파이프라인 (권장)

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────┐
│  원본 MP3   │ -> │ Demucs       │ -> │ stable-ts     │ -> │ LRC 출력 │
│             │    │ (보컬 분리)  │    │ (align/강제정렬)│    │          │
└─────────────┘    └──────────────┘    └───────────────┘    └──────────┘
                          ↓
                   ┌──────────────┐
                   │ vocals.wav   │
                   └──────────────┘
```

### 💡 커뮤니티 팁 모음

#### 1) 환각(Hallucination) 줄이기
```python
result = model.transcribe(
    audio,
    temperature=0,  # 결정론적 출력
    suppress_silence=True,
    vad=True,
    condition_on_previous_text=False,  # 이전 텍스트 의존 제거
)
```

#### 2) 타임스탬프 정확도 향상
```python
# Silero VAD 사용
result = model.transcribe(
    audio,
    vad='silero-vad',
    vad_threshold=0.35,
    min_word_dur=0.1,  # 최소 단어 길이
)
```

#### 3) 긴 오디오 처리
```python
# 메모리 효율적 처리
result = model.transcribe(
    audio,
    mel_first=True,  # 전체 오디오를 먼저 Mel 스펙트로그램으로 변환
)
```

#### 4) 일본어 + 영어 혼합 가사
```python
# 언어 자동 감지 사용
result = model.transcribe(
    audio,
    language=None,  # 자동 감지
    # 또는
    language='ja',  # 주 언어 설정 (영어 단어도 인식됨)
)
```

### 🛠️ 유용한 도구들

| 도구 | 용도 | 링크 |
|------|------|------|
| **WhisperX** | 정밀 타임스탬프 + 화자 분리 | github.com/m-bain/whisperX |
| **lyrics-transcriber** | 카라오케 LRC/ASS 생성 | pypi.org/project/lyrics-transcriber |
| **Open-Lyrics** | faster-whisper + GPT 번역 | - |
| **whisper-diarize** | 화자 분리 | - |

---

## 9. 권장 설정 및 워크플로우

### 🎯 호시마치 스이세이 프로젝트 최적 설정

```python
import stable_whisper
import torch

# GPU 확인
assert torch.cuda.is_available(), "CUDA를 사용할 수 없습니다!"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

# 모델 로드 (RTX 3070 Ti 8GB에 최적)
model = stable_whisper.load_model(
    'large-v3',  # 또는 'large-v3-turbo' (더 빠름)
    device='cuda'
)

# 가사 읽기
with open('lyrics/stellar_stellar.txt', 'r', encoding='utf-8') as f:
    lyrics = f.read().strip()

# 강제 정렬 (가사 싱크)
result = model.align(
    'songs/stellar_stellar.mp3',
    lyrics,
    language='ja',
    
    # 선택적: 보컬 분리
    # denoiser='demucs',
)

# 출력
result.to_srt_vtt('output/stellar_stellar.lrc', word_level=False)
```

### 📊 RTX 3070 Ti 예상 성능

| 작업 | 모델 | 3분 곡 처리 시간 | VRAM 사용 |
|------|------|-----------------|-----------|
| align (가사 정렬) | large-v3 | ~10-15초 | ~4-5GB |
| transcribe | large-v3 | ~20-30초 | ~5-6GB |
| align + demucs | large-v3 | ~30-45초 | ~6-7GB |

### 📦 환경 설정 요약

```bash
# 1. 가상환경 생성
python -m venv suisei_lyrics
source suisei_lyrics/bin/activate  # Linux/Mac
# suisei_lyrics\Scripts\activate  # Windows

# 2. PyTorch CUDA 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. 핵심 패키지
pip install stable-ts
pip install demucs

# 4. (선택) faster-whisper
pip install faster-whisper

# 5. 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import stable_whisper; print('stable-ts OK')"
```

### 🗂️ 권장 폴더 구조

```
suisei_lyrics/
├── songs/                    # MP3 파일
│   ├── stellar_stellar.mp3
│   ├── template.mp3
│   └── ghost.mp3
├── lyrics/                   # 원본 가사 (UTF-8)
│   ├── stellar_stellar.txt
│   ├── template.txt
│   └── ghost.txt
├── vocals/                   # (선택) Demucs 보컬 분리 결과
│   └── stellar_stellar_vocals.wav
├── output/                   # 결과 LRC 파일
│   ├── stellar_stellar.lrc
│   └── stellar_stellar_word.lrc
├── sync_suisei.py           # 메인 스크립트
└── requirements.txt
```

---

## 📚 참고 자료

### 공식 문서
- [stable-ts GitHub](https://github.com/jianfch/stable-ts)
- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper)
- [Demucs GitHub](https://github.com/adefossez/demucs)
- [PyTorch 설치](https://pytorch.org/get-started/locally/)

### 논문
- [Hybrid Transformers for Music Source Separation](https://arxiv.org/abs/2211.08553)
- [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) (Whisper)

### 커뮤니티
- [Whisper Discussions](https://github.com/openai/whisper/discussions)
- [stable-ts Issues](https://github.com/jianfch/stable-ts/issues)

---

## 10. 추가 도구: WhisperX

### 📌 개요
WhisperX는 Whisper에 **VAD + Forced Alignment + 화자 분리(Diarization)**를 추가한 확장 도구입니다.

### 🔧 아키텍처
```
┌─────────┐    ┌───────────┐    ┌──────────────┐    ┌────────────────┐
│  Audio  │ -> │ Silero    │ -> │ Whisper      │ -> │ Wav2Vec2       │
│         │    │ VAD       │    │ (faster-     │    │ Forced         │
│         │    │           │    │  whisper)    │    │ Alignment      │
└─────────┘    └───────────┘    └──────────────┘    └────────────────┘
                    ↓                  ↓                    ↓
              음성 구간 감지      텍스트 전사          단어별 타임스탬프
```

### 💻 사용법
```python
import whisperx

device = "cuda"
audio_file = "audio.mp3"

# 1. 전사 (faster-whisper 백엔드)
model = whisperx.load_model("large-v3", device, compute_type="float16")
result = model.transcribe(audio_file, batch_size=16)

# 2. 정렬 모델 로드
model_a, metadata = whisperx.load_align_model(
    language_code=result["language"], 
    device=device
)

# 3. 강제 정렬 (단어별 타임스탬프)
result_aligned = whisperx.align(
    result["segments"], 
    model_a, 
    metadata, 
    audio_file, 
    device
)

print(result_aligned["word_segments"])  # 단어별 타임스탬프
```

### 📊 stable-ts vs WhisperX

| 기능 | stable-ts | WhisperX |
|------|-----------|----------|
| 타임스탬프 안정화 | ✅ | ✅ |
| 단어별 타임스탬프 | ✅ | ✅ |
| 강제 정렬 (가사 싱크) | ✅ `model.align()` | ❌ (전사 후 정렬만) |
| 화자 분리 | ❌ | ✅ |
| 배치 처리 | ❌ | ✅ (70x 실시간) |
| 백엔드 | OpenAI Whisper | faster-whisper |

**가사 싱크 프로젝트**: stable-ts 권장 (`model.align()` 기능 때문)

### 📦 설치
```bash
pip install whisperx
# 또는
pip install git+https://github.com/m-bain/whisperX.git
```

---

## 11. Silero VAD 상세

### 📌 개요
**Silero VAD**는 음성 활동 감지(Voice Activity Detection) 모델로, Whisper 전처리에 필수적입니다.

### 🎯 핵심 특징
- **초경량**: ~1.8MB 모델 크기
- **초고속**: 30ms 청크당 <1ms 처리
- **다국어**: 100개+ 언어 지원
- **샘플레이트**: 8kHz, 16kHz 지원

### ⚙️ stable-ts에서 VAD 설정

```python
result = model.transcribe(
    'audio.mp3',
    language='ja',
    
    # VAD 활성화
    vad=True,  # 또는 'silero-vad'
    
    # VAD 파라미터
    vad_threshold=0.35,      # 음성 감지 임계값 (0.0-1.0)
                              # 높을수록 보수적 (노이즈 무시)
                              # 낮을수록 민감 (부드러운 음성도 감지)
    
    min_silence_duration_ms=500,  # 최소 침묵 길이 (ms)
    speech_pad_ms=400,            # 음성 전후 패딩 (ms)
)
```

### 📊 VAD 임계값 가이드

| 환경 | 권장 임계값 | 설명 |
|------|------------|------|
| 깨끗한 음성 | 0.5 | 기본값 |
| 노이즈 있는 환경 | 0.6-0.7 | 더 보수적 |
| 부드러운 음성 | 0.3-0.4 | 더 민감 |
| **노래/음악** | **0.35** | 권장 |

### ⚠️ 주의사항
- Silero VAD v4에서 긴 침묵 구간에 환각이 발생할 수 있음
- 일부 커뮤니티에서 v3.1 권장하는 경우도 있음

---

## 12. 환각(Hallucination) 문제 해결

### 🔍 환각이란?
Whisper가 실제 음성에 없는 텍스트를 생성하는 현상:
- "Thanks for watching"
- "Subscribe to my channel"
- 같은 문장 무한 반복 (Looping)

### 🛡️ 환각 방지 전략

#### 1) VAD 전처리 (필수)
```python
result = model.transcribe(
    audio,
    vad=True,
    vad_threshold=0.35,
    suppress_silence=True,
)
```

#### 2) condition_on_previous_text 비활성화
```python
result = model.transcribe(
    audio,
    condition_on_previous_text=False,  # 이전 텍스트 의존 제거
)
```

#### 3) 온도 설정
```python
result = model.transcribe(
    audio,
    temperature=0,  # 결정론적 출력 (환각 감소)
)
```

#### 4) 보컬 분리 (노래의 경우)
```python
result = model.transcribe(
    audio,
    denoiser='demucs',  # 배경 음악 제거
)
```

#### 5) FFmpeg 침묵 제거 전처리
```bash
ffmpeg -y -i input.mp3 \
    -af "silenceremove=start_periods=1:stop_periods=-1:start_threshold=-50dB:stop_threshold=-50dB:start_silence=0.1:stop_silence=0.1" \
    output.mp3
```

#### 6) 압축률 체크 (후처리)
```python
# 환각 감지: 비정상적으로 높은 compression_ratio
for segment in result.segments:
    if segment.compression_ratio > 2.4:
        print(f"의심스러운 세그먼트: {segment.text}")
```

### 📋 환각 체크리스트

| 증상 | 해결책 |
|------|--------|
| "Thanks for watching" 등 | VAD 활성화 |
| 같은 문장 반복 | `condition_on_previous_text=False` |
| 침묵에서 텍스트 생성 | `suppress_silence=True` |
| 배경 음악에서 환각 | Demucs 보컬 분리 |
| 긴 침묵 후 환각 | 오디오 분할 처리 |

---

## 13. 실전 트러블슈팅

### ❌ 문제 1: CUDA out of memory

```python
# 해결책 1: int8 양자화
model = stable_whisper.load_model('large-v3', device='cuda')
# 또는
model = WhisperModel("large-v3", device="cuda", compute_type="int8")

# 해결책 2: 더 작은 모델
model = stable_whisper.load_model('medium', device='cuda')

# 해결책 3: CPU 오프로드
model = stable_whisper.load_model('large-v3', device='cuda', cpu_preload=True)
```

### ❌ 문제 2: 타임스탬프 오차

```python
# 해결책: VAD + 세그먼트 재그룹화
result = model.transcribe(
    audio,
    vad=True,
    regroup=True,
)

# 또는 수동 조정
result.split_by_punctuation([('.', ' '), '。', '?', '？'])
result.split_by_gap(0.5)
result.merge_by_gap(0.15, max_words=3)
```

### ❌ 문제 3: 일본어 인식 불량

```python
# 해결책 1: 언어 명시
result = model.transcribe(audio, language='ja')

# 해결책 2: 초기 프롬프트
result = model.transcribe(
    audio, 
    language='ja',
    initial_prompt="以下は日本語の歌詞です。"
)

# 해결책 3: 보컬 분리
result = model.transcribe(audio, language='ja', denoiser='demucs')
```

### ❌ 문제 4: 가사와 싱크 불일치

```python
# 해결책: align() 대신 transcribe() 후 수동 매칭

# 1단계: 전사
result = model.transcribe(audio, language='ja')

# 2단계: 결과 검토 및 수정
for segment in result.segments:
    print(f"[{segment.start:.2f}s] {segment.text}")

# 3단계: 필요시 가사 파일과 비교/수정
```

---

## 14. 성능 최적화 팁

### ⚡ GPU 메모리 최적화

```python
import torch

# 처리 전 캐시 정리
torch.cuda.empty_cache()

# 배치 처리시 모델 재사용
model = stable_whisper.load_model('large-v3', device='cuda')

for song in songs:
    result = model.align(song['audio'], song['lyrics'], language='ja')
    # 중간 결과 저장
    result.save_as_json(f"cache/{song['name']}.json")

# 완료 후 정리
del model
torch.cuda.empty_cache()
```

### ⚡ 배치 처리 스크립트

```python
import stable_whisper
from pathlib import Path
import json

def batch_sync(songs_dir, lyrics_dir, output_dir):
    model = stable_whisper.load_model('large-v3', device='cuda')
    
    for audio_path in Path(songs_dir).glob('*.mp3'):
        name = audio_path.stem
        lyrics_path = Path(lyrics_dir) / f"{name}.txt"
        output_path = Path(output_dir) / f"{name}.lrc"
        
        if not lyrics_path.exists():
            print(f"⚠️ 가사 없음: {name}")
            continue
            
        if output_path.exists():
            print(f"⏭️ 스킵 (이미 존재): {name}")
            continue
        
        print(f"🎵 처리 중: {name}")
        
        lyrics = lyrics_path.read_text(encoding='utf-8')
        result = model.align(str(audio_path), lyrics, language='ja')
        result.to_srt_vtt(str(output_path), word_level=False)
        
        print(f"✅ 완료: {output_path}")
    
    del model
    
if __name__ == '__main__':
    batch_sync('songs/', 'lyrics/', 'output/')
```

---

## 📚 추가 참고 자료

### 학술 논문
- [WhisperX: Time-Accurate Speech Transcription](https://arxiv.org/abs/2303.00747)
- [Investigation of Whisper ASR Hallucinations](https://arxiv.org/html/2501.11378v1)
- [Whisper Has an Internal Word Aligner](https://arxiv.org/html/2509.09987v1)

### 유용한 GitHub 저장소
- [EtienneAb3d/WhisperHallu](https://github.com/EtienneAb3d/WhisperHallu) - 환각 방지
- [mikezzb/lyrics-sync](https://github.com/mikezzb/lyrics-sync) - 가사 싱크 파이프라인
- [beveradb/lyrics-transcriber](https://pypi.org/project/lyrics-transcriber/) - 카라오케 LRC 생성

### 일본어 특화
- [kotoba-tech/kotoba-whisper](https://huggingface.co/kotoba-tech/kotoba-whisper-v1.1) - 일본어 최적화 Whisper

---

*문서 작성일: 2025년 12월 28일*
*검색 기반 최신 정보 종합*
