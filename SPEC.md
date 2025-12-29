# 🎵 호시마치 스이세이 자막 생성기 - 프로그램 기획서

## 📋 문서 정보
- **프로젝트명**: 호시마치 스이세이 자막 생성기 (Suisei Lyrics Sync Tool)
- **버전**: 1.0.0
- **작성일**: 2025-12-28
- **용도**: 일본어 노래 가사 → LRC 자막 생성 (1회용 배치 처리 도구)

---

## 1. 프로젝트 개요

### 1.1 목적
호시마치 스이세이(星街すいせい)의 노래 MP3 파일과 일본어 가사 텍스트를 입력받아, **정확한 타임스탬프가 부여된 LRC 자막 파일**을 자동 생성하는 배치 처리 도구 개발.

### 1.2 핵심 가치
- **최고 품질 우선**: Whisper large-v3 모델을 활용한 ±0.2~0.3초 정확도
- **GPU 최적화**: RTX 3070 Ti CUDA 가속으로 10배 빠른 처리
- **견고한 실행**: 1회용 도구이므로 복잡한 구조보다 안정적 동작이 핵심
- **자동화**: 최소 설정으로 전체 곡 배치 처리 지원

### 1.3 주요 제약사항
- **1회용 도구**: 복잡한 아키텍처, 유지보수성보다 **즉시 동작**이 우선
- **일본어 전용**: 다국어 지원 불필요
- **오프라인 배치 처리**: 실시간 처리 불필요
- **학습 앱 데이터 생성**: 최종 사용자 UI 불필요

---

## 2. 핵심 목표

### 2.1 기능적 목표
1. ✅ MP3 + 일본어 가사 텍스트 → LRC 파일 생성
2. ✅ 배치 처리로 10개+ 곡 자동 처리
3. ✅ 타임스탬프 정확도 ±0.3초 이내
4. ✅ 개별 곡 실패 시에도 나머지 계속 처리
5. ✅ 처리 진행 상황 실시간 출력

### 2.2 비기능적 목표
1. ✅ GPU 활용률 80% 이상 (RTX 3070 Ti)
2. ✅ 3분 곡 기준 15초 이내 처리
3. ✅ 메모리 효율성 (VRAM 8GB 내 수용)
4. ✅ 명확한 에러 메시지 출력
5. ✅ UTF-8 인코딩 보장

### 2.3 Out of Scope (하지 않을 것)
- ❌ 웹 UI / CLI 인터페이스 (단순 스크립트 실행)
- ❌ 테스트 코드 (수동 검증)
- ❌ 복잡한 타입 시스템 / OOP 구조
- ❌ 패키징 / 배포 (단일 스크립트)
- ❌ 로깅 프레임워크 (print 출력으로 충분)
- ❌ 설정 파일 (하드코딩 허용)
- ❌ 다국어 지원 (일본어만)
- ❌ 실시간 처리 / 스트리밍

---

## 3. 기술 스택

### 3.1 핵심 라이브러리
| 항목 | 기술 | 버전 | 용도 |
|------|------|------|------|
| **언어** | Python | 3.10+ | 스크립트 실행 환경 |
| **핵심 엔진** | stable-whisper | 2.19.1+ | 오디오-텍스트 정렬 (Forced Alignment) |
| **ML 모델** | OpenAI Whisper large-v3 | 1.55B params | 최고 정확도 음성 인식 |
| **딥러닝** | PyTorch | 2.5.1+ | CUDA GPU 가속 |
| **CUDA** | CUDA Toolkit | 12.4+ | GPU 연산 |

### 3.2 선택적 라이브러리 (Phase 2)
| 라이브러리 | 용도 | 우선순위 |
|-----------|------|----------|
| Demucs | 보컬 분리 (품질 향상) | 낮음 (Phase 1에서는 생략 가능) |
| faster-whisper | 속도 최적화 대안 | 낮음 (stable-ts로 충분) |

### 3.3 하드웨어 사양
- **GPU**: RTX 3070 Ti (8GB VRAM)
- **VRAM 사용량**: ~4-5GB (large-v3)
- **예상 처리 속도**: 3분 곡 기준 10-15초

---

## 4. 시스템 아키텍처

### 4.1 전체 흐름도
```
┌──────────────────────────────────────────────────────────────────┐
│                       자막 생성 파이프라인                         │
└──────────────────────────────────────────────────────────────────┘

   사용자 입력 (폴더)
         │
         ▼
   ┌─────────────┐
   │ songs/*.mp3 │  ← MP3 파일들
   └─────────────┘
         │
         ├─────────────────┐
         │                 │
         ▼                 ▼
   ┌──────────┐      ┌──────────┐
   │ lyrics/  │      │ output/  │  ← 빈 폴더 (자동 생성)
   │ *.txt    │      └──────────┘
   └──────────┘
         │
         ▼
   ┌──────────────────────────────────┐
   │     sync_suisei.py 실행          │
   │  (메인 배치 처리 스크립트)        │
   └──────────────────────────────────┘
         │
         ├─ [1] 환경 검증 (GPU, CUDA, 모델)
         ├─ [2] 파일 매칭 (MP3 ↔ TXT)
         ├─ [3] 모델 로드 (large-v3, GPU)
         │
         ▼
   ┌──────────────────────────────────┐
   │   for each song:                 │
   │     1. 가사 읽기 (UTF-8)         │
   │     2. model.align()             │
   │     3. LRC 출력                  │
   │     4. 에러 핸들링               │
   └──────────────────────────────────┘
         │
         ▼
   ┌──────────────────────────────────┐
   │   output/*.lrc 생성 완료         │
   │   + 처리 요약 리포트 출력        │
   └──────────────────────────────────┘
```

### 4.2 모듈 구조 (단일 스크립트)
```python
# sync_suisei.py - 단일 스크립트 구조

# [1] 임포트 및 상수 정의
import stable_whisper
import torch
from pathlib import Path
import time

# [2] 환경 검증 함수
def verify_environment():
    """GPU, CUDA, 모델 다운로드 확인"""
    pass

# [3] 파일 검증 함수
def verify_files(songs_dir, lyrics_dir):
    """MP3-가사 파일 매칭 확인"""
    pass

# [4] 단일 곡 처리 함수
def process_song(model, mp3_path, lyrics_path, output_path):
    """1곡 처리: 가사 읽기 → align → LRC 저장"""
    pass

# [5] 메인 배치 처리 함수
def main():
    """전체 배치 처리 오케스트레이션"""
    verify_environment()
    song_list = verify_files('songs/', 'lyrics/')
    model = stable_whisper.load_model('large-v3', device='cuda')

    for song in song_list:
        process_song(model, song['mp3'], song['lyrics'], song['output'])

    print_summary()

if __name__ == '__main__':
    main()
```

### 4.3 데이터 흐름
```
MP3 Audio File → stable-whisper → Whisper large-v3 → Alignment Result → LRC File
    ↓                                    ↑
Lyrics Text ─────────────────────────────┘
(Japanese, UTF-8)
```

---

## 5. 주요 기능 명세

### 5.1 환경 검증 기능 (verify_environment)
**목적**: 실행 전 필수 조건 확인

**검증 항목**:
- [x] Python 버전 3.10+ 확인
- [x] `torch.cuda.is_available()` → CUDA 사용 가능 여부
- [x] `torch.cuda.get_device_name(0)` → GPU 이름 출력
- [x] VRAM 용량 확인 (8GB 이상 권장)
- [x] stable-whisper 임포트 가능 여부
- [x] 폴더 존재 확인: `songs/`, `lyrics/`, `output/`
- [x] 모델 다운로드 알림 (첫 실행시 2.9GB)

**실패 시 동작**:
- CUDA 미지원 → 에러 메시지 출력 후 종료
- 폴더 없음 → 자동 생성 또는 에러 출력
- VRAM 부족 경고 → 계속 진행 (실패 가능성 경고)

### 5.2 파일 검증 기능 (verify_files)
**목적**: MP3-가사 파일 매칭 및 누락 확인

**입력**:
- `songs_dir`: MP3 파일 폴더 경로
- `lyrics_dir`: 가사 파일 폴더 경로

**출력**:
```python
[
    {'name': 'stellar_stellar', 'mp3': Path('songs/stellar_stellar.mp3'),
     'lyrics': Path('lyrics/stellar_stellar.txt'), 'output': Path('output/stellar_stellar.lrc')},
    ...
]
```

**검증 로직**:
1. `songs/*.mp3` 파일 목록 스캔
2. 각 MP3에 대응하는 `lyrics/{name}.txt` 존재 확인
3. 매칭되는 파일만 리스트에 추가
4. 누락 파일 경고 출력

**출력 예시**:
```
✅ 매칭 완료: stellar_stellar (MP3 + TXT)
✅ 매칭 완료: template (MP3 + TXT)
⚠️ 가사 누락: ghost (MP3만 존재)
⚠️ MP3 누락: next_color (TXT만 존재)

총 처리 대상: 2곡
```

### 5.3 단일 곡 처리 기능 (process_song)
**목적**: 1곡의 MP3 + 가사 → LRC 변환

**함수 시그니처**:
```python
def process_song(
    model,           # stable_whisper.WhisperModel
    mp3_path,        # Path
    lyrics_path,     # Path
    output_path,     # Path
) -> dict:           # {'success': bool, 'time': float, 'error': str}
```

**처리 단계**:
1. **가사 읽기**
   ```python
   with open(lyrics_path, 'r', encoding='utf-8') as f:
       lyrics = f.read().strip()
   ```
   - UTF-8 강제 인코딩
   - 빈 라인 제거 (선택)

2. **모델 정렬 (Forced Alignment)**
   ```python
   result = model.align(
       str(mp3_path),
       lyrics,
       language='ja'  # 필수!
   )
   ```

3. **LRC 저장**
   ```python
   result.to_srt_vtt(str(output_path), word_level=False)
   ```

4. **메타데이터 수집**
   - 처리 시간 (초)
   - 가사 라인 수
   - 생성된 LRC 파일 크기

**에러 핸들링**:
- 파일 읽기 실패 → 인코딩 에러 명시
- 정렬 실패 → 모델 에러 메시지 출력
- 저장 실패 → 디스크 쓰기 권한 확인

**출력 예시**:
```
[1/10] 처리 중: stellar_stellar
------------------------------------------------------------
📝 가사 라인: 48개
🎵 MP3 길이: 3분 42초
⏳ 정렬 중... (GPU)
✅ 완료: output/stellar_stellar.lrc
⏱️ 소요시간: 12.3초
📊 LRC 크기: 2.1 KB
```

### 5.4 배치 처리 기능 (main)
**목적**: 전체 곡 자동 처리 및 요약

**처리 흐름**:
```python
def main():
    # [1] 환경 검증
    verify_environment()

    # [2] 파일 스캔
    songs = verify_files('songs/', 'lyrics/')

    if not songs:
        print("❌ 처리할 곡이 없습니다.")
        return

    # [3] 모델 로드
    print("🔄 large-v3 모델 로딩 중...")
    model = stable_whisper.load_model('large-v3', device='cuda')
    print("✅ 모델 로딩 완료!\n")

    # [4] 배치 처리
    total_start = time.time()
    results = []

    for i, song in enumerate(songs, 1):
        print(f"[{i}/{len(songs)}] 처리 중: {song['name']}")
        result = process_song(model, song['mp3'], song['lyrics'], song['output'])
        results.append(result)

    # [5] 요약 출력
    total_time = time.time() - total_start
    print_summary(results, total_time)
```

**요약 리포트 형식**:
```
============================================================
✅ 전체 처리 완료!
============================================================
총 곡 수: 10곡
성공: 9곡
실패: 1곡 (ghost - 인코딩 오류)
총 소요시간: 2분 15초
평균 처리 시간: 13.5초/곡
GPU 평균 사용률: 85%
============================================================
```

### 5.5 진행 상황 출력 기능
**목적**: 사용자에게 실시간 피드백 제공

**출력 레벨**:
1. **INFO**: 정상 진행 (✅)
2. **WARNING**: 누락 파일 등 (⚠️)
3. **ERROR**: 처리 실패 (❌)

**예시**:
```
✅ GPU 감지: NVIDIA GeForce RTX 3070 Ti
✅ VRAM: 8.0GB
🔄 large-v3 모델 로딩 중... (첫 실행시 2.9GB 다운로드)
✅ 모델 로딩 완료!

[1/3] 처리 중: stellar_stellar
⏳ 정렬 중... (GPU)
✅ 완료: 12.3초

[2/3] 처리 중: template
⏳ 정렬 중... (GPU)
✅ 완료: 10.8초

[3/3] 처리 중: ghost
❌ 오류: 가사 파일 인코딩 오류 (UTF-8 필요)
```

---

## 6. 구현 요구사항

### 6.1 필수 요구사항 (Must Have)
1. ✅ **GPU 필수 확인**: `torch.cuda.is_available()` 체크
2. ✅ **large-v3 모델 고정**: 다른 모델 사용 금지
3. ✅ **일본어 명시**: `language='ja'` 필수
4. ✅ **UTF-8 인코딩**: 가사 파일 읽기/쓰기 모두 UTF-8
5. ✅ **파일명 매칭**: `song_name.mp3` ↔ `song_name.txt` ↔ `song_name.lrc`
6. ✅ **에러 허용**: 1곡 실패 시에도 나머지 계속 처리
7. ✅ **진행 상황 출력**: 각 곡 처리 시작/완료 메시지
8. ✅ **처리 시간 기록**: 각 곡 소요 시간 출력

### 6.2 권장 요구사항 (Should Have)
1. ⭐ **output 폴더 자동 생성**: 없으면 생성
2. ⭐ **이미 존재하는 LRC 스킵**: 재처리 방지
3. ⭐ **가사 라인 수 출력**: 검증용
4. ⭐ **GPU 메모리 정리**: 처리 완료 후 `torch.cuda.empty_cache()`
5. ⭐ **타임스탬프 로깅**: 시작/종료 시각 기록

### 6.3 선택 요구사항 (Nice to Have)
1. 🔹 **Demucs 보컬 분리**: 품질 향상 (Phase 2)
2. 🔹 **단어별 LRC 출력**: Enhanced LRC (Phase 2)
3. 🔹 **JSON 중간 결과 저장**: 재처리용 (Phase 2)
4. 🔹 **설정 파일 지원**: YAML/JSON (Phase 2)

### 6.4 코드 품질 기준
- **간결성 우선**: 1회용이므로 복잡한 추상화 불필요
- **명시적 에러 메시지**: 사용자가 즉시 이해 가능한 설명
- **하드코딩 허용**: 경로, 모델명 등 하드코딩 OK
- **주석 최소화**: 코드 자체로 설명 가능하도록 작성
- **타입 힌트 선택적**: 복잡한 타입만 명시

---

## 7. 품질 기준

### 7.1 정확도 기준
- **타임스탬프 오차**: ±0.3초 이내
- **가사 일치율**: 100% (Forced Alignment이므로 보장)
- **일본어 인코딩**: 깨짐 없음

### 7.2 성능 기준
| 항목 | 목표 | 측정 방법 |
|------|------|----------|
| 처리 속도 | 3분 곡 기준 15초 이내 | 실제 측정 |
| GPU 사용률 | 80% 이상 | nvidia-smi |
| VRAM 사용 | 5GB 이하 | torch.cuda.memory_allocated() |
| 10곡 배치 처리 | 3분 이내 | 전체 소요 시간 측정 |

### 7.3 안정성 기준
- **에러 허용률**: 개별 곡 실패 시에도 전체 중단 없음
- **재실행 안전성**: 여러 번 실행해도 결과 동일 (멱등성)
- **파일 손상 방지**: 원본 MP3/TXT 파일 절대 수정 안 함

### 7.4 검증 방법
1. **수동 검증** (샘플 3곡)
   - 음악 플레이어에서 LRC 재생 테스트
   - 타임스탬프 정확도 육안 확인
   - 일본어 텍스트 깨짐 확인

2. **자동 검증** (스크립트 출력)
   - 모든 곡 LRC 파일 생성 확인
   - 에러 발생 곡 리스트 확인
   - 평균 처리 시간 확인

---

## 8. 위험 요소 및 대응

### 8.1 기술적 위험

| 위험 | 확률 | 영향 | 대응 방안 |
|------|------|------|----------|
| CUDA 미인식 | 중 | 높음 | 사전 환경 확인, 설치 가이드 제공 |
| VRAM 부족 | 낮 | 중 | 8GB 이상 GPU 확인, int8 양자화 옵션 |
| 인코딩 오류 | 중 | 낮 | UTF-8 강제, BOM 제거 로직 |
| 모델 다운로드 실패 | 낮 | 중 | 재시도 로직, 수동 다운로드 가이드 |
| 가사-음성 불일치 | 중 | 중 | 가사 원본 확인 안내, 수동 검증 |

### 8.2 데이터 위험

| 위험 | 대응 |
|------|------|
| 가사 파일 인코딩 오류 | UTF-8-sig 폴백, BOM 제거 |
| 파일명 불일치 | 매칭 검증 단계에서 경고 출력 |
| 빈 가사 파일 | 파일 크기 확인, 빈 파일 스킵 |
| 일본어 외 언어 혼입 | language='ja' 설정으로 자동 처리 |

### 8.3 운영 위험

| 위험 | 대응 |
|------|------|
| 디스크 공간 부족 | 사전 용량 확인 (LRC는 KB 단위) |
| 권한 부족 | output 폴더 쓰기 권한 확인 |
| 장시간 실행 | 진행 상황 출력으로 진행 확인 |

---

## 9. 테스트 계획

### 9.1 테스트 전략
**1회용 도구이므로 자동화된 테스트 대신 수동 검증 중심**

### 9.2 테스트 단계

#### Phase 1: 환경 테스트
- [ ] GPU 인식 확인
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```
- [ ] stable-ts 임포트 확인
  ```bash
  python -c "import stable_whisper"
  ```
- [ ] 모델 다운로드 테스트
  ```bash
  python -c "import stable_whisper; stable_whisper.load_model('large-v3')"
  ```

#### Phase 2: 단일 곡 테스트
- [ ] 1곡으로 전체 파이프라인 테스트
  - 입력: `stellar_stellar.mp3` + `stellar_stellar.txt`
  - 출력: `stellar_stellar.lrc`
  - 검증: 음악 플레이어에서 재생

#### Phase 3: 배치 처리 테스트
- [ ] 3곡으로 배치 처리 테스트
- [ ] 에러 케이스 테스트 (가사 파일 누락)
- [ ] 재실행 테스트 (이미 존재하는 LRC)

#### Phase 4: 품질 검증
- [ ] 타임스탬프 정확도 수동 확인 (3곡 샘플)
- [ ] 일본어 인코딩 확인 (모든 LRC 파일)
- [ ] 처리 시간 측정 (10곡 기준)

### 9.3 테스트 데이터
```
테스트용 최소 데이터셋:
- stellar_stellar.mp3 + .txt (정상 케이스)
- template.mp3 + .txt (정상 케이스)
- ghost.mp3 (가사 누락 케이스)
- next_color.txt (MP3 누락 케이스)
```

---

## 10. 실행 계획

### 10.1 구현 단계 (Phase 1: MVP)

#### Step 1: 환경 설정 (예상 10분)
- [x] Python 3.10+ 설치 확인
- [ ] `pip install stable-ts torch` 실행
- [ ] GPU 드라이버 확인 (`nvidia-smi`)
- [ ] CUDA 버전 확인

#### Step 2: 폴더 구조 생성 (예상 2분)
```bash
mkdir -p songs lyrics output
```

#### Step 3: 스크립트 작성 (예상 30분)
- [ ] `sync_suisei.py` 메인 스크립트 작성
  - [x] 환경 검증 함수
  - [ ] 파일 검증 함수
  - [ ] 단일 곡 처리 함수
  - [ ] 배치 처리 메인 함수
  - [ ] 요약 출력 함수

#### Step 4: 단일 곡 테스트 (예상 5분)
- [ ] 1곡으로 동작 확인
- [ ] LRC 파일 재생 테스트

#### Step 5: 배치 실행 (예상 3분)
- [ ] 전체 곡 처리
- [ ] 결과 검증

**총 예상 시간**: 50분

### 10.2 선택적 기능 (Phase 2)
- [ ] Demucs 보컬 분리 통합
- [ ] 단어별 LRC 출력 옵션
- [ ] 설정 파일 지원

---

## 11. 산출물

### 11.1 최종 산출물
1. **sync_suisei.py** - 메인 배치 처리 스크립트
2. **output/*.lrc** - 생성된 LRC 자막 파일들
3. **README_USAGE.md** - 사용 가이드 (선택)

### 11.2 디렉토리 구조 (최종)
```
suisei_lyrics/
├── songs/                      # 입력: MP3 파일
│   ├── stellar_stellar.mp3
│   ├── template.mp3
│   └── ...
├── lyrics/                     # 입력: 일본어 가사 (UTF-8)
│   ├── stellar_stellar.txt
│   ├── template.txt
│   └── ...
├── output/                     # 출력: LRC 자막 파일
│   ├── stellar_stellar.lrc
│   ├── template.lrc
│   └── ...
├── sync_suisei.py             # 메인 스크립트
├── SPEC.md                    # 이 문서
├── progress.md                # 구현 진행 상황
└── CLAUDE.md                  # 프로젝트 가이드라인
```

---

## 12. 성공 기준

### 12.1 필수 성공 기준
- [x] 모든 입력 곡에 대해 LRC 파일 생성 완료
- [ ] 타임스탬프 정확도 ±0.3초 이내 (수동 검증)
- [ ] 일본어 텍스트 깨짐 없음
- [ ] GPU 가속 정상 동작 (CUDA 활용)
- [ ] 10곡 기준 3분 이내 처리 완료

### 12.2 권장 성공 기준
- [ ] 평균 처리 시간 15초/곡 이하
- [ ] 에러 발생 시에도 나머지 곡 처리 계속
- [ ] 명확한 에러 메시지 출력

### 12.3 최종 검증 방법
1. ✅ **기능 검증**: 음악 플레이어(VLC, foobar2000 등)에서 3곡 재생
2. ✅ **정확도 검증**: 노래 들으며 타임스탬프 일치 확인
3. ✅ **성능 검증**: 총 처리 시간 측정
4. ✅ **안정성 검증**: 스크립트 2회 연속 실행 (멱등성)

---

## 부록 A: 주요 코드 패턴

### A.1 환경 검증 패턴
```python
import torch

def verify_environment():
    """GPU 및 CUDA 확인"""
    assert torch.cuda.is_available(), "❌ CUDA를 사용할 수 없습니다!"

    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"✅ GPU 감지: {gpu_name}")
    print(f"✅ VRAM: {vram:.1f}GB")

    if vram < 8:
        print("⚠️ 경고: VRAM 8GB 미만입니다. 처리 중 메모리 부족 발생 가능.")
```

### A.2 파일 매칭 패턴
```python
from pathlib import Path

def verify_files(songs_dir, lyrics_dir):
    """MP3-가사 파일 매칭"""
    songs_path = Path(songs_dir)
    lyrics_path = Path(lyrics_dir)

    matched = []

    for mp3 in songs_path.glob('*.mp3'):
        name = mp3.stem
        txt = lyrics_path / f"{name}.txt"

        if txt.exists():
            matched.append({
                'name': name,
                'mp3': mp3,
                'lyrics': txt,
                'output': Path('output') / f"{name}.lrc"
            })
            print(f"✅ 매칭 완료: {name}")
        else:
            print(f"⚠️ 가사 누락: {name}")

    return matched
```

### A.3 단일 곡 처리 패턴
```python
import time

def process_song(model, mp3_path, lyrics_path, output_path):
    """1곡 처리"""
    try:
        # 가사 읽기
        with open(lyrics_path, 'r', encoding='utf-8') as f:
            lyrics = f.read().strip()

        lines = len([l for l in lyrics.split('\n') if l.strip()])
        print(f"📝 가사 라인: {lines}개")

        # 정렬
        start = time.time()
        result = model.align(
            str(mp3_path),
            lyrics,
            language='ja'
        )
        elapsed = time.time() - start

        # 저장
        result.to_srt_vtt(str(output_path), word_level=False)

        print(f"✅ 완료: {output_path}")
        print(f"⏱️ 소요시간: {elapsed:.1f}초\n")

        return {'success': True, 'time': elapsed}

    except Exception as e:
        print(f"❌ 오류: {e}\n")
        return {'success': False, 'error': str(e)}
```

---

## 부록 B: 예상 출력 예시

### B.1 정상 실행 출력
```
============================================================
🎵 호시마치 스이세이 가사 싱크 시작
============================================================

✅ GPU 감지: NVIDIA GeForce RTX 3070 Ti
✅ VRAM: 8.0GB

🔄 large-v3 모델 로딩 중... (첫 실행시 2.9GB 다운로드)
✅ 모델 로딩 완료!

------------------------------------------------------------
파일 검증 중...
------------------------------------------------------------
✅ 매칭 완료: stellar_stellar
✅ 매칭 완료: template
✅ 매칭 완료: ghost

총 처리 대상: 3곡

[1/3] 처리 중: stellar_stellar
------------------------------------------------------------
📝 가사 라인: 48개
⏳ 정렬 중... (GPU)
✅ 완료: output/stellar_stellar.lrc
⏱️ 소요시간: 12.3초

[2/3] 처리 중: template
------------------------------------------------------------
📝 가사 라인: 52개
⏳ 정렬 중... (GPU)
✅ 완료: output/template.lrc
⏱️ 소요시간: 13.1초

[3/3] 처리 중: ghost
------------------------------------------------------------
📝 가사 라인: 45개
⏳ 정렬 중... (GPU)
✅ 완료: output/ghost.lrc
⏱️ 소요시간: 11.8초

============================================================
✅ 전체 처리 완료!
============================================================
총 곡 수: 3곡
성공: 3곡
실패: 0곡
총 소요시간: 37.2초
평균 처리 시간: 12.4초/곡
============================================================
```

### B.2 에러 발생 시 출력
```
[2/3] 처리 중: template
------------------------------------------------------------
❌ 오류: [Errno 2] No such file or directory: 'lyrics/template.txt'

⚠️ template 처리 실패, 다음 곡으로 계속...

[3/3] 처리 중: ghost
------------------------------------------------------------
✅ 완료: output/ghost.lrc
⏱️ 소요시간: 11.8초

============================================================
⚠️ 일부 오류 발생
============================================================
총 곡 수: 3곡
성공: 2곡
실패: 1곡
  - template: 가사 파일 없음
총 소요시간: 24.1초
============================================================
```

---

## 변경 이력
| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| 1.0.0 | 2025-12-28 | 초기 기획서 작성 |

---

**문서 종료**
