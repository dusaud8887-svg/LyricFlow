# 🎯 호시마치 스이세이 자막 생성기 - 상세 구현 스펙

## 📋 문서 정보
- **프로젝트명**: sync_suisei.py
- **버전**: 1.0.0
- **작성일**: 2025-12-28
- **용도**: 구현 단계별 체크리스트 및 상세 스펙

---

## 🎯 구현 목표
일본어 노래 가사와 MP3 파일을 입력받아 **정확한 타임스탬프가 부여된 LRC 자막 파일**을 자동 생성하는 배치 처리 스크립트 구현.

**핵심 원칙**:
- ✅ **견고성**: 1회용이므로 오류 없이 한 번에 잘 동작해야 함
- ✅ **단순성**: 복잡한 구조보다 직관적인 코드
- ✅ **최고 품질**: large-v3 모델 + GPU 가속
- ✅ **명확한 피드백**: 진행 상황 실시간 출력

---

## 📂 프로젝트 구조

### 파일 구조
```
/home/user/mp3/
├── songs/                      # 입력: MP3 파일들
│   ├── stellar_stellar.mp3
│   ├── template.mp3
│   └── ghost.mp3
├── lyrics/                     # 입력: 일본어 가사 (UTF-8)
│   ├── stellar_stellar.txt
│   ├── template.txt
│   └── ghost.txt
├── output/                     # 출력: LRC 자막 파일들
│   └── (자동 생성됨)
├── sync_suisei.py             # 메인 스크립트 (구현 대상)
├── SPEC.md                    # 기획서
├── progress.md                # 이 문서
└── CLAUDE.md                  # 프로젝트 가이드라인
```

### 스크립트 구조 (`sync_suisei.py`)
```python
# ============================================================
# sync_suisei.py - 호시마치 스이세이 가사 싱크 스크립트
# ============================================================

# [Section 1] Import 및 상수 정의
import stable_whisper
import torch
from pathlib import Path
import time
import sys

# [Section 2] 환경 검증 함수
def verify_environment() -> bool:
    """GPU, CUDA, 폴더 존재 확인"""
    pass

# [Section 3] 파일 검증 함수
def verify_files(songs_dir: str, lyrics_dir: str) -> list[dict]:
    """MP3-가사 파일 매칭 및 목록 생성"""
    pass

# [Section 4] 단일 곡 처리 함수
def process_song(model, mp3_path: Path, lyrics_path: Path, output_path: Path) -> dict:
    """1곡 처리: 가사 정렬 + LRC 저장"""
    pass

# [Section 5] 요약 출력 함수
def print_summary(results: list[dict], total_time: float) -> None:
    """처리 결과 요약 출력"""
    pass

# [Section 6] 메인 함수
def main() -> None:
    """배치 처리 메인 로직"""
    pass

# [Section 7] 엔트리 포인트
if __name__ == '__main__':
    main()
```

---

## 🔧 구현 단계별 체크리스트

### Phase 1: 환경 설정
- [ ] Python 3.10+ 설치 확인
- [ ] PyTorch 설치: `pip install torch --index-url https://download.pytorch.org/whl/cu124`
- [ ] stable-ts 설치: `pip install stable-ts`
- [ ] GPU 드라이버 확인: `nvidia-smi`
- [ ] CUDA 버전 확인: `nvcc --version` 또는 `nvidia-smi`

### Phase 2: 폴더 생성
- [ ] `songs/` 폴더 생성 (또는 확인)
- [ ] `lyrics/` 폴더 생성 (또는 확인)
- [ ] `output/` 폴더 생성 (스크립트에서 자동 생성)

### Phase 3: 스크립트 구현
- [ ] Section 1: Import 및 상수 정의
- [ ] Section 2: `verify_environment()` 함수 구현
- [ ] Section 3: `verify_files()` 함수 구현
- [ ] Section 4: `process_song()` 함수 구현
- [ ] Section 5: `print_summary()` 함수 구현
- [ ] Section 6: `main()` 함수 구현
- [ ] Section 7: 엔트리 포인트 작성

### Phase 4: 테스트
- [ ] 단일 곡 테스트 (1곡)
- [ ] 배치 처리 테스트 (3곡)
- [ ] 에러 케이스 테스트 (가사 파일 누락)
- [ ] 재실행 테스트 (멱등성 확인)

### Phase 5: 품질 검증
- [ ] LRC 파일 재생 테스트 (VLC 등)
- [ ] 타임스탬프 정확도 확인 (수동)
- [ ] 일본어 인코딩 확인
- [ ] 처리 시간 측정

---

## 📝 상세 구현 스펙

### Section 1: Import 및 상수 정의

**체크리스트**:
- [ ] 필수 라이브러리 임포트
- [ ] 상수 정의 (경로)

**코드**:
```python
"""
호시마치 스이세이 가사 싱크 스크립트
MP3 + 일본어 가사 → LRC 자막 생성

사용법:
    python sync_suisei.py

요구사항:
    - Python 3.10+
    - stable-ts
    - PyTorch (CUDA)
    - RTX 3070 Ti (또는 동급 GPU)
"""

import stable_whisper
import torch
from pathlib import Path
import time
import sys

# 상수 정의
SONGS_DIR = 'songs'
LYRICS_DIR = 'lyrics'
OUTPUT_DIR = 'output'
MODEL_NAME = 'large-v3'
LANGUAGE = 'ja'
```

**검증**:
- [ ] 모든 임포트가 정상 동작하는지 확인:
  ```bash
  python -c "import stable_whisper, torch; print('OK')"
  ```

---

### Section 2: verify_environment() 함수

**목적**: 실행 전 필수 조건 확인 (GPU, CUDA, 폴더)

**함수 시그니처**:
```python
def verify_environment() -> bool:
    """
    환경 검증: GPU, CUDA, 폴더 존재 확인

    Returns:
        bool: 검증 성공 여부 (False면 프로그램 종료)
    """
```

**체크리스트**:
- [ ] CUDA 사용 가능 여부 확인
- [ ] GPU 이름 출력
- [ ] VRAM 용량 출력 (8GB 미만이면 경고)
- [ ] 폴더 존재 확인 (`songs/`, `lyrics/`)
- [ ] `output/` 폴더 자동 생성

**상세 구현**:
```python
def verify_environment() -> bool:
    """환경 검증: GPU, CUDA, 폴더 존재 확인"""

    print("=" * 60)
    print("🎵 호시마치 스이세이 가사 싱크 시작")
    print("=" * 60)
    print()

    # [1] CUDA 확인
    if not torch.cuda.is_available():
        print("❌ 오류: CUDA를 사용할 수 없습니다!")
        print("   GPU 드라이버 및 PyTorch CUDA 버전을 확인하세요.")
        print("   설치: pip install torch --index-url https://download.pytorch.org/whl/cu124")
        return False

    # [2] GPU 정보 출력
    gpu_name = torch.cuda.get_device_name(0)
    gpu_props = torch.cuda.get_device_properties(0)
    vram_gb = gpu_props.total_memory / 1024**3

    print(f"✅ GPU 감지: {gpu_name}")
    print(f"✅ VRAM: {vram_gb:.1f}GB")

    # [3] VRAM 경고
    if vram_gb < 8:
        print(f"⚠️ 경고: VRAM이 {vram_gb:.1f}GB입니다. (권장: 8GB 이상)")
        print("   처리 중 메모리 부족이 발생할 수 있습니다.")

    print()

    # [4] 폴더 확인
    songs_path = Path(SONGS_DIR)
    lyrics_path = Path(LYRICS_DIR)
    output_path = Path(OUTPUT_DIR)

    # songs/ 폴더 확인
    if not songs_path.exists():
        print(f"❌ 오류: '{SONGS_DIR}/' 폴더가 없습니다!")
        print(f"   MP3 파일들을 '{SONGS_DIR}/' 폴더에 넣어주세요.")
        return False

    # lyrics/ 폴더 확인
    if not lyrics_path.exists():
        print(f"❌ 오류: '{LYRICS_DIR}/' 폴더가 없습니다!")
        print(f"   가사 파일들을 '{LYRICS_DIR}/' 폴더에 넣어주세요.")
        return False

    # output/ 폴더 자동 생성
    if not output_path.exists():
        output_path.mkdir(parents=True)
        print(f"✅ '{OUTPUT_DIR}/' 폴더 생성 완료")

    return True
```

**테스트**:
```python
# 단위 테스트
if verify_environment():
    print("환경 검증 성공")
else:
    print("환경 검증 실패")
```

**예상 출력**:
```
============================================================
🎵 호시마치 스이세이 가사 싱크 시작
============================================================

✅ GPU 감지: NVIDIA GeForce RTX 3070 Ti
✅ VRAM: 8.0GB

✅ 'output/' 폴더 생성 완료
```

---

### Section 3: verify_files() 함수

**목적**: MP3-가사 파일 매칭 및 처리 목록 생성

**함수 시그니처**:
```python
def verify_files(songs_dir: str, lyrics_dir: str) -> list[dict]:
    """
    MP3-가사 파일 매칭 검증

    Args:
        songs_dir: MP3 파일 폴더 경로
        lyrics_dir: 가사 파일 폴더 경로

    Returns:
        list[dict]: 매칭된 파일 정보 리스트
            [
                {
                    'name': 'stellar_stellar',
                    'mp3': Path('songs/stellar_stellar.mp3'),
                    'lyrics': Path('lyrics/stellar_stellar.txt'),
                    'output': Path('output/stellar_stellar.lrc')
                },
                ...
            ]
    """
```

**체크리스트**:
- [ ] `songs/*.mp3` 파일 스캔
- [ ] 각 MP3에 대응하는 가사 파일 확인
- [ ] 매칭된 파일만 리스트에 추가
- [ ] 매칭 상태 출력 (성공/누락)
- [ ] 이미 존재하는 LRC 파일 스킵 (선택)

**상세 구현**:
```python
def verify_files(songs_dir: str, lyrics_dir: str) -> list[dict]:
    """MP3-가사 파일 매칭 검증"""

    print("-" * 60)
    print("📂 파일 검증 중...")
    print("-" * 60)

    songs_path = Path(songs_dir)
    lyrics_path = Path(lyrics_dir)
    output_path = Path(OUTPUT_DIR)

    matched = []
    missing_lyrics = []
    skipped = []

    # MP3 파일 스캔
    mp3_files = sorted(songs_path.glob('*.mp3'))

    if not mp3_files:
        print(f"⚠️ 경고: '{songs_dir}/' 폴더에 MP3 파일이 없습니다!")
        return []

    for mp3 in mp3_files:
        name = mp3.stem
        txt = lyrics_path / f"{name}.txt"
        lrc = output_path / f"{name}.lrc"

        # 가사 파일 확인
        if not txt.exists():
            print(f"⚠️ 가사 누락: {name} (MP3만 존재)")
            missing_lyrics.append(name)
            continue

        # 이미 존재하는 LRC 스킵 (선택적 기능)
        if lrc.exists():
            print(f"⏭️ 스킵 (이미 존재): {name}")
            skipped.append(name)
            continue

        # 매칭 성공
        matched.append({
            'name': name,
            'mp3': mp3,
            'lyrics': txt,
            'output': lrc
        })
        print(f"✅ 매칭 완료: {name}")

    print()
    print(f"총 처리 대상: {len(matched)}곡")
    if missing_lyrics:
        print(f"가사 누락: {len(missing_lyrics)}곡 ({', '.join(missing_lyrics)})")
    if skipped:
        print(f"스킵: {len(skipped)}곡 (이미 존재)")
    print()

    return matched
```

**옵션: 이미 존재하는 LRC 스킵 기능 제거**
```python
# 스킵 기능 제거하려면 아래 부분 삭제
if lrc.exists():
    print(f"⏭️ 스킵 (이미 존재): {name}")
    skipped.append(name)
    continue
```

**테스트**:
```python
# 단위 테스트
songs = verify_files('songs', 'lyrics')
for song in songs:
    print(f"  {song['name']}: {song['mp3']} + {song['lyrics']}")
```

**예상 출력**:
```
------------------------------------------------------------
📂 파일 검증 중...
------------------------------------------------------------
✅ 매칭 완료: stellar_stellar
✅ 매칭 완료: template
⚠️ 가사 누락: ghost (MP3만 존재)

총 처리 대상: 2곡
가사 누락: 1곡 (ghost)
```

---

### Section 4: process_song() 함수

**목적**: 1곡의 MP3 + 가사 → LRC 변환

**함수 시그니처**:
```python
def process_song(model, mp3_path: Path, lyrics_path: Path, output_path: Path) -> dict:
    """
    단일 곡 처리: 가사 정렬 + LRC 저장

    Args:
        model: stable_whisper.WhisperModel (large-v3)
        mp3_path: MP3 파일 경로
        lyrics_path: 가사 파일 경로 (UTF-8)
        output_path: 출력 LRC 파일 경로

    Returns:
        dict: 처리 결과
            {
                'success': bool,
                'time': float,  # 소요 시간 (초)
                'error': str    # 에러 메시지 (실패 시)
            }
    """
```

**체크리스트**:
- [ ] 가사 파일 읽기 (UTF-8)
- [ ] 가사 라인 수 계산 및 출력
- [ ] `model.align()` 호출 (일본어)
- [ ] LRC 파일 저장 (`word_level=False`)
- [ ] 처리 시간 측정 및 출력
- [ ] 에러 핸들링 (try-except)

**상세 구현**:
```python
def process_song(model, mp3_path: Path, lyrics_path: Path, output_path: Path) -> dict:
    """단일 곡 처리: 가사 정렬 + LRC 저장"""

    try:
        # [1] 가사 읽기 (UTF-8)
        with open(lyrics_path, 'r', encoding='utf-8') as f:
            lyrics = f.read().strip()

        # 빈 가사 확인
        if not lyrics:
            print(f"❌ 오류: 가사 파일이 비어 있습니다.")
            return {'success': False, 'error': '빈 가사 파일'}

        # 가사 라인 수 계산
        lines = len([l for l in lyrics.split('\n') if l.strip()])
        print(f"📝 가사 라인: {lines}개")

        # [2] 모델 정렬 (Forced Alignment)
        print(f"⏳ 정렬 중... (GPU)")
        start = time.time()

        result = model.align(
            str(mp3_path),
            lyrics,
            language=LANGUAGE  # 'ja' (일본어)
        )

        elapsed = time.time() - start

        # [3] LRC 저장
        result.to_srt_vtt(str(output_path), word_level=False)

        # [4] 결과 출력
        file_size = output_path.stat().st_size / 1024  # KB
        print(f"✅ 완료: {output_path}")
        print(f"⏱️ 소요시간: {elapsed:.1f}초")
        print(f"📊 LRC 크기: {file_size:.1f} KB")
        print()

        return {
            'success': True,
            'time': elapsed,
            'lines': lines,
            'size': file_size
        }

    except UnicodeDecodeError as e:
        print(f"❌ 오류: 인코딩 오류 (UTF-8 필요)")
        print(f"   {e}")
        print()
        return {'success': False, 'error': f'인코딩 오류: {e}'}

    except FileNotFoundError as e:
        print(f"❌ 오류: 파일 없음")
        print(f"   {e}")
        print()
        return {'success': False, 'error': f'파일 없음: {e}'}

    except Exception as e:
        print(f"❌ 오류: {type(e).__name__}")
        print(f"   {e}")
        print()
        return {'success': False, 'error': f'{type(e).__name__}: {e}'}
```

**에러 핸들링 강화 (선택)**:
```python
# BOM 처리 (UTF-8-sig)
with open(lyrics_path, 'r', encoding='utf-8-sig') as f:
    lyrics = f.read().strip()

# 또는 BOM 제거
lyrics = lyrics.lstrip('\ufeff')
```

**테스트**:
```python
# 단위 테스트
model = stable_whisper.load_model('large-v3', device='cuda')
result = process_song(
    model,
    Path('songs/stellar_stellar.mp3'),
    Path('lyrics/stellar_stellar.txt'),
    Path('output/stellar_stellar.lrc')
)
print(result)
```

**예상 출력**:
```
📝 가사 라인: 48개
⏳ 정렬 중... (GPU)
✅ 완료: output/stellar_stellar.lrc
⏱️ 소요시간: 12.3초
📊 LRC 크기: 2.1 KB
```

---

### Section 5: print_summary() 함수

**목적**: 전체 처리 결과 요약 출력

**함수 시그니처**:
```python
def print_summary(results: list[dict], total_time: float) -> None:
    """
    처리 결과 요약 출력

    Args:
        results: process_song() 결과 리스트
        total_time: 전체 소요 시간 (초)
    """
```

**체크리스트**:
- [ ] 성공/실패 곡 수 집계
- [ ] 실패한 곡 목록 출력
- [ ] 총 소요 시간 출력
- [ ] 평균 처리 시간 출력

**상세 구현**:
```python
def print_summary(results: list[dict], total_time: float) -> None:
    """처리 결과 요약 출력"""

    print("=" * 60)

    # 성공/실패 집계
    success_count = sum(1 for r in results if r['success'])
    fail_count = len(results) - success_count

    if fail_count == 0:
        print("✅ 전체 처리 완료!")
    else:
        print("⚠️ 일부 오류 발생")

    print("=" * 60)
    print(f"총 곡 수: {len(results)}곡")
    print(f"성공: {success_count}곡")
    print(f"실패: {fail_count}곡")

    # 실패한 곡 목록
    if fail_count > 0:
        print()
        print("실패한 곡:")
        for i, r in enumerate(results):
            if not r['success']:
                song_name = r.get('name', f'곡{i+1}')
                error = r.get('error', '알 수 없는 오류')
                print(f"  - {song_name}: {error}")

    # 소요 시간
    print()
    print(f"총 소요시간: {total_time:.1f}초 ({total_time/60:.1f}분)")

    if success_count > 0:
        avg_time = sum(r['time'] for r in results if r['success']) / success_count
        print(f"평균 처리 시간: {avg_time:.1f}초/곡")

    print("=" * 60)
```

**테스트**:
```python
# 단위 테스트
test_results = [
    {'success': True, 'time': 12.3, 'name': 'stellar_stellar'},
    {'success': True, 'time': 10.8, 'name': 'template'},
    {'success': False, 'error': '인코딩 오류', 'name': 'ghost'}
]
print_summary(test_results, 37.2)
```

**예상 출력**:
```
============================================================
⚠️ 일부 오류 발생
============================================================
총 곡 수: 3곡
성공: 2곡
실패: 1곡

실패한 곡:
  - ghost: 인코딩 오류

총 소요시간: 37.2초 (0.6분)
평균 처리 시간: 11.6초/곡
============================================================
```

---

### Section 6: main() 함수

**목적**: 전체 배치 처리 오케스트레이션

**함수 시그니처**:
```python
def main() -> None:
    """배치 처리 메인 로직"""
```

**체크리스트**:
- [ ] 환경 검증 (`verify_environment()`)
- [ ] 파일 검증 (`verify_files()`)
- [ ] 모델 로드 (large-v3, CUDA)
- [ ] 배치 처리 루프
- [ ] 요약 출력 (`print_summary()`)
- [ ] GPU 메모리 정리

**상세 구현**:
```python
def main() -> None:
    """배치 처리 메인 로직"""

    # [1] 환경 검증
    if not verify_environment():
        sys.exit(1)

    # [2] 파일 검증
    songs = verify_files(SONGS_DIR, LYRICS_DIR)

    if not songs:
        print("❌ 처리할 곡이 없습니다.")
        print(f"   '{SONGS_DIR}/' 폴더에 MP3 파일을 추가하고")
        print(f"   '{LYRICS_DIR}/' 폴더에 대응하는 가사 파일(.txt)을 추가하세요.")
        sys.exit(1)

    # [3] 모델 로드
    print("🔄 large-v3 모델 로딩 중...")
    print("   (첫 실행시 2.9GB 다운로드됩니다)")
    print()

    try:
        model = stable_whisper.load_model(MODEL_NAME, device='cuda')
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        sys.exit(1)

    print("✅ 모델 로딩 완료!")
    print()

    # [4] 배치 처리
    total_start = time.time()
    results = []

    for i, song in enumerate(songs, 1):
        print(f"[{i}/{len(songs)}] 처리 중: {song['name']}")
        print("-" * 60)

        result = process_song(
            model,
            song['mp3'],
            song['lyrics'],
            song['output']
        )

        # 결과에 곡 이름 추가 (요약용)
        result['name'] = song['name']
        results.append(result)

    total_time = time.time() - total_start

    # [5] 요약 출력
    print_summary(results, total_time)

    # [6] GPU 메모리 정리
    del model
    torch.cuda.empty_cache()
```

**에러 핸들링 강화 (선택)**:
```python
# Ctrl+C 중단 처리
try:
    for i, song in enumerate(songs, 1):
        # ...
except KeyboardInterrupt:
    print("\n\n⚠️ 사용자가 중단했습니다.")
    print_summary(results, time.time() - total_start)
    sys.exit(0)
```

**테스트**:
```bash
# 전체 스크립트 실행
python sync_suisei.py
```

---

### Section 7: 엔트리 포인트

**체크리스트**:
- [ ] `if __name__ == '__main__':` 작성
- [ ] `main()` 호출

**상세 구현**:
```python
if __name__ == '__main__':
    main()
```

**완성된 스크립트 구조 확인**:
```bash
# 스크립트 구조 확인
grep -E "^def |^if __name__" sync_suisei.py

# 예상 출력:
# def verify_environment() -> bool:
# def verify_files(songs_dir: str, lyrics_dir: str) -> list[dict]:
# def process_song(model, mp3_path: Path, lyrics_path: Path, output_path: Path) -> dict:
# def print_summary(results: list[dict], total_time: float) -> None:
# def main() -> None:
# if __name__ == '__main__':
```

---

## 🧪 테스트 계획

### Test 1: 환경 테스트
**목적**: GPU, CUDA, 라이브러리 확인

**실행**:
```bash
# CUDA 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# 출력: CUDA: True

# GPU 정보
python -c "import torch; print(torch.cuda.get_device_name(0))"
# 출력: NVIDIA GeForce RTX 3070 Ti

# stable-ts 확인
python -c "import stable_whisper; print('OK')"
# 출력: OK
```

**체크리스트**:
- [ ] `torch.cuda.is_available()` = True
- [ ] GPU 이름 출력 정상
- [ ] stable_whisper 임포트 성공

---

### Test 2: 단일 곡 테스트
**목적**: 전체 파이프라인 동작 확인

**준비**:
```bash
# 테스트 파일 배치
cp stellar_stellar.mp3 songs/
cp stellar_stellar.txt lyrics/
```

**실행**:
```bash
python sync_suisei.py
```

**검증**:
- [ ] `output/stellar_stellar.lrc` 파일 생성 확인
- [ ] LRC 파일 내용 확인 (타임스탬프 형식)
- [ ] 음악 플레이어에서 재생 테스트

**LRC 내용 확인**:
```bash
head -20 output/stellar_stellar.lrc

# 예상 출력:
# [00:15.23] 行こう　この声に導かれ
# [00:19.45] 今日もまた一歩ずつ
# ...
```

---

### Test 3: 배치 처리 테스트
**목적**: 여러 곡 동시 처리 확인

**준비**:
```bash
# 3곡 준비
cp stellar_stellar.mp3 template.mp3 ghost.mp3 songs/
cp stellar_stellar.txt template.txt ghost.txt lyrics/
```

**실행**:
```bash
python sync_suisei.py
```

**검증**:
- [ ] 3개 LRC 파일 모두 생성
- [ ] 각 곡 처리 시간 출력 확인
- [ ] 전체 요약 리포트 출력 확인

---

### Test 4: 에러 케이스 테스트
**목적**: 에러 핸들링 확인

**케이스 1: 가사 파일 누락**
```bash
# ghost.txt 삭제
rm lyrics/ghost.txt

python sync_suisei.py
```

**예상 출력**:
```
⚠️ 가사 누락: ghost (MP3만 존재)
총 처리 대상: 2곡
```

**케이스 2: 인코딩 오류**
```bash
# 잘못된 인코딩 파일 생성
echo "test" | iconv -f UTF-8 -t EUC-KR > lyrics/bad_encoding.txt
cp stellar_stellar.mp3 songs/bad_encoding.mp3

python sync_suisei.py
```

**예상 출력**:
```
❌ 오류: 인코딩 오류 (UTF-8 필요)
```

**체크리스트**:
- [ ] 가사 누락 시 경고 출력 + 스킵
- [ ] 인코딩 오류 시 에러 메시지 출력
- [ ] 에러 발생 시에도 나머지 곡 계속 처리

---

### Test 5: 재실행 테스트 (멱등성)
**목적**: 같은 입력에 대해 같은 결과 생성 확인

**실행**:
```bash
# 1차 실행
python sync_suisei.py > run1.log

# 2차 실행
python sync_suisei.py > run2.log

# 결과 비교
diff output/stellar_stellar.lrc output/stellar_stellar.lrc.backup
```

**체크리스트**:
- [ ] LRC 파일 내용 동일 (타임스탬프 동일)
- [ ] 2차 실행 시 "스킵 (이미 존재)" 메시지 출력 (스킵 기능 활성화 시)

---

### Test 6: 품질 검증
**목적**: LRC 파일 품질 확인

**방법 1: 수동 재생 테스트**
```bash
# VLC, foobar2000, MusicBee 등 LRC 지원 플레이어 사용
vlc stellar_stellar.mp3
# LRC 파일 자동 로드 확인
```

**방법 2: 타임스탬프 확인**
```bash
# LRC 파일 샘플 확인
head -10 output/stellar_stellar.lrc

# 타임스탬프 형식 확인 (정규식)
grep -E '^\[[0-9]{2}:[0-9]{2}\.[0-9]{2}\]' output/stellar_stellar.lrc | head -5
```

**체크리스트**:
- [ ] 타임스탬프 형식: `[mm:ss.xx]`
- [ ] 일본어 텍스트 깨짐 없음
- [ ] 음악과 타임스탬프 일치 (±0.3초 이내)

---

## 📊 성능 측정

### 측정 항목
1. **처리 속도**: 3분 곡 기준 15초 이내
2. **GPU 사용률**: 80% 이상
3. **VRAM 사용**: 5GB 이하
4. **배치 처리**: 10곡 기준 3분 이내

### 측정 방법

**처리 속도**:
```python
# 스크립트에서 자동 측정 (process_song 내부)
⏱️ 소요시간: 12.3초
```

**GPU 사용률**:
```bash
# 별도 터미널에서 모니터링
watch -n 1 nvidia-smi

# 처리 중 GPU 사용률 확인
# 예상: 80-90%
```

**VRAM 사용**:
```bash
# 처리 중 VRAM 확인
nvidia-smi

# 예상 출력:
# |  GPU       Memory-Usage |
# |    0       4521MiB / 8192MiB |
```

**배치 처리 시간**:
```bash
# 10곡 처리 후 요약 확인
총 소요시간: 135.2초 (2.3분)
평균 처리 시간: 13.5초/곡
```

---

## 🐛 트러블슈팅

### 문제 1: CUDA out of memory
**증상**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**해결책**:
1. GPU 메모리 정리
   ```python
   torch.cuda.empty_cache()
   ```

2. 더 작은 모델 사용 (최후의 수단)
   ```python
   model = stable_whisper.load_model('medium', device='cuda')
   ```

3. 다른 GPU 프로세스 종료
   ```bash
   nvidia-smi  # GPU 사용 프로세스 확인
   kill <PID>  # 해당 프로세스 종료
   ```

---

### 문제 2: 인코딩 오류
**증상**:
```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff
```

**해결책**:
1. UTF-8-sig 사용 (BOM 처리)
   ```python
   with open(lyrics_path, 'r', encoding='utf-8-sig') as f:
       lyrics = f.read()
   ```

2. BOM 제거
   ```bash
   sed -i '1s/^\xEF\xBB\xBF//' lyrics/*.txt
   ```

3. 파일 인코딩 확인
   ```bash
   file -i lyrics/stellar_stellar.txt
   # 출력: text/plain; charset=utf-8
   ```

---

### 문제 3: 모델 다운로드 실패
**증상**:
```
HTTPError: 404 Client Error: Not Found
```

**해결책**:
1. 인터넷 연결 확인

2. 수동 다운로드 후 캐시 위치에 배치
   ```bash
   # 모델 캐시 위치 확인
   python -c "import os; print(os.path.expanduser('~/.cache/huggingface'))"

   # 수동 다운로드
   wget https://huggingface.co/openai/whisper-large-v3/resolve/main/model.bin
   ```

3. 재시도
   ```bash
   python sync_suisei.py
   ```

---

### 문제 4: 타임스탬프 부정확
**증상**: LRC 파일의 타임스탬프가 실제 음악과 1초 이상 차이

**원인**:
- 가사 텍스트가 실제 노래와 다름
- 음원 버전 불일치 (리믹스, 라이브 등)

**해결책**:
1. 가사 원본 확인
   - 공식 가사 사이트에서 확인
   - 일본어 원문 사용 (번역 X)

2. 음원 확인
   - 오리지널 버전 사용
   - 리믹스/라이브 버전 제외

3. 수동 조정 (최후의 수단)
   - LRC 파일을 텍스트 에디터로 열어 수동 조정

---

## ✅ 완료 기준

### Phase 1: 구현 완료
- [ ] `sync_suisei.py` 스크립트 작성 완료
- [ ] 모든 함수 구현 완료
- [ ] 에러 핸들링 구현 완료

### Phase 2: 테스트 완료
- [ ] 단일 곡 테스트 통과
- [ ] 배치 처리 테스트 통과
- [ ] 에러 케이스 테스트 통과
- [ ] 재실행 테스트 통과

### Phase 3: 품질 검증 완료
- [ ] LRC 파일 재생 테스트 (VLC 등)
- [ ] 타임스탬프 정확도 확인 (±0.3초)
- [ ] 일본어 인코딩 확인 (깨짐 없음)
- [ ] 처리 시간 측정 (3분 곡 15초 이내)

### Phase 4: 최종 검증 완료
- [ ] 10곡 배치 처리 성공
- [ ] 모든 LRC 파일 정상 생성
- [ ] 요약 리포트 정상 출력
- [ ] GPU 메모리 정리 확인

---

## 📝 진행 상황 추적

### 구현 진행률
```
[=============================================>            ] 80%

완료:
✅ 기획서 작성 (SPEC.md)
✅ 구현 스펙 작성 (progress.md)
⬜ 스크립트 구현 (sync_suisei.py) - 대기
⬜ 단위 테스트 - 대기
⬜ 통합 테스트 - 대기
⬜ 품질 검증 - 대기
```

### 다음 단계
1. **즉시 실행**: `sync_suisei.py` 스크립트 작성
2. **테스트**: 단일 곡으로 전체 파이프라인 검증
3. **배포**: 전체 곡 배치 처리 실행

---

## 📚 참고 자료

### 코드 예제
- SPEC.md 부록 A: 주요 코드 패턴
- docs/01_PRD.md: 완전 자동화 스크립트 예제
- CLAUDE.md: 핵심 코드 패턴

### 기술 문서
- docs/lyrics_sync_tech_guide_2025.md: stable-ts 상세 가이드
- stable-ts 공식 문서: https://github.com/jianfch/stable-ts

### 트러블슈팅
- SPEC.md 섹션 8: 위험 요소 및 대응
- progress.md 트러블슈팅 섹션

---

**다음 실행 명령**:
```bash
# 1. 스크립트 작성
# (이 문서를 참고하여 sync_suisei.py 작성)

# 2. 테스트 실행
python sync_suisei.py

# 3. 결과 확인
ls -lh output/
head -20 output/stellar_stellar.lrc
```

---

**문서 종료**
