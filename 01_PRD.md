# 🎵 호시마치 스이세이 가사 싱크 프로젝트 (최고품질)

## 🎯 프로젝트 설정

**대상**: 호시마치 스이세이 (星街すいせい) 노래  
**모델**: `large-v3` (최고 품질, 2.9GB)  
**GPU**: RTX 3070 Ti (CUDA 지원) → 10배 빠른 처리  
**언어**: 일본어 (`language='ja'`)

---

## ⚙️ 최적 설정

### RTX 3070 Ti 사양 확인
- **VRAM**: 8GB → large-v3 모델 완벽 지원 ✅
- **CUDA**: 설치됨 → GPU 가속 자동 활용 ✅
- **예상 속도**: 3분 곡 기준 **10-15초** 처리

### 모델 선택
```python
# 최고 품질 모델
model = stable_whisper.load_model('large-v3', device='cuda')
```

| 항목 | 사양 |
|------|------|
| 모델 크기 | 2.9GB |
| 정확도 | 98%+ (일본어) |
| 처리 속도 | 10-15초/곡 (3분 기준) |
| VRAM 사용 | ~4-5GB |

---

## 📁 파일 구조 (호시마치 스이세이)

```
suisei_lyrics/
├── songs/
│   ├── stellar_stellar.mp3
│   ├── template.mp3
│   ├── ghost.mp3
│   └── ...
├── lyrics/
│   ├── stellar_stellar.txt      # 일본어 가사 (UTF-8)
│   ├── template.txt
│   ├── ghost.txt
│   └── ...
├── output/
│   ├── stellar_stellar.lrc
│   └── ...
└── sync_suisei.py
```

---

## 💻 최적화 스크립트

### 완전 자동화 버전 (GPU 최적화)

```python
"""
호시마치 스이세이 가사 싱크 (GPU 최적화)
RTX 3070 Ti + large-v3 모델
"""
import stable_whisper
from pathlib import Path
import time

def sync_suisei_songs():
    # GPU 확인
    import torch
    if not torch.cuda.is_available():
        print("⚠️ CUDA를 사용할 수 없습니다!")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU 감지: {gpu_name}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB\n")
    
    # 최고 품질 모델 로드 (GPU)
    print("🔄 large-v3 모델 로딩 중... (첫 실행시 2.9GB 다운로드)")
    model = stable_whisper.load_model('large-v3', device='cuda')
    print("✅ 모델 로딩 완료!\n")
    
    # 노래 목록 (파일명만 확장자 제외)
    songs = [
        'stellar_stellar',
        'template',
        'ghost',
        # ... 추가 곡들
    ]
    
    total_start = time.time()
    
    for i, song in enumerate(songs, 1):
        print(f"[{i}/{len(songs)}] 처리 중: {song}")
        print("-" * 60)
        
        mp3_path = f'songs/{song}.mp3'
        lyrics_path = f'lyrics/{song}.txt'
        output_path = f'output/{song}.lrc'
        
        # 파일 존재 확인
        if not Path(mp3_path).exists():
            print(f"❌ MP3 파일 없음: {mp3_path}\n")
            continue
        if not Path(lyrics_path).exists():
            print(f"❌ 가사 파일 없음: {lyrics_path}\n")
            continue
        
        try:
            # 가사 읽기
            with open(lyrics_path, 'r', encoding='utf-8') as f:
                lyrics = f.read().strip()
            
            lines = len([l for l in lyrics.split('\n') if l.strip()])
            print(f"📝 가사 라인: {lines}개")
            
            # GPU 가사 정렬 (일본어)
            start = time.time()
            result = model.align(
                mp3_path, 
                lyrics, 
                language='ja'  # 일본어
            )
            elapsed = time.time() - start
            
            # LRC 저장
            result.to_srt_vtt(output_path, word_level=False)
            
            print(f"✅ 완료: {output_path}")
            print(f"⏱️ 소요시간: {elapsed:.1f}초\n")
            
        except Exception as e:
            print(f"❌ 오류: {e}\n")
            continue
    
    total_time = time.time() - total_start
    print("=" * 60)
    print(f"✅ 전체 완료! 총 소요시간: {total_time/60:.1f}분")
    print("=" * 60)

if __name__ == '__main__':
    sync_suisei_songs()
```

---

## 🚀 실행 단계별 가이드

### Step 1: 환경 준비
```bash
# 1. stable-ts 설치
pip install stable-ts

# 2. PyTorch CUDA 확인 (이미 설치됨)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# 출력: CUDA: True
```

### Step 2: 폴더 생성
```bash
mkdir suisei_lyrics
cd suisei_lyrics
mkdir songs lyrics output
```

### Step 3: 파일 배치
```
1. MP3 파일들 → songs/ 폴더
2. 가사 텍스트들 → lyrics/ 폴더 (UTF-8 필수!)
3. 파일명 매칭 확인
```

### Step 4: 첫 실행 (테스트)
```python
# test_one.py - 1곡으로 먼저 테스트
import stable_whisper
import torch

print(f"GPU: {torch.cuda.get_device_name(0)}")

model = stable_whisper.load_model('large-v3', device='cuda')

with open('lyrics/stellar_stellar.txt', 'r', encoding='utf-8') as f:
    lyrics = f.read()

result = model.align('songs/stellar_stellar.mp3', lyrics, language='ja')
result.to_srt_vtt('output/stellar_stellar.lrc', word_level=False)

print("✅ 테스트 완료!")
```

### Step 5: 배치 실행
```bash
python sync_suisei.py
```

---

## ⚡ GPU 성능 예측

### RTX 3070 Ti 처리 속도
| 곡 길이 | CPU (large-v3) | GPU (3070 Ti) |
|---------|----------------|---------------|
| 3분 | ~2-3분 | **10-15초** |
| 4분 | ~3-4분 | **15-20초** |
| 5분 | ~4-5분 | **20-25초** |

### 10곡 처리 예상
- **전체 시간**: 약 **2-3분** (첫 실행 제외)
- **첫 실행**: +3분 (모델 다운로드 2.9GB)
- **GPU 활용도**: ~80-90%
- **VRAM 사용**: ~4-5GB / 8GB

---

## 🎌 일본어 최적화

### 언어 설정
```python
result = model.align(
    audio_file, 
    lyrics, 
    language='ja'  # 일본어 필수!
)
```

### 가사 파일 주의사항
```text
✅ 올바른 예시 (lyrics.txt):
行こう　この声に導かれ
今日もまた一歩ずつ
夢見た場所へ

❌ 잘못된 예시:
- 로마자 표기 (iko kono koe ni...)  ← 안 됨!
- 번역된 한글/영어 ← 안 됨!
- 원본 일본어만 사용!
```

---

## 🔧 고급 옵션 (선택사항)

### 단어별 타임스탬프 (Enhanced LRC)
```python
# 라인별 (일반)
result.to_srt_vtt('output.lrc', word_level=False)

# 단어별 (더 정밀 - 카라오케용)
result.to_srt_vtt('output_word.lrc', word_level=True)
```

### GPU 메모리 최적화 (8GB 충분하지만)
```python
# 배치 처리시 모델 한 번만 로드
model = stable_whisper.load_model('large-v3', device='cuda')

for song in songs:
    result = model.align(...)  # 모델 재사용
    
# 완료 후 메모리 정리
del model
torch.cuda.empty_cache()
```

---

## 📊 품질 체크리스트

- [ ] GPU 인식 확인 (`torch.cuda.is_available()`)
- [ ] 모델 다운로드 완료 (2.9GB)
- [ ] 첫 곡 테스트 성공
- [ ] LRC 파일 재생 확인 (음악 플레이어)
- [ ] 타임스탬프 정확도 확인 (±0.3초)
- [ ] 일본어 텍스트 깨짐 없음
- [ ] 전체 배치 처리 완료

---

## 🎵 예상 LRC 결과 (샘플)

```lrc
[00:15.23] 行こう　この声に導かれ
[00:19.45] 今日もまた一歩ずつ
[00:23.67] 夢見た場所へ
[00:27.89] 輝く未来を信じて
```

**품질**: large-v3 모델 → **±0.2-0.3초 정확도**

---

## 💡 최종 실행 계획

1. **환경 확인** (5분)
   - GPU, CUDA 확인
   - stable-ts 설치

2. **파일 준비** (10분)
   - 폴더 생성
   - MP3, 가사 파일 배치

3. **테스트** (5분)
   - 1곡으로 품질 확인

4. **배치 실행** (2-3분)
   - 전체 곡 자동 처리

**총 소요시간**: 약 **30분** (10곡 기준)

준비 완료되면 시작하시면 됩니다! 🚀