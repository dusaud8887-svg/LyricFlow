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
    print()

    return matched


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


if __name__ == '__main__':
    main()
