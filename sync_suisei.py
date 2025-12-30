"""
호시마치 스이세이 가사 싱크 스크립트 (v2.1 Line-Preserve)
MP3 + 일본어 가사 → LRC 자막 생성

v2.1 신규 기능:
    - ⭐ 줄바꿈 보존 모드 (사용자 소절 유지)
    - 소절 기반 품질 검증

v2.0 기능 (유지):
    - Demucs 보컬 분리 (선택적)
    - VAD (Voice Activity Detection)
    - 세그먼트 최적화 (4단계 체인)
    - 품질 검증 및 경고
    - 프로파일 시스템 (ballad/normal/fast)
    - initial_prompt 최적화

사용법:
    python sync_suisei.py

요구사항:
    - Python 3.10+
    - stable-ts
    - PyTorch (CUDA)
    - RTX 3070 Ti (또는 동급 GPU)
    - (선택) demucs (보컬 분리용)
"""

import sys
from pathlib import Path
import time
import re
from typing import Optional

# 라이브러리 import 에러 처리
try:
    import torch
except ImportError:
    print("❌ 오류: PyTorch가 설치되지 않았습니다!")
    print("   설치: pip install torch --index-url https://download.pytorch.org/whl/cu124")
    sys.exit(1)

try:
    import stable_whisper
except ImportError:
    print("❌ 오류: stable-whisper가 설치되지 않았습니다!")
    print("   설치: pip install stable-ts")
    sys.exit(1)

# tqdm (선택적)
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# ============================================================
# 설정 (사용자 수정 가능)
# ============================================================

SONGS_DIR = 'songs'
LYRICS_DIR = 'lyrics'
OUTPUT_DIR = 'output'

# 모델 선택 (속도 vs 품질)
# 'large-v3': 최고 품질 (±0.2초), 느림
# 'large-v3-turbo': 6배 빠름, large-v2급 품질 (±0.3초)
MODEL_NAME = 'large-v3'

LANGUAGE = 'ja'

# Enhanced LRC 옵션 (단어별 타임스탬프 - 카라오케용)
# False: 일반 LRC (라인별)
# True: Enhanced LRC (단어별 - 더 정밀)
WORD_LEVEL_LRC = False

# 요약 로그 저장 여부
SAVE_SUMMARY_LOG = True
SUMMARY_LOG_FILE = 'summary.txt'

# ============================================================
# v2.0 고급 설정
# ============================================================

# ⭐ v2.1: 줄바꿈 보존 모드 (핵심 개선!)
# True: 가사 파일의 줄바꿈을 그대로 유지 → 소절별 타임스탬프 (권장!)
# False: 자동으로 세그먼트 분할 (글자 수 기반)
PRESERVE_LINES = True

# Demucs 보컬 분리 (최고 품질, 처리 시간 3배 증가)
# False: 비활성화 (기본, 빠름)
# True: 활성화 (보컬만 추출, WER 60% 감소)
USE_DEMUCS = False

# VAD (Voice Activity Detection) 사용
# True: 음성 구간만 처리 (환각 방지, 정확도 향상)
USE_VAD = True
VAD_THRESHOLD = 0.35  # 노래 권장값: 0.3~0.4

# 세그먼트 최적화 프로파일
# 'ballad': 발라드 (느린 템포, 긴 호흡)
# 'normal': 일반 곡 (표준 설정, 권장)
# 'fast': 빠른 곡 (랩, 업템포)
SEGMENT_PROFILE = 'normal'

# initial_prompt (일본어 인식 정확도 향상)
# 아티스트명, 곡명 등을 포함하면 고유명사 인식 개선
INITIAL_PROMPT = "以下は日本語の歌詞です。ホシマチスイセイの楽曲。"

# 품질 검증 옵션
ENABLE_QUALITY_VALIDATION = True  # 품질 경고 표시
WARN_LONG_SEGMENTS = 5.0  # 5초 이상 세그먼트 경고
WARN_AVG_CHARS = 35  # 평균 35자 이상 경고

# ============================================================
# 함수 정의
# ============================================================

def clean_lyrics(text: str, preserve_lines: bool = True) -> str:
    """
    가사 텍스트 전처리 및 정규화 (v2.1)

    처리 내용:
    - 전각 공백 → 반각 공백
    - 특수문자 제거 (괄호, 음악 기호 등)
    - 여러 공백 → 하나로 (줄바꿈 보존!)
    - 빈 라인 제거

    Args:
        text: 원본 가사 텍스트
        preserve_lines: True이면 사용자 줄바꿈 보존 (기본: True)
                       False이면 모든 공백 정규화 (자동 분할)

    Returns:
        정제된 가사 텍스트
    """
    # [1] 전각 공백을 반각 공백으로
    text = text.replace('\u3000', ' ')

    # [2] 특수문자 제거 (싱크 방해 요소)
    # 괄호류: （）()「」『』【】《》〈〉［］[]｛｝{}
    # 음악 기호: ♪♬♩♫～〜
    text = re.sub(r'[（）()「」『』【】《》〈〉［］\[\]｛｝\{\}]', '', text)
    text = re.sub(r'[♪♬♩♫～〜]', '', text)

    # [3] 줄바꿈 보존 처리 (v2.1 핵심 개선!)
    if preserve_lines:
        # 각 라인별로 처리 → 줄바꿈 보존!
        lines = []
        for line in text.split('\n'):
            # 각 라인 내부의 공백(스페이스, 탭)만 정규화
            line = re.sub(r'[ \t]+', ' ', line)
            line = line.strip()
            if line:
                lines.append(line)
        return '\n'.join(lines)
    else:
        # 기존 방식: 모든 공백을 정규화 (줄바꿈 포함)
        text = re.sub(r'\s+', ' ', text)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)


def optimize_segments(result, profile: str = 'normal', preserve_lines: bool = False):
    """
    세그먼트 4단계 최적화 체인 (v2.1)

    Args:
        result: stable-ts 결과 객체
        profile: 'ballad', 'normal', 'fast'
        preserve_lines: True이면 최소 최적화만 (줄바꿈 보존 모드)

    Returns:
        최적화된 result 객체 (in-place 수정)
    """
    # 프로파일 설정
    PROFILES = {
        'ballad': {
            'punctuation': [('。', ' '), ('、', ' '), ('？', ' '), ('！', ' '), ('…', ' ')],
            'gap_threshold': 2.5,
            'max_chars': 35,
            'merge_gap': 0.20,
        },
        'normal': {
            'punctuation': [('。', ' '), ('、', ' '), ('？', ' '), ('！', ' ')],
            'gap_threshold': 2.0,
            'max_chars': 30,
            'merge_gap': 0.15,
        },
        'fast': {
            'punctuation': [('。', ' '), ('、', ' ')],
            'gap_threshold': 1.5,
            'max_chars': 25,
            'merge_gap': 0.10,
        },
    }

    cfg = PROFILES.get(profile, PROFILES['normal'])

    # === 타임스탬프 보정 (항상 수행) ===
    result.clamp_max()

    # === v2.1: 줄바꿈 보존 모드 처리 ===
    if preserve_lines:
        # 줄바꿈 보존 모드: 최소 최적화만 수행
        # - 타임스탬프 보정만 수행 (위에서 완료)
        # - 세그먼트 분할/병합 스킵 (사용자 줄바꿈 보존!)
        return result

    # === 4단계 최적화 체인 (자동 분할 모드) ===

    # 1단계: 구두점으로 분할 (최우선)
    result.split_by_punctuation(cfg['punctuation'])

    # 2단계: 침묵 구간으로 분할
    result.split_by_gap(gap_threshold=cfg['gap_threshold'])

    # 3단계: 길이 제한
    result.split_by_length(
        max_chars=cfg['max_chars'],
        max_words=None,  # 일본어는 공백 없으므로 None
        even_split=True  # 균등 분할
    )

    # 4단계: 짧은 세그먼트 병합
    result.merge_by_gap(
        max_gap=cfg['merge_gap'],
        max_chars=cfg['max_chars']
    )

    return result


def validate_result(result, song_name: str) -> dict:
    """
    생성된 결과 품질 검증 (v2.1)

    Args:
        result: stable-ts 결과 객체
        song_name: 곡 이름

    Returns:
        dict: 품질 통계 및 경고
    """
    segments = result.segments

    if not segments:
        return {
            'total_segments': 0,
            'warnings': ['세그먼트가 비어 있음']
        }

    # 통계 계산
    durations = [seg.end - seg.start for seg in segments]
    char_counts = [len(seg.text) for seg in segments]

    stats = {
        'total_segments': len(segments),
        'avg_duration': sum(durations) / len(durations) if durations else 0,
        'avg_chars': sum(char_counts) / len(char_counts) if char_counts else 0,
        'min_duration': min(durations) if durations else 0,
        'max_duration': max(durations) if durations else 0,
        'long_segments': sum(1 for d in durations if d > WARN_LONG_SEGMENTS),
        'short_segments': sum(1 for d in durations if d < 0.5),
        'warnings': []
    }

    # === v2.1: 줄바꿈 보존 모드별 검증 ===
    if ENABLE_QUALITY_VALIDATION:
        if PRESERVE_LINES:
            # 줄바꿈 보존 모드: 소절 기반 검증
            if stats['long_segments'] > 0:
                stats['warnings'].append(f"긴 소절 {stats['long_segments']}개 ({WARN_LONG_SEGMENTS}초 이상) - 가사 파일 확인 권장")

            # 매우 짧은 세그먼트만 경고 (소절이 원래 짧을 수 있음)
            very_short = sum(1 for d in durations if d < 0.3)
            if very_short > 2:
                stats['warnings'].append(f"매우 짧은 소절 {very_short}개 (0.3초 미만) - 가사 파일 확인 권장")
        else:
            # 자동 분할 모드: 기존 검증
            if stats['long_segments'] > 0:
                stats['warnings'].append(f"긴 세그먼트 {stats['long_segments']}개 발견 ({WARN_LONG_SEGMENTS}초 이상)")

            if stats['short_segments'] > 3:
                stats['warnings'].append(f"짧은 세그먼트 {stats['short_segments']}개 발견 (0.5초 미만)")

            if stats['avg_chars'] > WARN_AVG_CHARS:
                stats['warnings'].append(f"평균 글자수 {stats['avg_chars']:.1f}자 (권장: {WARN_AVG_CHARS}자 이하)")

    return stats


def verify_environment() -> bool:
    """환경 검증: GPU, CUDA, 폴더 존재 확인"""

    print("=" * 60)
    print("🎵 호시마치 스이세이 가사 싱크 시작 (v2.1 Line-Preserve)")
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

    # [4] 설정 출력
    print(f"📊 설정:")
    print(f"   모델: {MODEL_NAME}")
    print(f"   ⭐ 줄바꿈 보존: {'활성화 (소절 유지)' if PRESERVE_LINES else '비활성화 (자동 분할)'}")
    print(f"   Enhanced LRC: {'활성화 (단어별)' if WORD_LEVEL_LRC else '비활성화 (라인별)'}")
    print(f"   Demucs 보컬 분리: {'활성화' if USE_DEMUCS else '비활성화'}")
    print(f"   VAD: {'활성화' if USE_VAD else '비활성화'} (임계값: {VAD_THRESHOLD})")
    print(f"   세그먼트 프로파일: {SEGMENT_PROFILE}")
    print(f"   품질 검증: {'활성화' if ENABLE_QUALITY_VALIDATION else '비활성화'}")
    print(f"   로그 저장: {'활성화' if SAVE_SUMMARY_LOG else '비활성화'}")
    print()

    # [5] 폴더 확인
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
    """
    단일 곡 처리: 가사 정렬 + 세그먼트 최적화 + LRC 저장 (v2.1)

    v2.1 개선사항:
    - ⭐ 줄바꿈 보존 모드 (사용자 소절 유지)
    - 가사 텍스트 전처리
    - VAD 및 고급 align() 옵션
    - Demucs 보컬 분리 (선택적)
    - 세그먼트 4단계 최적화
    - 품질 검증 및 경고
    """

    try:
        # [1] 가사 읽기 및 전처리
        try:
            with open(lyrics_path, 'r', encoding='utf-8-sig') as f:
                lyrics = f.read().strip()
        except UnicodeDecodeError:
            with open(lyrics_path, 'r', encoding='utf-8') as f:
                lyrics = f.read().strip()

        # BOM 제거
        lyrics = lyrics.lstrip('\ufeff')

        # 빈 가사 확인
        if not lyrics:
            print(f"❌ 오류: 가사 파일이 비어 있습니다.")
            return {'success': False, 'error': '빈 가사 파일'}

        # === v2.1: 가사 전처리 (줄바꿈 보존!) ===
        lyrics = clean_lyrics(lyrics, preserve_lines=PRESERVE_LINES)

        # 가사 라인 수 계산
        lyrics_lines = [line for line in lyrics.split('\n') if line.strip()]
        lines = len(lyrics_lines)
        preserve_status = "소절 보존" if PRESERVE_LINES else "자동 분할"
        print(f"📝 가사 라인: {lines}개 ({preserve_status}, 전처리 완료)")

        # [2] 모델 정렬 (Forced Alignment) - v2.0 개선
        demucs_status = "Demucs 활성화" if USE_DEMUCS else "기본"
        print(f"⏳ 정렬 중... ({demucs_status}, GPU)")
        start = time.time()

        # align() 옵션 준비
        align_options = {
            'language': LANGUAGE,
        }

        # initial_prompt 추가
        if INITIAL_PROMPT:
            align_options['initial_prompt'] = INITIAL_PROMPT

        # === v2.0: VAD 및 고급 옵션 (try-except로 안전하게) ===
        try:
            # Demucs 옵션
            if USE_DEMUCS:
                align_options['denoiser'] = 'demucs'
                align_options['denoiser_options'] = {'device': 'cuda'}

            # VAD 옵션
            if USE_VAD:
                align_options['vad'] = True
                align_options['vad_threshold'] = VAD_THRESHOLD
                align_options['suppress_silence'] = True

            # 환각 방지 옵션
            align_options['temperature'] = 0  # 결정론적
            align_options['condition_on_previous_text'] = False

            # regroup은 수동으로 할 것이므로 비활성화
            align_options['regroup'] = False

            result = model.align(str(mp3_path), lyrics, **align_options)

        except TypeError as e:
            # 일부 파라미터가 align()에서 미지원될 경우 기본으로 fallback
            print(f"   ⚠️ 일부 고급 옵션 미지원, 기본 모드로 실행")
            result = model.align(
                str(mp3_path),
                lyrics,
                language=LANGUAGE
            )

        elapsed = time.time() - start
        print(f"   ✓ 정렬 완료 ({elapsed:.1f}초)")

        # [3] 세그먼트 최적화 (v2.1: 줄바꿈 보존 고려!)
        if PRESERVE_LINES:
            # 줄바꿈 보존 모드: 최소 최적화만
            print(f"✂️ 타임스탬프 보정 중... (소절 보존 모드)")
            optimize_segments(result, profile=SEGMENT_PROFILE, preserve_lines=True)
            print(f"   ✓ 보정 완료 ({len(result.segments)}개 세그먼트 - 소절 유지)")
        else:
            # 자동 분할 모드: 4단계 최적화 체인
            print(f"✂️ 세그먼트 최적화 중... (프로파일: {SEGMENT_PROFILE})")
            optimize_segments(result, profile=SEGMENT_PROFILE, preserve_lines=False)
            print(f"   ✓ 최적화 완료 ({len(result.segments)}개 세그먼트)")

        # [4] 품질 검증
        validation = validate_result(result, mp3_path.stem)

        # 경고 출력
        if validation['warnings']:
            for warning in validation['warnings']:
                print(f"   ⚠️ {warning}")

        # [5] LRC 저장
        result.to_srt_vtt(str(output_path), word_level=WORD_LEVEL_LRC)

        # [6] 결과 출력
        file_size = output_path.stat().st_size / 1024  # KB
        lrc_type = "Enhanced (단어별)" if WORD_LEVEL_LRC else "일반 (라인별)"
        print(f"✅ 완료: {output_path}")
        print(f"   타입: {lrc_type}")
        print(f"   소요시간: {elapsed:.1f}초")
        print(f"   세그먼트: {validation['total_segments']}개")
        print(f"   평균 길이: {validation['avg_duration']:.1f}초")
        print(f"   평균 글자수: {validation['avg_chars']:.1f}자")
        print(f"   크기: {file_size:.1f} KB")
        print()

        return {
            'success': True,
            'time': elapsed,
            'lines': lines,
            'size': file_size,
            'lrc_type': lrc_type,
            'segments': validation['total_segments'],
            'avg_duration': validation['avg_duration'],
            'avg_chars': validation['avg_chars'],
            'warnings': validation['warnings']
        }

    except UnicodeDecodeError as e:
        print(f"❌ 오류: 인코딩 오류 (UTF-8 필요)")
        print(f"   파일: {lyrics_path}")
        print(f"   상세: {e}")
        print()
        return {'success': False, 'error': f'인코딩 오류'}

    except FileNotFoundError as e:
        print(f"❌ 오류: 파일 없음")
        print(f"   {e}")
        print()
        return {'success': False, 'error': f'파일 없음'}

    except Exception as e:
        print(f"❌ 오류: {type(e).__name__}")
        print(f"   파일: {mp3_path.name}")
        print(f"   상세: {e}")
        print()
        return {'success': False, 'error': f'{type(e).__name__}'}


def print_summary(results: list[dict], total_time: float, save_to_file: bool = False) -> None:
    """처리 결과 요약 출력 (및 파일 저장)"""

    # 요약 텍스트 생성
    summary_lines = []
    summary_lines.append("=" * 60)

    # 성공/실패 집계
    success_count = sum(1 for r in results if r.get('success', False))
    fail_count = len(results) - success_count

    if fail_count == 0:
        summary_lines.append("✅ 전체 처리 완료!")
    else:
        summary_lines.append("⚠️ 일부 오류 발생")

    summary_lines.append("=" * 60)
    summary_lines.append(f"총 곡 수: {len(results)}곡")
    summary_lines.append(f"성공: {success_count}곡")
    summary_lines.append(f"실패: {fail_count}곡")
    summary_lines.append(f"모델: {MODEL_NAME}")
    summary_lines.append(f"⭐ 줄바꿈 보존: {'활성화 (소절 유지)' if PRESERVE_LINES else '비활성화 (자동 분할)'}")
    summary_lines.append(f"LRC 타입: {'Enhanced (단어별)' if WORD_LEVEL_LRC else '일반 (라인별)'}")
    summary_lines.append(f"Demucs: {'활성화' if USE_DEMUCS else '비활성화'}")
    summary_lines.append(f"VAD: {'활성화' if USE_VAD else '비활성화'}")
    summary_lines.append(f"세그먼트 프로파일: {SEGMENT_PROFILE}")

    # 실패한 곡 목록
    if fail_count > 0:
        summary_lines.append("")
        summary_lines.append("실패한 곡:")
        for r in results:
            if not r.get('success', False):
                song_name = r.get('name', '알 수 없음')
                error = r.get('error', '알 수 없는 오류')
                summary_lines.append(f"  - {song_name}: {error}")

    # 소요 시간
    summary_lines.append("")
    summary_lines.append(f"총 소요시간: {total_time:.1f}초 ({total_time/60:.1f}분)")

    # 평균 처리 시간 (안전하게 계산)
    if success_count > 0:
        successful_times = [r.get('time', 0) for r in results if r.get('success', False) and 'time' in r]
        if successful_times:
            avg_time = sum(successful_times) / len(successful_times)
            summary_lines.append(f"평균 처리 시간: {avg_time:.1f}초/곡")

    # 성공한 곡 상세 (v2.0 개선)
    if success_count > 0:
        summary_lines.append("")
        summary_lines.append("성공한 곡:")
        for r in results:
            if r.get('success', False):
                song_name = r.get('name', '알 수 없음')
                elapsed = r.get('time', 0)
                lines_count = r.get('lines', 0)
                segments = r.get('segments', 0)
                avg_chars = r.get('avg_chars', 0)

                # v2.0 정보 포함
                detail = f"  ✓ {song_name}: {elapsed:.1f}초"
                if segments > 0:
                    detail += f", {segments}개 세그먼트"
                if avg_chars > 0:
                    detail += f", 평균 {avg_chars:.0f}자"

                summary_lines.append(detail)

    summary_lines.append("=" * 60)

    # 출력
    summary_text = '\n'.join(summary_lines)
    print(summary_text)

    # 파일 저장 (선택적)
    if save_to_file:
        try:
            with open(SUMMARY_LOG_FILE, 'w', encoding='utf-8') as f:
                f.write(f"호시마치 스이세이 가사 싱크 결과 요약\n")
                f.write(f"생성 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n")
                f.write(summary_text)
            print(f"\n📄 요약 로그 저장: {SUMMARY_LOG_FILE}")
        except Exception as e:
            print(f"\n⚠️ 요약 로그 저장 실패: {e}")


def main() -> None:
    """배치 처리 메인 로직"""

    try:
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
        print("🔄 모델 로딩 중...")
        print(f"   모델: {MODEL_NAME}")
        print("   (첫 실행시 자동 다운로드됩니다)")
        print()

        try:
            model = stable_whisper.load_model(MODEL_NAME, device='cuda')
        except RuntimeError as e:
            print(f"❌ 모델 로드 실패: GPU 메모리 부족 또는 CUDA 오류")
            print(f"   상세: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ 모델 로드 실패: {type(e).__name__}")
            print(f"   상세: {e}")
            print()
            print("해결 방법:")
            print("  1. 인터넷 연결 확인 (모델 다운로드)")
            print("  2. stable-ts 버전 확인: pip install --upgrade stable-ts")
            print("  3. GPU 메모리 확인: nvidia-smi")
            sys.exit(1)

        print("✅ 모델 로딩 완료!")
        print()

        # [4] 배치 처리
        total_start = time.time()
        results = []

        # tqdm 사용 가능하면 진행률 바 표시
        if TQDM_AVAILABLE:
            song_iter = tqdm(songs, desc="전체 진행", unit="곡")
        else:
            song_iter = songs

        for i, song in enumerate(song_iter, 1):
            if not TQDM_AVAILABLE:
                print(f"[{i}/{len(songs)}] 처리 중: {song['name']}")
                print("-" * 60)
            else:
                song_iter.set_description(f"처리 중: {song['name']}")

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
        print()  # 줄바꿈
        print_summary(results, total_time, save_to_file=SAVE_SUMMARY_LOG)

        # [6] GPU 메모리 정리
        del model
        torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자가 중단했습니다.")
        if 'results' in locals() and results:
            total_time = time.time() - total_start
            print_summary(results, total_time, save_to_file=SAVE_SUMMARY_LOG)
        sys.exit(0)


if __name__ == '__main__':
    main()
