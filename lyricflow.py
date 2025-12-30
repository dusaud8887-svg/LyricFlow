#!/usr/bin/env python3
"""
LyricFlow - AI-Powered Lyrics Synchronization Tool
Intelligently align lyrics with MP3 audio using Whisper AI

Interactive CLI for easy-to-use subtitle generation
Supports multiple languages with intelligent line preservation
"""

import sys
import os
from pathlib import Path

# 기존 모듈 import
try:
    import sync_suisei
    from sync_suisei import (
        verify_environment, verify_files, process_song, print_summary,
        stable_whisper, torch, time,
        SONGS_DIR, LYRICS_DIR, OUTPUT_DIR, MODEL_NAME, LANGUAGE,
        PRESERVE_LINES, WORD_LEVEL_LRC, USE_DEMUCS, USE_VAD, SEGMENT_PROFILE
    )
except ImportError as e:
    print(f"❌ Error: Failed to import core modules: {e}")
    print("   Make sure sync_suisei.py is in the same directory.")
    sys.exit(1)

# Global language setting (can be changed by user)
CURRENT_LANGUAGE = LANGUAGE


def print_banner():
    """프로그램 배너 출력"""
    banner = """
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║         ██╗  ██╗   ██╗██████╗ ██╗ ██████╗                 ║
║         ██║  ╚██╗ ██╔╝██╔══██╗██║██╔════╝                 ║
║         ██║   ╚████╔╝ ██████╔╝██║██║                      ║
║         ██║    ╚██╔╝  ██╔══██╗██║██║                      ║
║         ███████╗██║   ██║  ██║██║╚██████╗                 ║
║         ╚══════╝╚═╝   ╚═╝  ╚═╝╚═╝ ╚═════╝                 ║
║                                                            ║
║        ███████╗██╗      ██████╗ ██╗    ██╗                ║
║        ██╔════╝██║     ██╔═══██╗██║    ██║                ║
║        █████╗  ██║     ██║   ██║██║ █╗ ██║                ║
║        ██╔══╝  ██║     ██║   ██║██║███╗██║                ║
║        ██║     ███████╗╚██████╔╝╚███╔███╔╝                ║
║        ╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝                 ║
║                                                            ║
║           AI-Powered Lyrics Synchronization                ║
║                     v2.1 Line-Preserve                     ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
"""
    print(banner)
    print("  🎵 Let your lyrics flow with perfect timing")
    print()


def print_menu():
    """메인 메뉴 출력"""
    print("=" * 60)
    print("📋 MENU")
    print("=" * 60)
    print("  [1] 🚀 Batch Process (All songs in folder)")
    print("  [2] 🎯 Single Song Process")
    print("  [3] 🌍 Change Language")
    print("  [4] ⚙️  View Current Settings")
    print("  [5] 📊 System Information")
    print("  [0] 🚪 Exit")
    print("=" * 60)


def show_settings():
    """현재 설정 표시"""
    global CURRENT_LANGUAGE
    print("\n" + "=" * 60)
    print("⚙️  CURRENT SETTINGS")
    print("=" * 60)
    print(f"  Model: {MODEL_NAME}")
    print(f"  🌍 Language: {CURRENT_LANGUAGE.upper()}")
    print(f"  ⭐ Line Preservation: {'✅ ON (preserves verse structure)' if PRESERVE_LINES else '❌ OFF (auto-split)'}")
    print(f"  LRC Type: {'Enhanced (word-level)' if WORD_LEVEL_LRC else 'Standard (line-level)'}")
    print(f"  Demucs Vocal Separation: {'✅ ON' if USE_DEMUCS else '❌ OFF'}")
    print(f"  VAD (Voice Activity Detection): {'✅ ON' if USE_VAD else '❌ OFF'}")
    print(f"  Segment Profile: {SEGMENT_PROFILE.upper()}")
    print("=" * 60)
    print("\n💡 Tip: Edit sync_suisei.py to change advanced settings")
    print()


def show_system_info():
    """시스템 정보 표시"""
    print("\n" + "=" * 60)
    print("📊 SYSTEM INFORMATION")
    print("=" * 60)

    # GPU 정보
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_props = torch.cuda.get_device_properties(0)
        vram_gb = gpu_props.total_memory / 1024**3
        print(f"  GPU: ✅ {gpu_name}")
        print(f"  VRAM: {vram_gb:.1f} GB")
    else:
        print("  GPU: ❌ CUDA not available")

    # 폴더 정보
    print(f"\n  Songs folder: {SONGS_DIR}/")
    print(f"  Lyrics folder: {LYRICS_DIR}/")
    print(f"  Output folder: {OUTPUT_DIR}/")

    # 파일 개수
    songs_count = len(list(Path(SONGS_DIR).glob('*.mp3'))) if Path(SONGS_DIR).exists() else 0
    lyrics_count = len(list(Path(LYRICS_DIR).glob('*.txt'))) if Path(LYRICS_DIR).exists() else 0

    print(f"\n  MP3 files: {songs_count}")
    print(f"  Lyrics files: {lyrics_count}")

    print("=" * 60)
    print()


def change_language():
    """언어 변경 메뉴"""
    global CURRENT_LANGUAGE

    # Whisper에서 지원하는 주요 언어
    LANGUAGES = {
        '1': ('ja', 'Japanese (日本語)'),
        '2': ('ko', 'Korean (한국어)'),
        '3': ('en', 'English'),
        '4': ('zh', 'Chinese (中文)'),
        '5': ('es', 'Spanish (Español)'),
        '6': ('fr', 'French (Français)'),
        '7': ('de', 'German (Deutsch)'),
        '8': ('it', 'Italian (Italiano)'),
        '9': ('pt', 'Portuguese (Português)'),
        '10': ('ru', 'Russian (Русский)'),
        '11': ('ar', 'Arabic (العربية)'),
        '12': ('hi', 'Hindi (हिन्दी)'),
        '13': ('th', 'Thai (ไทย)'),
        '14': ('vi', 'Vietnamese (Tiếng Việt)'),
        '15': ('id', 'Indonesian (Bahasa Indonesia)'),
    }

    print("\n" + "=" * 60)
    print("🌍 LANGUAGE SELECTION")
    print("=" * 60)
    print(f"Current: {CURRENT_LANGUAGE.upper()}")
    print()

    print("Select target language:")
    print("-" * 60)
    for key, (code, name) in LANGUAGES.items():
        marker = "✅" if code == CURRENT_LANGUAGE else "  "
        print(f"  [{key:>2}] {marker} {name}")
    print("-" * 60)
    print("  [ 0] Cancel")
    print()

    choice = input("Select language number: ").strip()

    if choice == '0':
        print("❌ Cancelled.")
        input("\nPress Enter to continue...")
        return

    if choice in LANGUAGES:
        lang_code, lang_name = LANGUAGES[choice]
        CURRENT_LANGUAGE = lang_code
        # ⭐ 중요: sync_suisei 모듈의 LANGUAGE도 변경 (실제 적용)
        sync_suisei.LANGUAGE = lang_code
        print(f"\n✅ Language changed to: {lang_name}")
        print(f"   Code: {lang_code}")
        print("\n💡 Note: This change applies only to this session.")
        print("   To change the default, edit LANGUAGE in sync_suisei.py")
    else:
        print("❌ Invalid selection.")

    input("\nPress Enter to continue...")


def batch_process():
    """배치 처리 모드"""
    print("\n🚀 Starting Batch Process...\n")

    # 환경 검증
    if not verify_environment():
        input("\nPress Enter to continue...")
        return

    # 파일 검증
    songs = verify_files(SONGS_DIR, LYRICS_DIR)

    if not songs:
        print("❌ No songs to process.")
        print(f"   Add MP3 files to '{SONGS_DIR}/' folder")
        print(f"   Add matching lyrics (.txt) to '{LYRICS_DIR}/' folder")
        input("\nPress Enter to continue...")
        return

    # 확인
    print(f"\n📌 {len(songs)} song(s) ready to process.")
    confirm = input("Continue? [Y/n]: ").strip().lower()

    if confirm and confirm != 'y':
        print("❌ Cancelled.")
        input("\nPress Enter to continue...")
        return

    # 모델 로드
    print("\n🔄 Loading model...")
    print(f"   Model: {MODEL_NAME}")
    print("   (First run will download the model)")
    print()

    try:
        model = stable_whisper.load_model(MODEL_NAME, device='cuda')
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        input("\nPress Enter to continue...")
        return

    print("✅ Model loaded!\n")

    # 배치 처리
    total_start = time.time()
    results = []

    for i, song in enumerate(songs, 1):
        print(f"\n[{i}/{len(songs)}] Processing: {song['name']}")
        print("-" * 60)

        result = process_song(
            model,
            song['mp3'],
            song['lyrics'],
            song['output']
        )

        result['name'] = song['name']
        results.append(result)

    total_time = time.time() - total_start

    # 요약
    print()
    print_summary(results, total_time, save_to_file=True)

    # GPU 메모리 정리
    del model
    torch.cuda.empty_cache()

    input("\nPress Enter to continue...")


def single_process():
    """단일 곡 처리 모드"""
    print("\n🎯 Single Song Process\n")

    # 환경 검증
    if not verify_environment():
        input("\nPress Enter to continue...")
        return

    # MP3 파일 목록
    songs_path = Path(SONGS_DIR)
    if not songs_path.exists():
        print(f"❌ Error: '{SONGS_DIR}/' folder not found!")
        input("\nPress Enter to continue...")
        return

    mp3_files = sorted(songs_path.glob('*.mp3'))

    if not mp3_files:
        print(f"❌ No MP3 files in '{SONGS_DIR}/' folder")
        input("\nPress Enter to continue...")
        return

    # 목록 표시
    print("Available songs:")
    print("-" * 60)
    for i, mp3 in enumerate(mp3_files, 1):
        lyrics_exists = (Path(LYRICS_DIR) / f"{mp3.stem}.txt").exists()
        status = "✅" if lyrics_exists else "❌ (no lyrics)"
        print(f"  [{i}] {mp3.stem} {status}")
    print("-" * 60)

    # 선택
    try:
        choice = input("\nSelect song number (or 0 to cancel): ").strip()
        choice_num = int(choice)

        if choice_num == 0:
            print("❌ Cancelled.")
            input("\nPress Enter to continue...")
            return

        if choice_num < 1 or choice_num > len(mp3_files):
            print("❌ Invalid selection.")
            input("\nPress Enter to continue...")
            return

        selected_mp3 = mp3_files[choice_num - 1]
        song_name = selected_mp3.stem
        lyrics_file = Path(LYRICS_DIR) / f"{song_name}.txt"
        output_file = Path(OUTPUT_DIR) / f"{song_name}.lrc"

        # 가사 파일 확인
        if not lyrics_file.exists():
            print(f"❌ Error: Lyrics file not found: {lyrics_file}")
            input("\nPress Enter to continue...")
            return

        # 처리 시작
        print(f"\n📌 Processing: {song_name}")
        print("-" * 60)

        # 모델 로드
        print("\n🔄 Loading model...")
        try:
            model = stable_whisper.load_model(MODEL_NAME, device='cuda')
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            input("\nPress Enter to continue...")
            return

        print("✅ Model loaded!\n")

        # 처리
        result = process_song(model, selected_mp3, lyrics_file, output_file)

        if result.get('success'):
            print("\n✅ Success!")
            print(f"   Output: {output_file}")
        else:
            print(f"\n❌ Failed: {result.get('error', 'Unknown error')}")

        # GPU 메모리 정리
        del model
        torch.cuda.empty_cache()

        input("\nPress Enter to continue...")

    except ValueError:
        print("❌ Invalid input. Please enter a number.")
        input("\nPress Enter to continue...")
    except KeyboardInterrupt:
        print("\n\n⚠️ Cancelled by user.")
        input("\nPress Enter to continue...")


def ensure_folders():
    """필요한 폴더 자동 생성"""
    folders = [SONGS_DIR, LYRICS_DIR, OUTPUT_DIR]
    created = []

    for folder in folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)
            created.append(folder)

    if created:
        print("📁 Created missing folders:")
        for folder in created:
            print(f"   ✅ {folder}/")
        print()


def main():
    """메인 함수"""
    # 첫 실행 시 폴더 자동 생성
    ensure_folders()

    while True:
        # 화면 클리어 (선택적)
        if os.name == 'nt':  # Windows
            os.system('cls')
        else:  # Unix/Linux/Mac
            os.system('clear')

        print_banner()
        print_menu()

        choice = input("\nSelect option: ").strip()

        if choice == '1':
            batch_process()
        elif choice == '2':
            single_process()
        elif choice == '3':
            change_language()
        elif choice == '4':
            show_settings()
            input("\nPress Enter to continue...")
        elif choice == '5':
            show_system_info()
            input("\nPress Enter to continue...")
        elif choice == '0':
            print("\n👋 Thanks for using LyricFlow!")
            print("   Star us on GitHub: https://github.com/YOUR_USERNAME/LyricFlow\n")
            sys.exit(0)
        else:
            print("\n❌ Invalid option. Please try again.")
            input("\nPress Enter to continue...")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user. Exiting...")
        sys.exit(0)
