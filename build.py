#!/usr/bin/env python3
"""
LineSync Build Script
Creates standalone executables for Windows, Linux, and macOS
"""

import os
import sys
import shutil
import platform
from pathlib import Path

print("=" * 60)
print("LyricFlow - Build Script")
print("=" * 60)
print()

# 플랫폼 확인
current_platform = platform.system()
print(f"Platform: {current_platform}")
print()

# PyInstaller 확인
try:
    import PyInstaller
    print(f"✅ PyInstaller {PyInstaller.__version__} found")
except ImportError:
    print("❌ PyInstaller not found!")
    print("   Install: pip install pyinstaller")
    sys.exit(1)

print()

# 빌드 옵션
APP_NAME = "LyricFlow"
SCRIPT_NAME = "lyricflow.py"
ICON_FILE = None  # 나중에 아이콘 추가 가능

# 빌드 명령어 구성
build_cmd = [
    "pyinstaller",
    "--onefile",                    # 단일 실행 파일
    "--name", APP_NAME,             # 실행 파일 이름
    "--clean",                      # 빌드 전 캐시 정리
    "--noconfirm",                  # 덮어쓰기 확인 안 함
]

# 콘솔 윈도우 유지 (CLI 도구이므로)
build_cmd.append("--console")

# 추가 모듈 포함 (중요!)
build_cmd.extend(["--hidden-import", "sync_suisei"])
build_cmd.extend(["--hidden-import", "stable_whisper"])
build_cmd.extend(["--hidden-import", "torch"])
build_cmd.extend(["--hidden-import", "tqdm"])

# 아이콘 추가 (있을 경우)
if ICON_FILE and Path(ICON_FILE).exists():
    build_cmd.extend(["--icon", ICON_FILE])

# 스크립트 파일
build_cmd.append(SCRIPT_NAME)

# 빌드 시작
print("🔨 Building executable...")
print(f"   Command: {' '.join(build_cmd)}")
print()

import subprocess
result = subprocess.run(build_cmd)

if result.returncode != 0:
    print("\n❌ Build failed!")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ Build successful!")
print("=" * 60)

# 결과 확인
dist_dir = Path("dist")
if current_platform == "Windows":
    exe_file = dist_dir / f"{APP_NAME}.exe"
else:
    exe_file = dist_dir / APP_NAME

if exe_file.exists():
    file_size = exe_file.stat().st_size / (1024 * 1024)  # MB
    print(f"\nExecutable: {exe_file}")
    print(f"Size: {file_size:.1f} MB")
    print()

    if current_platform != "Windows":
        print("💡 Making executable...")
        os.chmod(exe_file, 0o755)
        print("   chmod +x applied")
        print()

    print("🚀 Usage:")
    if current_platform == "Windows":
        print(f"   {exe_file}")
    else:
        print(f"   ./{exe_file}")
    print()

    print("📦 Distribution:")
    print(f"   1. Copy '{exe_file}' to your destination")
    print("   2. Ensure CUDA and GPU drivers are installed on target machine")
    print("   3. Create songs/ and lyrics/ folders in the same directory")
    print()
else:
    print(f"\n⚠️ Warning: Executable not found at {exe_file}")

print("=" * 60)
