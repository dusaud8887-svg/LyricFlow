# 🎵 LyricFlow

> AI-Powered Lyrics Synchronization Tool with Intelligent Line Preservation

Generate perfectly timed LRC subtitle files from MP3 audio and lyrics using OpenAI Whisper AI. **Preserve your verse structure** while achieving ±0.2-0.3 second accuracy.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![GPU](https://img.shields.io/badge/GPU-CUDA_Required-green.svg)
![Version](https://img.shields.io/badge/version-2.1-brightgreen.svg)

[한국어 문서](README_KO.md) • [English](README.md)

---

## ✨ Features

- 🎯 **High Accuracy**: ±0.2-0.3 second precision with Whisper large-v3
- ⚡ **GPU Accelerated**: 10x faster processing with CUDA
- ⭐ **Line Preservation**: Maintain your lyric structure (verse-by-verse timestamps)
- 🌍 **Multi-Language**: Supports 15+ languages (Japanese, Korean, English, Chinese, etc.)
- 🔄 **Batch Processing**: Process multiple songs automatically
- 📊 **Quality Validation**: Automatic quality checks and warnings
- 🎤 **Advanced Options**: Demucs vocal separation, VAD, segment optimization

---

## 🌟 What's New in v2.1

### Line Preservation Mode (Core Feature!)
- ⭐ **Preserve line breaks** in your lyrics file → verse-by-verse timestamps
- `PRESERVE_LINES = True` (default): Respects your lyric structure
- `PRESERVE_LINES = False`: Automatic segmentation (character-based)

### Example
**Lyrics file** (lyrics/song.txt):
```
行こう　この声に導かれ
今日もまた一歩ずつ
夢見た場所へ
輝く未来を信じて
```

**Generated LRC** (output/song.lrc):
```lrc
[00:15.23] 行こう　この声に導かれ
[00:18.45] 今日もまた一歩ずつ
[00:22.67] 夢見た場所へ
[00:26.89] 輝く未来を信じて
```

**Each line becomes a separate timestamp!** 🎵

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM (CUDA required)
- 5GB+ disk space (for model download)

### Installation with uv (Recommended)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/YOUR_USERNAME/LyricFlow.git
cd LyricFlow

# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

### Traditional Installation

```bash
# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install LyricFlow
pip install -r requirements.txt
```

---

## 📖 Usage

### Interactive CLI (Recommended)

```bash
python lyricflow.py
```

**Features:**
- 🚀 Batch processing (all songs)
- 🎯 Single song processing
- 🌍 Language selection (15+ languages)
- ⚙️  Settings view
- 📊 System information

### Batch Processing (Original)

```bash
python sync_suisei.py
```

### Folder Structure

```
LyricFlow/
├── songs/          # Put MP3 files here
│   └── song.mp3
├── lyrics/         # Put lyrics (UTF-8 .txt) here
│   └── song.txt
└── output/         # Generated LRC files
    └── song.lrc
```

---

## ⚙️ Configuration

Edit `sync_suisei.py` to customize:

```python
# Line Preservation (v2.1 Core Feature!)
PRESERVE_LINES = True  # Preserve your verse structure (recommended!)

# Model Selection
MODEL_NAME = 'large-v3'  # Highest quality (±0.2s)

# Language
LANGUAGE = 'ja'  # Japanese (change in CLI or here)

# Advanced Options
USE_DEMUCS = False  # Vocal separation (3x slower, 60% WER reduction)
USE_VAD = True      # Voice Activity Detection (hallucination prevention)
SEGMENT_PROFILE = 'normal'  # 'ballad', 'normal', 'fast'
```

---

## 🌍 Supported Languages

**15+ languages via Whisper:**
- 🇯🇵 Japanese (日本語)
- 🇰🇷 Korean (한국어)
- 🇬🇧 English
- 🇨🇳 Chinese (中文)
- 🇪🇸 Spanish (Español)
- 🇫🇷 French (Français)
- 🇩🇪 German (Deutsch)
- 🇮🇹 Italian (Italiano)
- And more...

**Change language:**
1. Interactive CLI: Option [3]
2. Edit `LANGUAGE` in `sync_suisei.py`

---

## 🔧 Building Standalone Executable

```bash
# Build with PyInstaller
python build.py

# Output: dist/LyricFlow or dist/LyricFlow.exe
```

**Distribution:**
- Copy executable to destination
- Ensure CUDA and GPU drivers installed
- Create `songs/` and `lyrics/` folders

---

## 📊 Performance

| Song Length | Processing Time | Accuracy |
|-------------|-----------------|----------|
| 3 minutes   | 10-15 seconds   | ±0.2s    |
| 4 minutes   | 15-20 seconds   | ±0.2s    |
| 5 minutes   | 20-25 seconds   | ±0.3s    |

**Hardware:** RTX 3070 Ti, CUDA 12.4

---

## 🛠️ Troubleshooting

### CUDA Not Detected

```bash
# Reinstall PyTorch with CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Encoding Errors

Ensure lyrics files are **UTF-8 encoded**:

```bash
# Check encoding
file -i lyrics/song.txt

# Convert to UTF-8 (if needed)
iconv -f EUC-KR -t UTF-8 old.txt > lyrics/new.txt
```

---

## 📚 Documentation

- [Korean Documentation](README_KO.md) - Full Korean guide
- [Advanced Configuration](docs/ADVANCED.md) - Coming soon
- [API Reference](docs/API.md) - Coming soon

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

## 🌟 Credits

- **Whisper AI**: OpenAI (speech recognition)
- **stable-ts**: jianfch (Whisper stability improvements)
- **Demucs**: Meta AI (vocal separation)

---

## 💬 Support

- 🐛 [Report Issues](https://github.com/YOUR_USERNAME/LyricFlow/issues)
- 💡 [Feature Requests](https://github.com/YOUR_USERNAME/LyricFlow/discussions)
- ⭐ Star us on GitHub!

---

<p align="center">
  <strong>Let your lyrics flow with perfect timing</strong> 🎵
</p>
