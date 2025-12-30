# 🚀 Quick Start Guide - LyricFlow

**Get started in 5 minutes!**

---

## 📋 Prerequisites

Before you begin, ensure you have:
- ✅ Python 3.10 or higher
- ✅ NVIDIA GPU with 8GB+ VRAM
- ✅ CUDA installed (check with `nvidia-smi`)
- ✅ At least 5GB free disk space

---

## ⚡ Installation

### Option 1: Using uv (Recommended)

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone repository
git clone https://github.com/YOUR_USERNAME/LyricFlow.git
cd LyricFlow

# 3. Install dependencies
uv sync

# 4. Activate environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

### Option 2: Traditional pip

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/LyricFlow.git
cd LyricFlow

# 2. Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 3. Install other dependencies
pip install -r requirements.txt
```

---

## 🎯 First Run

### Step 1: Prepare Your Files

LyricFlow will automatically create these folders on first run:
```
LyricFlow/
├── songs/      # Put your MP3 files here
├── lyrics/     # Put your lyrics (.txt, UTF-8) here
└── output/     # Generated LRC files will appear here
```

**Important:**
- MP3 and lyrics files must have **matching names**
- Example: `song.mp3` + `song.txt` → `song.lrc`

### Step 2: Add Your Files

```bash
# Example
songs/stellar_stellar.mp3
lyrics/stellar_stellar.txt
```

**Lyrics file format** (UTF-8 text):
```
行こう　この声に導かれ
今日もまた一歩ずつ
夢見た場所へ
輝く未来を信じて
```

### Step 3: Run LyricFlow

```bash
python lyricflow.py
```

You'll see an interactive menu:
```
╔════════════════════════════════════════════════════════════╗
║                     LyricFlow v2.1                         ║
╚════════════════════════════════════════════════════════════╝

📋 MENU
  [1] 🚀 Batch Process (All songs in folder)
  [2] 🎯 Single Song Process
  [3] 🌍 Change Language
  [4] ⚙️  View Current Settings
  [5] 📊 System Information
  [0] 🚪 Exit

Select option:
```

### Step 4: Process Your Lyrics

- **Option 1**: Select `[1]` for batch processing (all songs)
- **Option 2**: Select `[2]` to process a single song

The first run will download the Whisper model (~3GB, one-time download).

---

## 🌍 Language Support

### Changing Language

1. Run `python lyricflow.py`
2. Select `[3] Change Language`
3. Choose from 15+ supported languages:
   - 🇯🇵 Japanese (日本語) - Best tested
   - 🇰🇷 Korean (한국어)
   - 🇬🇧 English
   - 🇨🇳 Chinese (中文)
   - And more...

### Default Language

To change the default language permanently, edit `sync_suisei.py`:
```python
LANGUAGE = 'ja'  # Change to your language code
```

---

## ⚙️ Configuration

### Line Preservation Mode (v2.1 Key Feature!)

**Enabled by default** - Respects your lyric structure!

Each line in your lyrics file = one LRC subtitle line.

To change:
```python
# In sync_suisei.py
PRESERVE_LINES = True   # Keep verse structure (recommended)
PRESERVE_LINES = False  # Auto-split by character count
```

### Model Selection

```python
# In sync_suisei.py
MODEL_NAME = 'large-v3'          # Best quality (±0.2s, slower)
# MODEL_NAME = 'large-v3-turbo'  # 6x faster, good quality (±0.3s)
```

---

## 🛠️ Troubleshooting

### Problem: CUDA not available

**Solution:**
```bash
# Reinstall PyTorch with CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Verify
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

### Problem: Encoding error

**Solution:** Ensure lyrics files are UTF-8 encoded
```bash
# Check encoding
file -i lyrics/song.txt

# Convert if needed (Linux/Mac)
iconv -f YOUR_ENCODING -t UTF-8 old.txt > lyrics/new.txt
```

### Problem: Model download fails

**Solution:**
- Check internet connection
- Ensure 5GB+ free space
- Model downloads to `~/.cache/whisper/`

### Problem: Out of memory

**Solution:**
- Your GPU needs 8GB+ VRAM for large-v3
- Try `large-v3-turbo` or `medium` model instead

---

## 📊 Expected Performance

| Song Length | Processing Time | Accuracy |
|-------------|-----------------|----------|
| 3 minutes   | 10-15 seconds   | ±0.2s    |
| 4 minutes   | 15-20 seconds   | ±0.2s    |
| 5 minutes   | 20-25 seconds   | ±0.3s    |

*Tested on RTX 3070 Ti*

---

## 📖 Next Steps

- Read the [full README](README.md) for advanced features
- Check [Korean docs](README_KO.md) for detailed Korean guide
- Report issues on [GitHub](https://github.com/YOUR_USERNAME/LyricFlow/issues)

---

## 💡 Tips

1. **First run**: Expect 2-3 minutes for model download
2. **Batch processing**: Process all songs at once for efficiency
3. **Quality check**: Always preview LRC files in your music player
4. **Line breaks matter**: Each line in lyrics = one subtitle line
5. **UTF-8 is required**: Non-UTF-8 files will cause errors

---

**That's it! You're ready to sync lyrics!** 🎵

Need help? [Open an issue](https://github.com/YOUR_USERNAME/LyricFlow/issues)
