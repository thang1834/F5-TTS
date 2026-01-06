---
title: F5 TTS Vietnamese 100h Demo
emoji: 🎙️
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 5.36.2
app_file: app.py
pinned: false
---

# 🎤 F5-TTS Vietnamese

High-quality Vietnamese Text-to-Speech system based on the F5-TTS architecture. 

## 🚀 Quick Start

1. **Setup Environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run Application**:
   ```bash
   python app.py
   ```

## 📖 Documentation

For detailed installation instructions, architecture overview, and Vietnamese-specific optimizations, please refer to the:

👉 **[DOCUMENTATION.md](DOCUMENTATION.md)**

## ✨ Highlights
- **1000h Training Data**: Specifically optimized for northern and southern Vietnamese accents.
- **Windows Support**: Custom patches for audio loading and text normalization on Windows systems.
- **GPU Ready**: Pre-configured for CUDA acceleration.