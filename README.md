### Music Autocensor Web App

## What it Does:
A Streamlit web application that automatically censors music. Users can upload audio files of explicit music and retrieve censored audio files, where expicit words have been silenced from the vocals. 
The app uses Demucs to isolate vocals, Whisper to transcribe the vocals with timestamps, ChatGPT to identify explicit words with few-shot in-context learning, and FFmpeg to censor the vocals and recombine them with the instrumental track.

## Quick start
# Prerequisites
1. **Python 3.11** (required for torch 2.1.0 compatibility with demucs)
2. **FFmpeg** installed and available in PATH
3. **OpenAI API Key** for ChatGPT integration

Python and FFmpeg can be installed with Homebrew on Mac / Linux. 

# Instalation

1. **Create virtual environment with Python 3.11**:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   
   Note: This will install torch 2.1.0 and torchaudio 2.1.0, which are compatible with demucs' requirements.

3. **Set your OpenAI API key**:
   From the `project` directory:
   ```bash
   mkdir .streamlit
   echo "OPENAI_API_KEY = [YOUR API KEY]" >> .streamlit/secrets.toml
   ```

4. **Verify FFmpeg is installed**:
```bash
ffmpeg -version
```

# Running the App
   From the `project` directory:
   ```bash
   streamlit run app.py
   ```

The app will open in your browser at `http://localhost:8501`

## Video Links

## Evaluation







