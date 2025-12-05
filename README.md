# Music Autocensor Web App

A Streamlit web application that automatically censors music using:
- **Demucs**: Isolates vocals from music tracks
- **Whisper**: Transcribes vocals with word-level timestamps
- **ChatGPT**: Identifies words to censor using few-shot in-context learning
- **FFmpeg**: Silences vocals at censored word timestamps and recombines with instrumental

## Setup

### Prerequisites

1. **Python 3.11** (required for torch 2.1.0 compatibility with demucs)
2. **FFmpeg** installed and available in PATH
3. **OpenAI API Key** for ChatGPT integration

### Installation

1. **Install Python 3.11** (if not already installed):
   ```bash
   # macOS with Homebrew:
   brew install python@3.11
   
   # Or use pyenv:
   pyenv install 3.11.9
   pyenv local 3.11.9
   ```

2. **Create virtual environment with Python 3.11**:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   
   Note: This will install torch 2.1.0 and torchaudio 2.1.0, which are compatible with demucs' requirements.

2. Set your OpenAI API key:
```bash
mkdir .streamlit
echo "OPENAI_API_KEY = [YOUR API KEY]" >> .streamlit/secrets.toml
```

3. Verify FFmpeg is installed:
```bash
ffmpeg -version
```

### Running the App

From the `project` directory:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Upload Audio**: Click "Upload Audio File" and select a music file (mp3, wav, flac, etc.)

2. **Configure Settings**
   - Select Demucs model
   - Select Whisper model 
   - Select ChatGPT model
   - Adjust silence padding if needed

3. **Provide Few-shot Examples** (optional):
   - Go to the "Few-shot Examples" tab
   - Enter examples showing how to identify words to censor
   - Click "Save Examples"

4. **Process Audio**:
   - Click "🚀 Process Audio"
   - Wait for processing to complete (this may take several minutes)
   - Download the censored audio file

## File Structure
```
project/
├── app.py                    # Main Streamlit application
├── processors/
│   ├── demucs_processor.py   # Vocal isolation
│   ├── whisper_processor.py  # Transcription
│   ├── chatgpt_censor.py     # Censoring logic
│   └── ffmpeg_processor.py   # Audio manipulation
└── requirements.txt          # Python dependencies
```

