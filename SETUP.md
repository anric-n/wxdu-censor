# Setup Guide

## 1. Prerequisites

### i. Python 3.11:

The application requires **Python 3.11** because:

- `demucs` depends on `torchaudio<2.2.0`
- The latest version of Python that supports `torchaudio<2.2.0` is 3.11

Verify your Python version:

```bash
python3 --version
```

If needed, install Python 3.11:

**macOS / Linux (Homebrew):**
```bash
brew install python@3.11
```

**Windows:**  
  Download from: https://www.python.org/downloads/release/python-3110/


### ii. FFmpeg:

FFmpeg is required for audio processing and must be available on your system PATH.

Check if it’s installed:

```bash
ffmpeg -version
```

If not installed:

**macOS / Linux (Homebrew):**
```bash
brew install ffmpeg
```

**Windows:**  
Download a build from: https://ffmpeg.org/download.html  
Extract it and add the `bin/` folder to your PATH.


### iii. OpenAI API Key:

An OpenAI key is required for ChatGPT integration.

Create one here (if you don’t have one already):  
https://platform.openai.com/api-keys


## 2. Creating the Virtual Environment

From the project root, create and activate a Python 3.11 virtual environment:
**macOS / Linux:**
```bash
python3.11 -m venv venv
source venv/bin/activate        
```

**Windows:** 
```
venv\Scripts\activate
```

## 3. Installing Dependencies

Upgrade pip and install all required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Adding Your OpenAI API Key

The app reads your API key from Streamlit secrets.

From the project directory:

```bash
mkdir -p .streamlit
echo "OPENAI_API_KEY = '[YOUR API KEY]'" >> .streamlit/secrets.toml
```

Alternatively, manually create this file:

**.streamlit/secrets.toml**
```toml
OPENAI_API_KEY = "your_key_here"
```


## 5. Running the Application

Launch the Streamlit app:

    ```bash
    streamlit run src/app.py
    ```

The app will open automatically in your browser at:

    ```
    http://localhost:8501
    ```


# Evaluation Data Replication
The typical `load_dataset` method used to load Hugging-Face datasets does not work for this project, as there is a dependency conlict between `torchcodec` and the legacy version of `torchaudio` needed to run Demucs. Instead, the dataset, including the audio files must be cloned from the Github repository through a submodule. 

First, install Git LFS if you haven't already:
```bash
git lfs install
```

Then clone the repository with submodules:
```bash
git clone --recurse-submodules https://github.com/anric-n/wxdu-censor.git
cd wxdu-censor
```

Or if you already cloned without submodules, initialize and update:
```bash
cd data
git submodule update --init
```

The dataset will be cloned to `data/jamendolyrics/` from https://huggingface.co/datasets/jamendolyrics/jamendolyrics

Right now running `src/eval.py` will output the results for the first english song alphabetically for `en_transcription_comparison.csv` to `data/eval/transcription_comparison.csv`. Additonally, CSV files for each song comparing their transcriptions at the word level will populate in `data/eval/normalize_transcripts`.

To get results for all 20 songs in English, modify line 26 in `src/eval.py` to
```python
for audio_file in sorted(project_root.glob("data/jamendolyrics/subsets/en/mp3/*.mp3")):

```
Similarly, to get results for German, modify line 26 in `src/eval.py` to

```python
for audio_file in sorted(project_root.glob("data/jamendolyrics/subsets/de/mp3/*.mp3")):

```
