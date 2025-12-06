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
    streamlit run app.py
    ```

The app will open automatically in your browser at:

    ```
    http://localhost:8501
    ```



