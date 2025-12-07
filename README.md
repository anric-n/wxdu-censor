### WXDU Music Autocensorer

## What it Does:
This is a Streamlit web application that automatically censors music. Users can upload audio files of explicit music and retrieve censored audio files, where expicit words have been silenced from the vocals. 
The app uses Demucs to isolate vocals, Whisper to transcribe the vocals with timestamps, ChatGPT to identify explicit words with few-shot in-context learning, and FFmpeg to censor the vocals and recombine them with the instrumental track.

## Quick start

1. **Install Python 3.11, FFmpeg**:
   More information can be found in SETUP.md

2. **Create a virtual environment**:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  
   ```
   On Windows:
   ```
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Set your OpenAI API key**:
   From the `project` directory:
   ```bash
   mkdir .streamlit
   echo "OPENAI_API_KEY = [YOUR API KEY]" >> .streamlit/secrets.toml
   ```

5. **Run the app**:
   From the `project` directory:
   ```bash
   streamlit run src/app.py
   ```
   The app will open in your browser at `http://localhost:8501`

6. **Upload audio file**
   From your system, upload an audio file of the formats mp3, wav, flac, m4a, or ogg. After clicking "Process Audio", the model will run and censor the audio file. This may take several minutes depending on the length and size of the file. 

## Video Links

## Evaluation


TRANSCRIPTION COMPARISON SUMMARY

Successfully processed files: 20

Word Error Rate (WER) Statistics:
  Average WER: 0.551
  Min WER: 0.093
  Max WER: 1.429
  Total correct matches: 3207
  Total substitutions: 1661
  Total deletions: 825
  Total insertions: 494

Timing Accuracy (MSE in seconds²):
  Average Start MSE: 424.694781
  Average End MSE: 419.645329
  Average Start Time Diff: 5.071354s
  Average End Time Diff: 5.001832s
  Avg matched words with timing: 160.3

Average processing time:
  Average Demucs time: 12.292s
  Average Whisper time: 12.983s






