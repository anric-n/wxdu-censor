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
Full pipeline evaluation metrics are not available, as there is no opensource dataset of words censored in clean versions of songs.
However, the JamandoLyrics dataset allows us to evaluate the automatic lyric alingment retrived from the transcripts of the songs as modeled by Demucs and Whisper. 

| Evaluation Metric (n = 20, JamendoLyrics Dataset)                                      	| English 	| German 	|
|----------------------------------------------------------------------------------------	|---------	|--------	|
| Median Demucs Inference Time (s)                                                       	| 11.807  	| 12.817 	|
| Median Whisper Inference Time (s)                                                      	| 11.846  	| 11.717 	|
| [Word Error Rate](https://en.wikipedia.org/wiki/Word_error_rate)                       	| 0.426   	| 0.330  	|
| Root Mean-Squared Error (RMSE, s)<br>of Word-Start Timestamps <br>(For correct words)  	| 0.609   	| 0.446  	|
| RMSE of Word-End Timestamps (s)                                                        	| 0.568   	| 0.265  	|

More detailed evaluation metrics can be found in this [Google Drive](https://drive.google.com/drive/folders/1FN1GMNgh-ZXHhlEAexvp4p-SD99q8mo-)


