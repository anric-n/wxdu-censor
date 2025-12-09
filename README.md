### WXDU Music Autocensor

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

Project Demo:
https://drive.google.com/file/d/14yVkD3NS4po_yzrLN-9KusGWY1SVxrVV/view?usp=drive_link
Technical Walkthrough:
https://drive.google.com/file/d/1WzvH7WlkEx4Jw92ikJqS_g96UL1RilNl/view?usp=drive_link

## Evaluation

### Quantitative evaluation
Full pipeline evaluation metrics are not available, as there is no opensource dataset of words censored in clean versions of songs.
However, the JamandoLyrics dataset allows us to evaluate the automatic lyric alingment retrived from the transcripts of the songs as modeled by Demucs and Whisper.


| Evaluation Metric (n = 20, JamendoLyrics Dataset)                                      	| English 	| German 	|
|----------------------------------------------------------------------------------------	|---------	|--------	|
| Median Demucs Inference Time (s)                                                       	| 11.807  	| 12.817 	|
| Median Whisper Inference Time (s)                                                      	| 11.846  	| 11.717 	|
| [Word Error Rate](https://en.wikipedia.org/wiki/Word_error_rate)                       	| 0.426   	| 0.330  	|
| Root Mean-Squared Error (RMSE, s)<br>of Word-Start Timestamps <br>(For correct words)  	| 0.609   	| 0.446  	|
| RMSE of Word-End Timestamps (s)                                                        	| 0.568   	| 0.265  	|

The word error rate represents the proportion of words in the transcript that were correctly identified, which means that the Demucs -> Whisper pipeline was able to capture around 57% of the words from the English dataset and 67% of the words from the German dataset. However, this word-error rate varied dramatically between songs. Several songs in the German database had a word error rate less than 0.200, while one song in the English database, `Songwriterz_-_Back_In_Time` actually had a word error rate greater than 1. 



The inference time shown in the table above was gathered by running the pipeline in Google Colab on their T4 GPUS and the 2025.07 runtime version. The songs in that dataset vary in length from around 2:50 to 5:00 minutes long, with most between 3 and 4 minutes.

Performance on non-CUDA systems, such as with my Macbook Pro with an M3 and 8GB of RAM, is much slower. While Demucs is able to utilize the `mps` backend, `faster-whisper` does not support mps. Compared to the GPU, my system takes around 2-3x as long to run Demucs source separation and 5-10x as long to run Whisper transcription. 

More detailed evaluation metrics can be found in this [Google Drive](https://drive.google.com/drive/folders/1FN1GMNgh-ZXHhlEAexvp4p-SD99q8mo-)

### Qualitative evaluation
Qualitiatively, the largest chokepoint within the pipeline is the Whisper model. This is likely because of the use of zero-shot generalization to generate transcripts from the source-separted vocals. When analysing the poorest performing audio files, it was clear that many vocalists do not enunciate their words as clearly as in normal speech. Because no fine-tuning was performed on the model, I believe that difference made it difficult for Whisper to identify phonemes. 

The in-context learning combined with few-shot prompting of ChatGPT worked quite well. Without the specific instructions from the PDX document, ChatGPT often flagged words that should not have been marked as explicit, such as 'suck' or 'kill'. 

The RMSE of the word-timestamps of approximatly 0.50 seconds meant that the silencing intervals of explicit words were often mistimed. For example, the censored audio file might have the beginning or end of an explicit word silenced, but the majority of the word was still audible. Therefore, silence padding is recommended to ensure censorship of identified words. 

### Error cases
The most commmon error case that occurs is when there is no vocals in the audio file. Demucs cannot identify if an audio file has / does not have vocals. In these cirumstances, Demucs produces an vocal file that consists mostly of silence with a bit of noise. That noise causes Whisper to hallucinate, often producing a transcript that consists of the words "Thank you" repeated many times. In this case, the final result is simply the original audio file.

The repeated "Thank you" hallucination / failure case extends to songs with actual vocals that do not start at the beginning of the song. The threshold appears to be around 30-40 seconds for this halucination to occur, it gets progressivly worse the longer the silence is. If enough halucinated 'Thank you's appear, then no actual vocals are transcribed, as Whisper is a transformer model that attends to its own previous transcriptions. 

Another error case occured when a word was repeated many times. In one case, Whisper repeated the same word at the same timestamp several times when it was previously said in succession.

Documentation about these error cases can be found in this [Google Drive folder](https://drive.google.com/drive/folders/118xuvnJWTjxIsT3ILP9f8yuh0qNwXVwg?usp=sharing).




