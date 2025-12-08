## Libraries:

**Demucs Music Source Separation**:
[Repository](https://github.com/adefossez/demucs)
[Documentation](https://github.com/adefossez/demucs/blob/main/docs/api.md)

**ChatGPT 5 / 5.1**
[Documentation](https://platform.openai.com/docs/guides/structured-outputs)

**Faster Whisper**
[Original Model](https://github.com/openai/whisper)
[faster-whisper repository](https://github.com/SYSTRAN/faster-whisper/tree/master)

**FFmpeg**
[Documentation](https://ffmpeg.org/ffmpeg-filters.html#volume)

**Streamlit**
[Repository](https://github.com/streamlit/streamlit)
[Documentation](https://docs.streamlit.io/)

## Data:
The dataset used to evaluate Whisper / Demucs automatic lyrics alignment comes from the JamendoLyrics MultiLang dataset:

The dataset was introduced in the ICASSP 2023 paper (full citation below):
Similarity-based Audio-Lyrics Alignment of Multiple Languages
Simon Durand, Daniel Stoller, Sebastian Ewert (Spotify)
[Dataset](https://huggingface.co/datasets/jamendolyrics/jamendolyrics)

@inproceedings{durand-2023-contrastive,
  author={Durand, Simon and Stoller, Daniel and Ewert, Sebastian},
  booktitle={2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Contrastive Learning-Based Audio to Lyrics Alignment for Multiple Languages}, 
  year={2023},
  pages={1-5},
  address={Rhodes Island, Greece},
  doi={10.1109/ICASSP49357.2023.10096725}
}

## Other sources:
FCC guidelines for indecent / profane content were used for prompting instructions:
[Source](https://www.fcc.gov/sites/default/files/obscene_indecent_and_profane_broadcasts.pdf)

Guide from Public Radio Exchange (PRX) added to prompt for further guidance:
[Source](https://help.prx.org/hc/en-us/articles/360044988133-A-guide-to-broadcast-obscenities-and-issuing-content-advisories)

## AI-Generation Usage
Among other uses, an AI agent was used to 
