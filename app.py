"""Streamlit web app for automatic music censoring."""

import streamlit as st
import sys
from pathlib import Path
import tempfile
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
# sys.path.insert(0, str(project_root.parent / "demucs"))

from processors.demucs_processor import isolate_vocals
from processors.whisper_processor import transcribe_vocals
from processors.chatgpt_censor import censor_with_chatgpt
from processors.ffmpeg_processor import process_censored_audio



# Page configuration
st.set_page_config(
    page_title="Music Autocensor",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Music Autocensor")
st.markdown(
    "Automatically censor music using Demucs, Whisper, ChatGPT, and FFmpeg. "
    "Upload a music file and let AI identify and silence inappropriate words."
)

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Model selection
    demucs_model = st.selectbox(
        "Demucs Model",
        ["htdemucs", "htdemucs_ft", "mdx_extra", "mdx"],
        index=2,
        help="Model for vocal separation. htdemucs is recommended."
    )
    
    whisper_model = st.selectbox(
        "Whisper Model",
        ["turbo", "large"],
        index=0,
        help="Larger models are more accurate but slower."
    )
    
    chatgpt_model = st.selectbox(
        "ChatGPT Model",
        ["gpt-5", "gpt-5.1"],
        index=1,
        help="Model for identifying words to censor."
    )
    
    silence_padding = st.slider(
        "Silence Padding (seconds)",
        min_value=0.0,
        max_value=0.5,
        value=0.1,
        step=0.05,
        help="Additional silence before and after censored words."
    )
    
    # API key check
    api_key_status = "✅ Set" if st.secrets["OPENAI_API_KEY"] else "❌ Not set"
    st.markdown(f"**OpenAI API Key:** {api_key_status}")
    if not st.secrets["OPENAI_API_KEY"]:
        st.warning(
            "Please set the OPENAI_API_KEY environment variable. "
            "You can do this by adding 'OPENAI_API_KEY' to the .streamlit/secrets.toml file."
        )

# Main content
tab1, tab2 = st.tabs(["Process Audio", "Few-shot Examples"])

with tab1:
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Audio File",
        type=["mp3", "wav", "flac", "m4a", "ogg"],
        help="Upload a music file to process"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.info(f"📁 File: {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.2f} MB)")
        
        # Process button
        if st.button("🚀 Process Audio", type="primary", use_container_width=True):
            # Check API key
            if not st.secrets["OPENAI_API_KEY"]:
                st.error("❌ OPENAI_API_KEY environment variable is not set!")
                st.stop()
            
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory(prefix="autocensor_") as temp_dir:
                temp_path = Path(temp_dir)
                
                # Save uploaded file
                input_audio_path = temp_path / uploaded_file.name
                with open(input_audio_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Isolate vocals with Demucs
                    status_text.text("🎤 Step 1/5: Isolating vocals with Demucs...")
                    progress_bar.progress(0.1)
                    
                    output_dir = temp_path / "separated"
                    vocals_path, instrumental_path, separated_stems = isolate_vocals(
                        input_audio_path,
                        output_dir,
                        model=demucs_model
                    )
                    
                    st.success("✅ Vocals isolated successfully!")
                    progress_bar.progress(0.3)

                    # Provide download
                    with open(vocals_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Isolated Vocals",
                            data=f.read(),
                            file_name=f"vocals_{uploaded_file.name}",
                            mime="audio/wav",
                            use_container_width=True
                        )
                    
                    # Display audio player
                    st.audio(str(vocals_path))
                    
                    # Step 2: Transcribe vocals with Whisper
                    status_text.text("📝 Step 2/5: Transcribing vocals with Whisper...")
                    
                    transcription = transcribe_vocals(
                        vocals_path,
                        model_size=whisper_model
                    )
                    
                    st.success("✅ Transcription complete!")
                    progress_bar.progress(0.5)
                    
                    # Display transcription
                    with st.expander("📄 View Transcription", expanded=True):
                        st.text(transcription["text"])
                        st.caption(f"Language: {transcription.get('language', 'unknown')}")
                        st.caption(f"Words: {len(transcription['words'])}")
                    
                    # Step 3: Get few-shot examples
                    status_text.text("🤖 Step 3/5: Analyzing with ChatGPT...")
                    progress_bar.progress(0.6)
                    
                    few_shot_examples = st.session_state.get("few_shot_examples", "")
                    
                    # Step 4: Identify words to censor with ChatGPT
                    censored_words = censor_with_chatgpt(
                        transcription["words"],
                        few_shot_examples=few_shot_examples if few_shot_examples else None,
                        model=chatgpt_model
                    )
                    
                    progress_bar.progress(0.8)
                    
                    if censored_words:
                        st.warning(f"⚠️ Found {len(censored_words)} word(s) to censor:")
                        censored_display = []
                        for word in censored_words:
                            censored_display.append(
                                f"'{word['word']}' ({word['start']:.2f}s - {word['end']:.2f}s)"
                            )
                        st.text(", ".join(censored_display))
                    else:
                        st.info("ℹ️ No words identified for censoring.")
                    
                    # Step 5: Process audio with FFmpeg
                    status_text.text("🔇 Step 4/5: Silencing vocals and recombining...")
                    
                    output_audio_path = temp_path / "censored_output.wav"
                    process_censored_audio(
                        vocals_path,
                        instrumental_path,
                        censored_words,
                        output_audio_path,
                        padding=silence_padding
                    )
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Processing complete!")
                    
                    st.success("🎉 Audio processing complete!")
                    
                    # Provide download
                    with open(output_audio_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Censored Audio",
                            data=f.read(),
                            file_name=f"censored_{uploaded_file.name}",
                            mime="audio/wav",
                            use_container_width=True
                        )
                    
                    # Display audio player
                    st.audio(str(output_audio_path))
                    
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    st.exception(e)
                    progress_bar.progress(0.0)
                    status_text.text("")

with tab2:
    st.header("Few-shot Examples")
    st.markdown(
        "Provide few-shot examples to help ChatGPT identify words to censor. "
        "These examples should show the expected input/output format."
    )
    
    default_examples = """Example 1:
Input transcript:
[0.5s-0.8s] hello
[1.0s-1.3s] world
[1.5s-1.8s] damn
[2.0s-2.3s] this
[2.5s-2.8s] is
[3.0s-3.3s] great

Output JSON:
[
  {"word": "damn", "start": 1.5, "end": 1.8}
]

Example 2:
Input transcript:
[0.2s-0.5s] what
[0.6s-0.9s] the
[1.0s-1.4s] hell
[1.5s-1.8s] is
[2.0s-2.3s] this

Output JSON:
[
  {"word": "hell", "start": 1.0, "end": 1.4}
]"""
    
    few_shot_examples = st.text_area(
        "Few-shot Examples",
        value=st.session_state.get("few_shot_examples", default_examples),
        height=400,
        help="Enter examples showing how to identify words to censor. "
             "Include both input transcripts and expected JSON output."
    )
    
    if st.button("💾 Save Examples", use_container_width=True):
        st.session_state["few_shot_examples"] = few_shot_examples
        st.success("✅ Examples saved! They will be used in the next processing run.")
    
    st.markdown("### Tips:")
    st.markdown(
        "- Include examples that show both what to censor and what NOT to censor\n"
        "- Be specific about the format (timestamps, word matching)\n"
        "- The examples will be included in the ChatGPT prompt to guide its behavior"
    )

