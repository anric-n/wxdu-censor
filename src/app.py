"""Streamlit web app for automatic music censoring."""

import streamlit as st
import sys
from pathlib import Path
import logging
import json
import shutil
import tempfile
import zipfile
import io
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
# Add project root to path
project_root = Path(__file__).resolve().parents[1] 
sys.path.insert(0, str(project_root))

from models.demucs_processor import isolate_vocals
from models.whisper_processor import transcribe_vocals
from models.chatgpt_censor import censor_with_chatgpt
from models.ffmpeg_processor import process_censored_audio


def initialize_output_tracking():
    """Initialize the output tracking dictionary in session state."""
    if "output_files" not in st.session_state:
        st.session_state.output_files = {
            "Original Audio": [],
            "Vocals": [],
            "Instrumental": [],
            "Transcription": [],
            "Censored Words": [],
            "Censored Audio": []
        }
    if "output_dir" not in st.session_state:
        # Create a persistent directory for outputs
        temp_output_dir = tempfile.mkdtemp(prefix="autocensor_outputs_")
        st.session_state.output_dir = Path(temp_output_dir)
        st.session_state.output_dir.mkdir(parents=True, exist_ok=True)


def clear_output_tracking():
    """Clear the output tracking and remove old output directory."""
    if "output_dir" in st.session_state and st.session_state.output_dir.exists():
        try:
            shutil.rmtree(st.session_state.output_dir)
        except Exception:
            pass
    st.session_state.output_files = {
        "Original Audio": [],
        "Vocals": [],
        "Instrumental": [],
        "Transcription": [],
        "Censored Words": [],
        "Censored Audio": []
    }


def get_file_metadata(file_path: str, output_type: str, file_idx: int):
    """Get mime type and download name for a file."""
    if not file_path or not Path(file_path).exists():
        return None, None, None
    
    path_obj = Path(file_path)
    if output_type == "Transcription":
        mime_type = "text/plain"
        download_name = f"file_{file_idx+1}_transcription.txt"
    elif output_type == "Censored Words":
        mime_type = "application/json"
        download_name = f"file_{file_idx+1}_censored_words.json"
    elif output_type in ["Vocals", "Instrumental", "Censored Audio"]:
        mime_type = "audio/wav"
        download_name = f"file_{file_idx+1}_{output_type.lower().replace(' ', '_')}{path_obj.suffix}"
    else:  # Original Audio
        ext = path_obj.suffix
        mime_map = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".m4a": "audio/mp4",
            ".ogg": "audio/ogg"
        }
        mime_type = mime_map.get(ext, "application/octet-stream")
        download_name = f"file_{file_idx+1}_original{ext}"
    
    return mime_type, download_name, file_path


def display_download_table(output_files: Dict[str, List[str]]):
    """Display the download table with all outputs."""
    if not any(output_files.get("Original Audio", [])):
        return
    
    st.markdown("---")
    st.markdown("### 📥 Download Outputs")
    
    num_files = len(output_files["Original Audio"])
    
    # Create header
    col_names = ["File"] + list(output_files.keys())
    cols = st.columns(len(col_names))
    for idx, col_name in enumerate(col_names):
        with cols[idx]:
            st.markdown(f"**{col_name}**")
    
    # Create rows
    for file_idx in range(num_files):
        cols = st.columns(len(col_names))
        
        with cols[0]:
            # Get original filename for display
            orig_path = output_files["Original Audio"][file_idx]
            if orig_path:
                filename = Path(orig_path).name
                st.write(f"File {file_idx + 1}: {filename}")
            else:
                st.write(f"File {file_idx + 1}")
        
        for col_idx, output_type in enumerate(output_files.keys(), start=1):
            with cols[col_idx]:
                if file_idx < len(output_files[output_type]):
                    file_path = output_files[output_type][file_idx]
                    mime_type, download_name, path = get_file_metadata(file_path, output_type, file_idx)
                    
                    if path and Path(path).exists():
                        with open(path, "rb") as f:
                            file_data = f.read()
                        
                        st.download_button(
                            label="⬇️ Download",
                            data=file_data,
                            file_name=download_name,
                            mime=mime_type,
                            key=f"dl_{output_type}_{file_idx}",
                            use_container_width=True
                        )
                    else:
                        st.write("—")  # No file available (e.g., no censored words)
                else:
                    st.write("—")
    
    # Add download all as zip button
    st.markdown("---")
    st.markdown("### Download All Outputs")
    zip_buffer = create_zip_from_outputs(output_files)
    st.download_button(
        label="📦 Download All Outputs as ZIP",
        data=zip_buffer,
        file_name="all_outputs.zip",
        mime="application/zip",
        use_container_width=True,
        key="download_all_zip"
    )


def create_zip_from_outputs(output_files: Dict[str, List[str]]) -> io.BytesIO:
    """Create a zip file containing all output files."""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Get the number of files processed
        num_files = len(output_files["Original Audio"])
        
        for file_idx in range(num_files):
            file_base_name = Path(output_files["Original Audio"][file_idx]).stem if output_files["Original Audio"][file_idx] else f"file_{file_idx}"
            
            # Add each output file if it exists
            for output_type, paths in output_files.items():
                if file_idx < len(paths) and paths[file_idx]:
                    file_path = Path(paths[file_idx])
                    if file_path.exists():
                        # Create a clean filename
                        if output_type == "Original Audio":
                            zip_name = f"{file_idx+1}_{file_base_name}_original{file_path.suffix}"
                        elif output_type == "Vocals":
                            zip_name = f"{file_idx+1}_{file_base_name}_vocals{file_path.suffix}"
                        elif output_type == "Instrumental":
                            zip_name = f"{file_idx+1}_{file_base_name}_instrumental{file_path.suffix}"
                        elif output_type == "Transcription":
                            zip_name = f"{file_idx+1}_{file_base_name}_transcription.txt"
                        elif output_type == "Censored Words":
                            zip_name = f"{file_idx+1}_{file_base_name}_censored_words.json"
                        elif output_type == "Censored Audio":
                            zip_name = f"{file_idx+1}_{file_base_name}_censored{file_path.suffix}"
                        else:
                            zip_name = f"{file_idx+1}_{file_base_name}_{output_type.lower().replace(' ', '_')}{file_path.suffix}"
                        
                        zip_file.write(file_path, zip_name)
    
    zip_buffer.seek(0)
    return zip_buffer

def main():

    # Initialize output tracking
    initialize_output_tracking()

    # Page configuration
    st.set_page_config(
        page_title="Music Autocensor",
        page_icon="🎵",
        layout="wide"
    )

    st.title("🎵 Music Autocensor")
    st.markdown(
        "Automatically censor music using Demucs, Whisper, ChatGPT, and FFmpeg. "
        "Upload music files and let AI identify and silence inappropriate words so they are safe to play on public radio."
    )

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        demucs_model = st.selectbox(
            "Demucs Model",
            ["htdemucs", "htdemucs_ft"],
            index=0
        )
        
        whisper_model = st.selectbox(
            "Whisper Model",
            ["turbo", "large"],
            index=0
        )
        
        chatgpt_model = st.selectbox(
            "ChatGPT Model",
            ["gpt-5-mini", "gpt-5.1"],
            index=0,
        )
        
        silence_padding = st.slider(
            "Silence Padding (seconds)",
            min_value=0.0,
            max_value=0.5,
            value=0.0,
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
        uploaded_files = st.file_uploader(
            "Upload Audio File(s)",
            type=["mp3", "wav", "flac", "m4a", "ogg"],
            accept_multiple_files=True,
            help="Upload music file(s) to process"
        )
        
        # Display download table if outputs exist from previous processing
        if "output_files" in st.session_state and any(st.session_state.output_files.get("Original Audio", [])):
            display_download_table(st.session_state.output_files)
        
        if uploaded_files is not None:

            # Clear previous outputs when starting new processing
            if st.button("🚀 Process Audio", type="primary", use_container_width=True):
                # Clear previous outputs
                clear_output_tracking()
                initialize_output_tracking()

                # Check API key
                if not st.secrets["OPENAI_API_KEY"]:
                    st.error("❌ OPENAI_API_KEY environment variable is not set!")
                    st.stop()

                progress_bar = st.progress(0)
                file_processing = st.empty()
                status_text = st.empty()

                # Process each uploaded file
                for index, uploaded_file in enumerate(uploaded_files):
                    progress_addder = index / len(uploaded_files) # tracks percentage of files processed
                    file_processing.text(f"Processing file {index + 1} of {len(uploaded_files)}: {uploaded_file.name}")
                    
                    # Create a subdirectory for this file's outputs
                    file_output_dir = st.session_state.output_dir / f"file_{index}"
                    file_output_dir.mkdir(parents=True, exist_ok=True)

                    try:
                        # Save uploaded file to persistent directory
                        input_audio_path = file_output_dir / uploaded_file.name
                        with open(input_audio_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Track original audio path
                        st.session_state.output_files["Original Audio"].append(str(input_audio_path))
                        # Step 1: Isolate vocals with Demucs
                        status_text.text(f"🎤 Step 1/5: Isolating vocals with Demucs... ({uploaded_file.name})")
                        progress_bar.progress((0.1 / len(uploaded_files)) + progress_addder)
                        
                        separated_dir = file_output_dir / "separated"
                        vocals_path, instrumental_path, separated_stems = isolate_vocals(
                            input_audio_path,
                            separated_dir,
                            model=demucs_model
                        )
                        
                        # Copy vocals and instrumental to main output directory for easy access
                        saved_vocals_path = file_output_dir / "vocals.wav"
                        saved_instrumental_path = file_output_dir / "instrumental.wav"
                        shutil.copy2(vocals_path, saved_vocals_path)
                        shutil.copy2(instrumental_path, saved_instrumental_path)
                        
                        st.session_state.output_files["Vocals"].append(str(saved_vocals_path))
                        st.session_state.output_files["Instrumental"].append(str(saved_instrumental_path))
                        
                        progress_bar.progress((0.3 / len(uploaded_files)) + progress_addder)
                        
                        # Step 2: Transcribe vocals with Whisper
                        status_text.text(f"📝 Step 2/5: Transcribing vocals with Whisper... ({uploaded_file.name})")
                        
                        transcription = transcribe_vocals(
                            vocals_path,
                            model_size=whisper_model
                        )
                        
                        # Save transcription to file
                        transcription_path = file_output_dir / "transcription.txt"
                        with open(transcription_path, "w", encoding="utf-8") as f:
                            f.write(f"Language: {transcription.get('language', 'unknown')}\n")
                            f.write(f"Total words: {len(transcription['words'])}\n\n")
                            f.write("Full transcription:\n")
                            f.write(transcription["text"])
                            f.write("\n\nWord-level timestamps:\n")
                            for word in transcription["words"]:
                                f.write(f"[{word['start']:.2f}s-{word['end']:.2f}s] {word['word']}\n")
                        
                        st.session_state.output_files["Transcription"].append(str(transcription_path))
                        progress_bar.progress((0.5 / len(uploaded_files)) + progress_addder)
                        
                        # Step 3: Get few-shot examples
                        status_text.text(f"🤖 Step 3/5: Analyzing with ChatGPT... ({uploaded_file.name})")
                        progress_bar.progress((0.6 / len(uploaded_files)) + progress_addder)
                        
                        few_shot_examples = st.session_state.get("few_shot_examples", "")
                        
                        censored_words = censor_with_chatgpt(
                            transcription["words"],
                            few_shot_examples=few_shot_examples if few_shot_examples else None,
                            model=chatgpt_model
                        )
                        
                        # Save censored words to JSON file
                        censored_words_path = file_output_dir / "censored_words.json"
                        if censored_words:
                            with open(censored_words_path, "w", encoding="utf-8") as f:
                                json.dump(censored_words, f, indent=2)
                            st.session_state.output_files["Censored Words"].append(str(censored_words_path))
                        else:
                            # No censored words, add empty string
                            st.session_state.output_files["Censored Words"].append("")
                        
                        progress_bar.progress((0.8 / len(uploaded_files)) + progress_addder)
                        
                        # Step 5: Process audio with FFmpeg
                        status_text.text(f"🔇 Step 4/5: Silencing vocals and recombining... ({uploaded_file.name})")
                        
                        output_audio_path = file_output_dir / "censored_output.wav"
                        process_censored_audio(
                            vocals_path,
                            instrumental_path,
                            censored_words,
                            output_audio_path,
                            padding=silence_padding
                        )
                        
                        st.session_state.output_files["Censored Audio"].append(str(output_audio_path))
                        progress_bar.progress((1.0 / len(uploaded_files)) + progress_addder)
                        
                    except Exception as e:
                        st.error(f"❌ Error during processing {uploaded_file.name}: {str(e)}")
                        st.exception(e)
                        # Ensure all columns have the same length by adding empty strings for missing outputs
                        current_length = len(st.session_state.output_files["Original Audio"])
                        expected_length = index + 1
                        
                        # Fill any missing entries with empty strings to keep columns aligned
                        for key in st.session_state.output_files:
                            while len(st.session_state.output_files[key]) < expected_length:
                                st.session_state.output_files[key].append("")
                        continue
                
                file_processing.text("")
                status_text.text("")
                st.success("✅ All files processed!")
                
                # Display download table
                display_download_table(st.session_state.output_files) 
    with tab2:
        st.header("Few-shot Examples")
        st.markdown(
            "Provide few-shot examples to help ChatGPT identify words to censor. "
            "These examples should show the expected input/output format."
        )
        
        default_examples = """Example 1:
    Input transcript:
    [0.5s-0.8s] you
    [1.0s-1.3s] should
    [1.5s-1.8s] go
    [2.0s-2.3s] and
    [2.5s-2.8s] fuck
    [3.0s-3.3s] yourself

    Output JSON:
    {
    "words": [
        { "word": "fuck", "start": 2.5, "end": 2.8 }
    ]
    }

    Example 2:
    Input transcript:
    [0.2s-0.5s] what
    [0.6s-0.9s] the
    [1.0s-1.4s] frick
    [1.5s-1.8s] is
    [2.0s-2.3s] this

    Output JSON:
    {
    "words": []
    }"""
        
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

if __name__ == "__main__":
    main()

