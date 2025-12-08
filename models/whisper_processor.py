"""Whisper processor for transcribing vocals with word-level timestamps using faster-whisper."""

from torch.cuda import is_available as is_cuda_available
from torch.backends.mps import is_available as is_mps_available
from faster_whisper import WhisperModel
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import tempfile

# Import noise gate function
from models.ffmpeg_processor import apply_noise_gate

# Module logger
logger = logging.getLogger(__name__)
# Ensure there's a stdout handler so logs are visible in the Streamlit server terminal
if not logger.handlers:
    import sys
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.propagate = False


def transcribe_vocals(
    audio_path: Path,
    model_size: str = "turbo",
    language: Optional[str] = None,
    apply_noise_suppression: bool = True,
    noise_threshold_db: float = -40
) -> Dict[str, Any]:
    """Transcribe vocals using faster-whisper with word-level timestamps.
    
    Args:
        audio_path: Path to the vocals audio file
        model_size: Whisper model size ("turbo", "large")
        language: Language code (e.g., "en") or None for auto-detection
        apply_noise_suppression: Whether to apply noise gate filter before transcription
        noise_threshold_db: Noise threshold in dB for suppression (more negative = more aggressive)
        
    Returns:
        Dictionary containing:
        - "text": Full transcript text
        - "words": List of word dicts with "word", "start", "end" keys
        - "segments": List of segment dicts with timing and text info
        - "language": Detected language
    """
    # Apply noise gate if requested
    audio_to_transcribe = audio_path
    temp_file = None
    
    if apply_noise_suppression:
        try:
            # Create temporary file for noise-gated audio
            fd, temp_path = tempfile.mkstemp(suffix=".wav")
            import os
            os.close(fd)
            temp_file = Path(temp_path)
            
            logger.info(f"Applying noise gate (threshold: {noise_threshold_db} dB) to suppress background noise")
            apply_noise_gate(audio_path, temp_file, threshold_db=noise_threshold_db)
            audio_to_transcribe = temp_file
        except Exception as e:
            logger.warning(f"Failed to apply noise gate: {e}. Proceeding with original audio.")
            audio_to_transcribe = audio_path
    # Map common model names to faster-whisper model names
    model_map = {
        "turbo": "turbo",
        "large": "large-v3",
    }
    model_name = model_map.get(model_size, model_size)
    
    # Detect device: prefer CUDA > MPS > CPU
    if is_cuda_available():
        device = "cuda"
        compute_type = "float16"
    elif is_mps_available():
        device = "cpu"  # faster-whisper uses "cpu" for MPS via CTransformers backend
        compute_type = "int8"  # Use int8 quantization for efficiency on MPS
    else:
        device = "cpu"
        compute_type = "float32"
    
    # Load faster-whisper model
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    
    # Transcribe with word-level timestamps in batched fashion
    # model.transcribe returns a generator of segments and TranscriptionInfo
    segments_generator, info = model.transcribe(
        str(audio_to_transcribe),
        word_timestamps=True,
        without_timestamps=False
    )
    
    # Collect all segments from the generator and extract words
    words = []
    segments_list = []
    full_text_parts = []
    
    try:
        for segment in segments_generator:
            segment_dict = {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "words": []
            }
            
            full_text_parts.append(segment.text)
            
            # Extract word-level timestamps if available
            if segment.words:
                for word_info in segment.words:
                    word_entry = {
                        "word": word_info.word.strip(),
                        "start": word_info.start,
                        "end": word_info.end
                    }
                    words.append(word_entry)
                    segment_dict["words"].append(word_entry)
            
            segments_list.append(segment_dict)
    finally:
        # Clean up temporary noise-gated audio file if it was created
        if temp_file and temp_file.exists():
            try:
                import os
                os.remove(str(temp_file))
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {temp_file}: {e}")
    
    # Construct the full text from segments
    full_text = " ".join(full_text_parts).strip()
    
    # Check if the entire audio was silenced (no transcription output)
    if not full_text and not words and not segments_list:
        logger.warning("Audio file was completely silenced or contains no speech. Returning empty transcription.")
        return {
            "text": "",
            "words": [],
            "segments": [],
            "language": "unknown"
        }
    
    return {
        "text": full_text,
        "words": words,
        "segments": segments_list,
        "language": info.language if hasattr(info, "language") else "unknown"
    }

