"""FFmpeg processor for silencing vocals and recombining audio."""

from subprocess import run
from tempfile import mkstemp
from pathlib import Path
from typing import List, Dict, Any, Tuple


def apply_noise_gate(
    input_path: Path,
    output_path: Path,
    threshold_db: float = -40
) -> Path:
    """Apply noise gate filter to suppress background noise.
    
    Uses FFmpeg's silenceremove filter to remove nearly-silent audio below the threshold.
    
    Args:
        input_path: Path to the input audio file
        output_path: Path where to save the noise-gated audio
        threshold_db: Noise threshold in dB (more negative = more aggressive)
                      Default -40 dB suppresses quieter background noise
        
    Returns:
        Path to the output file
    """
    # Use silenceremove to suppress audio below threshold
    # Also use highpass filter to remove very low frequency rumble
    filter_complex = f"silenceremove=start_periods=1:start_duration=0.5:start_threshold={threshold_db}dB:stop_periods=-1:stop_duration=0.5:stop_threshold={threshold_db}dB,highpass=f=80"
    
    run(
        [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-af", filter_complex,
            "-c:a", "pcm_s16le",
            str(output_path)
        ],
        check=True,
        capture_output=True
    )
    
    return output_path


def create_timestamp_ranges(censored_words: List[Dict[str, Any]], 
                           padding: float = 0.1) -> List[Tuple[float, float]]:
    """Create timestamp ranges for silencing, with optional padding.
    
    Args:
        censored_words: List of censored word dicts with 'start' and 'end'
        padding: Additional time (seconds) to add before and after each word
        
    Returns:
        List of (start, end) tuples for silence ranges
    """
    ranges = []
    for word in censored_words:
        start = max(0, word["start"] - padding)
        end = word["end"] + padding
        ranges.append((start, end))
    
    # Merge overlapping ranges
    if not ranges:
        return []
    
    ranges.sort(key=lambda x: x[0])
    merged = [ranges[0]]
    
    for current in ranges[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            # Overlapping or adjacent, merge them
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    
    return merged


def silence_vocals_at_timestamps(
    vocals_path: Path,
    output_path: Path,
    censored_words: List[dict],
    padding: float = 0.1
) -> Path:
    """Silence vocals at specified timestamps using FFmpeg.
    
    Args:
        vocals_path: Path to the vocals audio file
        output_path: Path where to save the silenced vocals
        censored_words: List of dicts with "start" and "end" keys (timestamps in seconds)
        padding: Additional time (seconds) to add before and after each word
        
    Returns:
        Path to the output file
    """
    if not censored_words:
        # No words to censor, just copy the file
        run(
            ["ffmpeg", "-y", "-i", str(vocals_path), "-c", "copy", str(output_path)],
            check=True,
            capture_output=True
        )
        return output_path
    
    # Create timestamp ranges with padding
    silence_ranges = create_timestamp_ranges(censored_words, padding)
    
    # Build FFmpeg filter to silence at specified times
    # For multiple ranges, chain volume filters
    if len(silence_ranges) == 1:
        start, end = silence_ranges[0]
        filter_complex = f"volume=enable='between(t,{start},{end})':volume=0"
    else:
        # Chain multiple volume filters - each one silences a specific range
        # FFmpeg will apply them sequentially
        filter_parts = []
        for start, end in silence_ranges:
            filter_parts.append(f"volume=enable='between(t,{start},{end})':volume=0")
        filter_complex = ",".join(filter_parts)
    
    # Apply filter
    run(
        [
            "ffmpeg", "-y",
            "-i", str(vocals_path),
            "-af", filter_complex,
            "-c:a", "pcm_s16le",  # Ensure output is WAV-compatible
            str(output_path)
        ],
        check=True,
        capture_output=True
    )
    
    return output_path


def recombine_audio(
    vocals_path: Path,
    instrumental_path: Path,
    output_path: Path
) -> Path:
    """Recombine silenced vocals with instrumental track.
    
    Args:
        vocals_path: Path to the (censored) vocals audio file
        instrumental_path: Path to the instrumental audio file
        output_path: Path where to save the final combined audio
        
    Returns:
        Path to the output file
    """
    # Use FFmpeg to mix the two audio tracks
    run(
        [
            "ffmpeg", "-y",
            "-i", str(vocals_path),
            "-i", str(instrumental_path),
            "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=longest",
            "-c:a", "pcm_s16le",
            str(output_path)
        ],
        check=True,
        capture_output=True
    )
    
    return output_path


def process_censored_audio(
    vocals_path: Path,
    instrumental_path: Path,
    censored_words: List[dict],
    output_path: Path,
    padding: float = 0.1
) -> Path:
    """Complete pipeline: silence vocals and recombine with instrumental.
    
    Args:
        vocals_path: Path to the original vocals audio file
        instrumental_path: Path to the instrumental audio file
        censored_words: List of dicts with "start" and "end" keys
        output_path: Path where to save the final censored audio
        padding: Additional time (seconds) to add before and after each word
        
    Returns:
        Path to the output file
    """

    
    # Create temporary file for silenced vocals
    temp_vocals = Path(mkstemp(suffix=".wav", prefix="censored_vocals_")[1])
    
    try:
        # Silence vocals at censored timestamps
        silence_vocals_at_timestamps(
            vocals_path,
            temp_vocals,
            censored_words,
            padding
        )
        
        # Recombine with instrumental
        recombine_audio(
            temp_vocals,
            instrumental_path,
            output_path
        )
        
        return output_path
        
    finally:
        # Clean up temporary file
        if temp_vocals.exists():
            temp_vocals.unlink()

