"""Demucs processor for isolating vocals from music."""

from pathlib import Path
from typing import Tuple, Dict
import logging
import torch

from demucs.api import Separator, save_audio

def isolate_vocals(
    input_audio_path: Path,
    output_dir: Path,
    model: str = "htdemucs",
    device: str = "auto"
) -> Tuple[Path, Path, Dict[str, torch.Tensor]]:
    """Isolate vocals from an audio file using Demucs.
    
    Args:
        input_audio_path: Path to the input audio file
        output_dir: Directory where to save separated stems
        model: Demucs model name (default: "htdemucs")
        device: Device to use ("auto", "cpu", "cuda", "mps")
        
    Returns:
        Tuple of (vocals_path, instrumental_path, separated_stems_dict)
        where separated_stems_dict contains all stems (vocals, drums, bass, other)
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    separator = Separator(
        model=model,
        device=device,
        progress=False,
        split=True,
        segment=10
    )


    # By default, htdemucs sets  `use_train_segment` to true, which forces input segments to be of fixed size.
    # This can lead to issues when processing audio files of arbitrary lengths.
    # htdemucs is stored by itself as a submodel in a BagOfModels
    # The code below disables `use_train_segment` for htdemucs.
    model_obj = separator.model
    models_to_update = []
    if hasattr(model_obj, "models"):
        try:
            models_to_update = list(model_obj.models)
        except Exception:
            models_to_update = []
    else:
        models_to_update = [model_obj]

    for i, m in enumerate(models_to_update):
        if hasattr(m, "use_train_segment"):
            try:
                m.use_train_segment = False
            except Exception:
                pass
    

    _, separated = separator.separate_audio_file(input_audio_path)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save vocals
    vocals_path = output_dir / "vocals.wav"
    save_audio(separated["vocals"], str(vocals_path), samplerate=separator.samplerate)
    
    # Create instrumental track (drums + bass + other)
    instrumental = torch.zeros_like(separated["vocals"])
    for stem_name in ["drums", "bass", "other"]:
        if stem_name in separated:
            instrumental += separated[stem_name]
    
    instrumental_path = output_dir / "instrumental.wav"
    save_audio(instrumental, str(instrumental_path), samplerate=separator.samplerate)
    
    return vocals_path, instrumental_path, separated

