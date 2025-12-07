"""Demucs processor for isolating vocals from music."""

from pathlib import Path
from typing import Tuple, Dict
import logging
import torch

from demucs.api import Separator, save_audio

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
        segment=8 
    )

    logger.info("Loaded top-level model type: %s", type(separator.model))
    if hasattr(separator.model, "models"):
        logger.info("Top-level model contains %d submodels", len(separator.model.models))
    # Some Demucs variants (HTDemucs/HDemucs) keep a `use_train_segment` flag
    # which forces tensors to be reshaped to the training segment length. This
    # can lead to shape mismatches during inference when chunking/splitting is
    # used. Disable it on the loaded model to avoid the runtime reshape error.
    # Disable `use_train_segment` on the loaded model and any submodels.
    # BagOfModels wraps multiple models; ensure we cover both cases so the
    # HTDemucs/HDemucs forward method does not try to `.view(...)` to the
    # training segment length during inference.
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
        logger.info("Submodel[%d] type: %s", i, type(m))
        if hasattr(m, "segment"):
            logger.info("Submodel[%d] segment: %s", i, getattr(m, "segment"))
        if hasattr(m, "use_train_segment"):
            logger.info("Submodel[%d] use_train_segment before: %s", i, getattr(m, "use_train_segment"))
            try:
                m.use_train_segment = False
            except Exception:
                logger.exception("Failed to disable use_train_segment on submodel[%d]", i)
            else:
                logger.info("Submodel[%d] use_train_segment after: %s", i, getattr(m, "use_train_segment"))
    
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

