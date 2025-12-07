import sys
from pathlib import Path
import logging
import json
from typing import Dict, List, Any
import re
import difflib

logging.basicConfig(level=logging.INFO)

def load_metadata_jsonl(metadata_path: Path) -> Dict[str, Dict]:
    """Load metadata from JSONL file and index by filename."""
    metadata = {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                # Extract filename from file_name field
                file_name = Path(entry.get('file_name', '')).stem
                metadata[file_name] = entry
    return metadata

def normalize_word(word: str) -> str:
    """Normalize word: lowercase and remove punctuation."""
    # Convert to lowercase
    word = word.lower()
    # Remove punctuation
    word = re.sub(r'[^\w]', '', word)
    return word

def calculate_word_error_rate(whisper_words: List[Dict], actual_words: List[Dict]) -> Dict[str, Any]:
    """Calculate Word Error Rate between Whisper and actual transcriptions.
    
    Words are normalized (lowercase, punctuation removed) for comparison.
    
    Returns:
        Dict with WER metrics:
        - wer: Word error rate (0-1)
        - substitutions: Count of substitutions
        - deletions: Count of deletions
        - insertions: Count of insertions
        - correct_matches: Count of correctly matched words
        - total_actual_words: Total words in actual transcription
    """
    # Assumes input lists are already normalized: actual_words[].text and whisper_words[].word
    actual_normalized = [w.get('text', '') for w in actual_words]
    whisper_normalized = [w.get('word', '') for w in whisper_words]

    # Use difflib to align sequences
    matcher = difflib.SequenceMatcher(None, actual_normalized, whisper_normalized)
    matching_blocks = matcher.get_matching_blocks()

    # Count matches
    correct_matches = sum(block.size for block in matching_blocks)

    total_actual = len(actual_normalized)
    total_whisper = len(whisper_normalized)

    # Substitutions are minimum of unmatched words in both
    unmatched_actual = total_actual - correct_matches
    unmatched_whisper = total_whisper - correct_matches

    substitutions = min(unmatched_actual, unmatched_whisper)
    deletions = max(0, unmatched_actual - substitutions)  # Words in actual but not in whisper
    insertions = max(0, unmatched_whisper - substitutions)  # Words in whisper but not in actual

    # Calculate WER
    wer = (substitutions + deletions + insertions) / total_actual if total_actual > 0 else 0.0
    
    return {
        'wer': wer,
        'substitutions': substitutions,
        'deletions': deletions,
        'insertions': insertions,
        'correct_matches': correct_matches,
        'total_actual_words': total_actual,
    }

def calculate_timing_mse(whisper_words: List[Dict], actual_words: List[Dict]) -> Dict[str, Any]:
    """Calculate Mean Squared Error between word timing for correctly matched words.
    
    Only considers words that are transcribed correctly (after normalization).
    
    Returns:
        Dict with timing metrics:
        - start_mse: Mean squared error for start times
        - end_mse: Mean squared error for end times
        - matched_words_with_timing: Number of words matched with timing info
        - avg_start_diff: Average absolute difference in start times (for reference)
        - avg_end_diff: Average absolute difference in end times (for reference)
    """
    if not actual_words or not whisper_words:
        return {
            'start_mse': 0,
            'end_mse': 0,
            'matched_words_with_timing': 0,
            'avg_start_diff': 0,
            'avg_end_diff': 0,
        }
    
    # Assumes `actual_words` and `whisper_words` already contain normalized text
    actual_normalized = [w.get('text', '') for w in actual_words]
    whisper_normalized = [w.get('word', '') for w in whisper_words]

    # Find aligned matches
    matcher = difflib.SequenceMatcher(None, actual_normalized, whisper_normalized)
    matching_blocks = matcher.get_matching_blocks()
    
    start_diffs_sq = []
    end_diffs_sq = []
    start_diffs_abs = []
    end_diffs_abs = []
    matched_count = 0
    
    for block in matching_blocks:
        # block = (actual_idx, whisper_idx, size)
        for i in range(block.size):
            actual_idx = block.a + i
            whisper_idx = block.b + i
            
            actual_word = actual_words[actual_idx]
            whisper_word = whisper_words[whisper_idx]
            
            # Calculate timing differences
            start_diff = actual_word.get('start', 0) - whisper_word.get('start', 0)
            end_diff = actual_word.get('end', 0) - whisper_word.get('end', 0)
            
            start_diffs_sq.append(start_diff ** 2)
            end_diffs_sq.append(end_diff ** 2)
            start_diffs_abs.append(abs(start_diff))
            end_diffs_abs.append(abs(end_diff))
            matched_count += 1
    
    # Calculate MSE
    start_mse = sum(start_diffs_sq) / len(start_diffs_sq) if start_diffs_sq else 0
    end_mse = sum(end_diffs_sq) / len(end_diffs_sq) if end_diffs_sq else 0
    avg_start_diff = sum(start_diffs_abs) / len(start_diffs_abs) if start_diffs_abs else 0
    avg_end_diff = sum(end_diffs_abs) / len(end_diffs_abs) if end_diffs_abs else 0
    
    return {
        'start_mse': start_mse,
        'end_mse': end_mse,
        'matched_words_with_timing': matched_count,
        'avg_start_diff': avg_start_diff,
        'avg_end_diff': avg_end_diff,
    }


def combine_lyrics_annotations():
    """Placeholder for combining lyrics annotations."""
    pass

def normalize_transcript_words(words: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    """Normalize transcript words by lowercasing and removing punctuation.
    
    Args:
        words: List of word dicts with timing info
        key: Key in dict to normalize ("word" for whisper, "text" for actual)
        
    Returns:
        List of normalized word dicts with timing info
    """
    normalized = []
    for w in words:
        token = w.get(key, '')
        norm = normalize_word(token)
        normalized.append({
            key: norm,
            'start': w.get('start', 0),
            'end': w.get('end', 0)
        })
    return normalized