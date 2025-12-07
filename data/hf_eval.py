import sys
from pathlib import Path
import logging
import pandas as pd
import json
import time
import csv
from typing import Dict, List, Any
import re
import difflib

logging.basicConfig(level=logging.INFO)
# Add project root to path
project_root = Path(__file__).resolve().parents[1] 
sys.path.insert(0, str(project_root))

from models.demucs_processor import isolate_vocals
from models.whisper_processor import transcribe_vocals
from data.datautils import load_metadata_jsonl, normalize_word, calculate_word_error_rate, calculate_timing_mse, normalize_transcript_words



def main():
    # Load metadata
    metadata_path = project_root / "data/jamendolyrics/metadata.jsonl"
    metadata = load_metadata_jsonl(metadata_path)
    
    results = []
    
    for audio_file in sorted(project_root.glob("data/jamendolyrics/subsets/de/mp3/*.mp3"))[:1]:  # Process first one for testing
        file_stem = audio_file.stem
        logging.info(f"Processing file: {file_stem}")

        
        # Get actual transcription from metadata
        if file_stem not in metadata:
            logging.warning(f"No metadata found for {file_stem}")
            continue
        
        actual_data = metadata[file_stem]
        actual_words = actual_data.get('words', [])
        
        if not actual_words:
            logging.warning(f"No words found for {file_stem}")
            continue
        
        try:
            # Isolate vocals (measure time)
            demucs_start = time.perf_counter()
            try:
                vocals_path, instrumental_path, _ = isolate_vocals(
                    input_audio_path=audio_file,
                    output_dir=project_root / "data/isolated_stems"
                )
                logging.info(f"Isolated vocals saved to: {vocals_path}")
            except Exception as demucs_error:
                logging.error(f"Failed to isolate vocals for {file_stem}: {demucs_error}")
                logging.info("Skipping file due to audio loading error...")
                continue
            demucs_end = time.perf_counter()
            demucs_time = demucs_end - demucs_start
            
            # Transcribe vocals
            # Transcribe vocals (measure time)
            whisper_start = time.perf_counter()
            try:
                transcription = transcribe_vocals(
                    audio_path=vocals_path,
                    model_size="turbo"
                )
                logging.info(f"Transcription completed. Transcript text: {transcription['text'][:100]}...")
            except Exception as whisper_error:
                logging.error(f"Failed to transcribe {file_stem}: {whisper_error}")
                continue
            whisper_end = time.perf_counter()
            whisper_time = whisper_end - whisper_start
            
            whisper_words = transcription.get('words', [])
            
            if not whisper_words:
                logging.warning(f"No words extracted from transcription for {file_stem}")
                continue
            
            # Normalize both whisper and actual words ONCE here
            # normalized whisper_words: list of dicts with keys 'word','start','end'
            whisper_norm = normalize_transcript_words(whisper_words, key='word')
            actual_norm = normalize_transcript_words(actual_words, key='text')

            # Save normalized transcripts to CSV for inspection / reuse
            norm_dir = project_root / "data/normalized_transcripts"
            norm_dir.mkdir(parents=True, exist_ok=True)
            csv_path = norm_dir / f"{file_stem}.csv"
            try:
                # Write rows aligning by index; if lengths differ, fill blanks
                max_len = max(len(whisper_norm), len(actual_norm))
                with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
                    writer = csv.writer(cf)
                    # Header
                    writer.writerow(['whisper_text','actual_text','whisper_start','actual_start','whisper_end','actual_end'])
                    for i in range(max_len):
                        w = whisper_norm[i] if i < len(whisper_norm) else {'word':'','start':'','end':''}
                        a = actual_norm[i] if i < len(actual_norm) else {'text':'','start':'','end':''}
                        writer.writerow([w.get('word',''), a.get('text',''), w.get('start',''), a.get('start',''), w.get('end',''), a.get('end','')])
            except Exception:
                logging.exception(f"Failed to write normalized transcript CSV for {file_stem}")

            # Calculate Word Error Rate using normalized lists
            wer_metrics = calculate_word_error_rate(whisper_norm, actual_norm)

            # Calculate timing MSE (only for correctly matched words)
            timing_metrics = calculate_timing_mse(whisper_norm, actual_norm)

            result = {
                'file': file_stem,
                'demucs_time': demucs_time,
                'whisper_time': whisper_time,
                **wer_metrics,
                **timing_metrics,
            }
            
            results.append(result)
            
            logging.info(f"✓ WER: {wer_metrics['wer']:.3f} | Correct: {wer_metrics['correct_matches']}/{wer_metrics['total_actual_words']}")
            logging.info(f"✓ Start MSE: {timing_metrics['start_mse']:.6f} | End MSE: {timing_metrics['end_mse']:.6f}")
            logging.info(f"✓ Matched words with timing: {timing_metrics['matched_words_with_timing']}")
            
        except Exception as e:
            logging.error(f"Unexpected error processing {file_stem}: {e}", exc_info=True)
            continue
    
    # Create comparison dataframe
    if results:
        df = pd.DataFrame(results)
        
        # Save results
        output_path = project_root / "data/transcription_comparison.csv"
        df.to_csv(output_path, index=False)
        logging.info(f"\nResults saved to: {output_path}")
        
        # Print summary statistics
        print("\n" + "="*70)
        print("TRANSCRIPTION COMPARISON SUMMARY")
        print("="*70)
        print(f"Successfully processed files: {len(df)}")
        print(f"\nWord Error Rate (WER) Statistics:")
        print(f"  Average WER: {df['wer'].mean():.3f}")
        print(f"  Min WER: {df['wer'].min():.3f}")
        print(f"  Max WER: {df['wer'].max():.3f}")
        print(f"  Total correct matches: {int(df['correct_matches'].sum())}")
        print(f"  Total substitutions: {int(df['substitutions'].sum())}")
        print(f"  Total deletions: {int(df['deletions'].sum())}")
        print(f"  Total insertions: {int(df['insertions'].sum())}")
        print(f"\nTiming Accuracy (MSE in seconds²):")
        print(f"  Average Start MSE: {df['start_mse'].mean():.6f}")
        print(f"  Average End MSE: {df['end_mse'].mean():.6f}")
        print(f"  Average Start Time Diff: {df['avg_start_diff'].mean():.6f}s")
        print(f"  Average End Time Diff: {df['avg_end_diff'].mean():.6f}s")
        print(f"  Avg matched words with timing: {df['matched_words_with_timing'].mean():.1f}")
        print(f"\nAverage processing time:")
        if 'demucs_time' in df.columns:
            print(f"  Average Demucs time: {df['demucs_time'].mean():.3f}s")
        if 'whisper_time' in df.columns:
            print(f"  Average Whisper time: {df['whisper_time'].mean():.3f}s")
        print("="*70)
        print("\nDetailed Results:")
        print(df.to_string())
    else:
        logging.warning("No results to save")


if __name__ == "__main__":
    main()
