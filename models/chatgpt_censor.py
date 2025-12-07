"""ChatGPT processor for identifying words to censor from a music transcript."""

from json import loads
from typing import List, Dict, Any, Optional
from openai import OpenAI
from pydantic import BaseModel
from streamlit import secrets



class Word(BaseModel):
    word: str
    start: float
    end: float

class CensorWords(BaseModel):
    words: List[Word]

def create_censoring_prompt(
    transcript_words: List[Dict[str, Any]],
    few_shot_examples: Optional[str] = None
) -> str:
    """Create a prompt for ChatGPT to identify words to censor.
    Args:
        transcript_words: List of word dicts with "word", "start", "end" keys
        few_shot_examples: Optional string of few-shot examples to include in the prompt
    Returns:
        Prompt string
    """


    transcript_lines = []
    for word in transcript_words:
        transcript_lines.append(
            f'[{word["start"]:.2f}s-{word["end"]:.2f}s] {word["word"]}'
        )
    transcript_text = "\n".join(transcript_lines)

    prompt_parts = [
        "You are a content moderation assistant. Your task is to identify words "
        "that should be censored in a music transcript based on profanity, explicit content, "
        "or inappropriate language."
    ]

    if few_shot_examples:
        prompt_parts.append("\n## Few-shot Examples:\n" + few_shot_examples)
        prompt_parts.append("\n## Current Transcript:")
    else:
        prompt_parts.append("\n## Transcript with Timestamps:")

    prompt_parts.append(transcript_text)

    prompt_parts.append(
        "\n## Instructions:\n"
        "Analyze the transcript and identify words that should be censored. "
        "Return ONLY a JSON object of the form: { \"words\": [ { \"word\": string, \"start\": number, \"end\": number } ] }\n"
    )

    return "\n".join(prompt_parts)


def censor_with_chatgpt(
    transcript_words: List[Dict[str, Any]],
    few_shot_examples: Optional[str] = None,
    model: str = "gpt-5.1",
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Use ChatGPT Responses API to identify words to censor.
    Args:
        transcript_words: List of word dicts with "word", "start", "end" keys
        few_shot_examples: Optional string of few-shot examples to include in the prompt
        model: ChatGPT model name (default: "gpt-5.1")
        api_key: OpenAI API key (if None, uses from streamlit secrets)
    Returns:
        List of censored word dicts with "word", "start", "end" keys
    """

    if api_key is None:
        api_key = secrets["OPENAI_API_KEY"]
        if not api_key:
            raise ValueError(
                "OpenAI API key missing. Set OPENAI_API_KEY env var or pass api_key."
            )

    client = OpenAI(api_key=api_key)
    prompt = create_censoring_prompt(transcript_words, few_shot_examples)

    try:
        response = client.responses.parse(
            model=model,
            input=prompt,
            text_format=CensorWords
        )

        if response.output_parsed is not None:
            return [w.model_dump() for w in response.output_parsed.words]
        else:
            # fallback: try to extract/parse raw JSON text
            try:
                data = loads(response.output_text)
                if isinstance(data, dict) and "words" in data:
                    return data["words"]
                if isinstance(data, list):
                    return data
            except Exception:
                pass
            return []

    except Exception as e:
        raise Exception(f"OpenAI Responses API error: {e}")