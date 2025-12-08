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

prompt_base = """
You are a content moderation assistant at a radio station in the United States. 
Your task is to identify specific words that violate FCC guidelines for indecent or profane content. 
The defintions of indecent and profane content are as follows:

Indecent Content: portrays sexual or excretory organs or activities in a way that is patently
offensive but does not meet the three-prong test for obscenity.

Profane Content: “grossly offensive” language that is considered a public nuisance.

Below is a modified guide from Public Radio Exchange on how to follow FCC guidelines:

Words that must always be dropped:
shit, piss, fuck, cunt, cocksucker, motherfucker, tits- and variations of these words-
such as bullshit (see below on this particular word).

The word 'pussy' when used in a sexual context is usually also to be censored,
although the band Pussy Riot is okay to say.

With scatological terms (obscenities relating to feces, urine, and defecation)
should always have the obscenity bleeped or dropped to silence. 
For example: 'shit,' 'bullsh*t,' and 'sh*thead.

Medical/anatomical terms are acceptable, e.g. anus, colon, penis, vagina, etc.

While the medical dictionary terms are always acceptable,
some words referring to parts of the human anatomy,
mostly the so-called “private” parts, are considered crude.
Thus, “vagina” is acceptable, “cunt” is not. The latter should be bleeped or dropped to silence.

Due to “cock” meaning both male bird and reproductive organ, context is absolutely key. Dropped in most cases.

Ass, horse’s ass, etc are allowable as mild euphemism- if the word donkey will fit, so will ‘ass’- 'asshole' IS an issue- edit it.
Religious profanities should generally be dropped. Words and expressions that religious people find profane and blasphemous.
Examples include 'God damn' and 'God damn you.' 
Although not technically against the rules, these have been bleeped or dropped to silence for certain audiences;
but not always- such as in the song  “Mississippi, God Damn”\n
Derogatory terms: Words or expressions that are used to denigrate and insult one's racial or ethnic background, gender or sexual orientation:
Examples in this area include “wog”, “wop”, “nigger”, “kike,” “gook”, “gypsy”, as well as anti-homosexual terms like “faggot,” etc.
“Nigger” and other racist/bigoted terms require particular cultural sensitivity and should be given priority consideration.

Instructions:
Analyze the transcript and identify words that should be censored.
Return ONLY a JSON object of the form: { \"words\": [ { \"word\": string, \"start\": number, \"end\": number } ] }
"""
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

def create_censoring_prompt(
    transcript_words: List[Dict[str, Any]],
    prompt_base: Optional[str] = prompt_base,
    few_shot_examples: Optional[str] = default_examples
) -> str:
    """Create a prompt for ChatGPT to identify words to censor.
    Args:
        transcript_words: List of word dicts with "word", "start", "end" keys
        prompt_base: Base instructions for prompt
        few_shot_examples: Optional string of few-shot examples to include in the prompt
    Returns:
        Prompt string
    """
    prompt_parts = []

    transcript_lines = []
    for word in transcript_words:
        transcript_lines.append(
            f'[{word["start"]:.2f}s-{word["end"]:.2f}s] {word["word"]}'
        )
    transcript_text = "\n".join(transcript_lines)

    prompt_parts.append(prompt_base)

    if few_shot_examples:
        prompt_parts.append("\nFew-shot Examples:\n" + few_shot_examples)

    prompt_parts.append("\nTranscript with Timestamps:")
    prompt_parts.append(transcript_text)

    return "\n".join(prompt_parts)


def censor_with_chatgpt(
    transcript_words: List[Dict[str, Any]],
    few_shot_examples: Optional[str] = None,
    model: str = "gpt-5.1-mini",
    api_key: Optional[str] = None,
    language: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Use ChatGPT Responses API to identify words to censor.
    Args:
        transcript_words: List of word dicts with "word", "start", "end" keys
        few_shot_examples: Optional string of few-shot examples to include in the prompt
        model: ChatGPT model name (default: "gpt-5.1-mini")
        api_key: OpenAI API key (if None, uses from streamlit secrets)
        language: Detected language code (e.g., "en"). Returns empty list if "unknown" or None
    Returns:
        List of censored word dicts with "word", "start", "end" keys
    """
    
    # Skip ChatGPT if language is unknown or not provided
    if not language or language == "unknown":
        return []

    if api_key is None:
        api_key = secrets["OPENAI_API_KEY"]
        if not api_key:
            raise ValueError(
                "OpenAI API key missing. Set OPENAI_API_KEY env var or pass api_key."
            )

    client = OpenAI(api_key=api_key)
    prompt = create_censoring_prompt(transcript_words, few_shot_examples=few_shot_examples)

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