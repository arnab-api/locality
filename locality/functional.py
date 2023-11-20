from typing import Any, Optional

import names
import numpy as np
import torch


def find_token_range(
    string: str,
    substring: str,
    tokenizer=None,
    occurrence: int = 0,
    offset_mapping=None,
    **kwargs: Any,
) -> tuple[int, int]:
    """Find index range of tokenized string containing tokens for substring.

    The kwargs are forwarded to the tokenizer.

    A simple example:

        string = 'The batman is the night.'
        substring = 'batman'
        tokenizer = ...

        # Example tokenization: ['the', 'bat', '##man', 'is', 'the', 'night']
        assert find_token_range(string, substring, tokenizer) == (1, 3)

    Args:
        string: The string.
        substring: The substring to find token range for.
        tokenizer: The tokenizer. If not set, offset_mapping must be.
        occurrence: The occurence of the substring to look for.
            Zero indexed. Defaults to 0, the first occurrence.
        offset_mapping: Precomputed offset mapping. If not set, tokenizer will be run.

    Raises:
        ValueError: If substring is not actually in string or if banned
            kwargs are specified.

    Returns:
        Tuple[int, int]: The start (inclusive) and end (exclusive) token idx.
    """
    if tokenizer is None and offset_mapping is None:
        raise ValueError("must set either tokenizer= or offset_mapping=")
    if "return_offsets_mapping" in kwargs:
        raise ValueError("cannot set return_offsets_mapping")
    if substring not in string:
        raise ValueError(f'"{substring}" not found in "{string}"')
    if occurrence < 0:
        # If occurrence is negative, count from the right.
        char_start = string.rindex(substring)
        for _ in range(-1 - occurrence):
            try:
                char_start = string.rindex(substring, 0, char_start)
            except ValueError as error:
                raise ValueError(
                    f"could not find {-occurrence} occurrences "
                    f'of "{substring} in "{string}"'
                ) from error
    else:
        char_start = string.index(substring)
        for _ in range(occurrence):
            try:
                char_start = string.index(substring, char_start + 1)
            except ValueError as error:
                raise ValueError(
                    f"could not find {occurrence + 1} occurrences "
                    f'of "{substring} in "{string}"'
                ) from error
    char_end = char_start + len(substring)

    if offset_mapping is None:
        assert tokenizer is not None
        tokens = tokenizer(string, return_offsets_mapping=True, **kwargs)
        offset_mapping = tokens.offset_mapping

    token_start, token_end = None, None
    for index, (token_char_start, token_char_end) in enumerate(offset_mapping):
        # Skip special tokens # ! Is this the proper way to do this?
        if token_char_start == token_char_end:
            continue
        if token_start is None:
            if token_char_start <= char_start and token_char_end >= char_start:
                token_start = index
        if token_end is None:
            if token_char_start <= char_end and token_char_end >= char_end:
                token_end = index
                break

    assert token_start is not None
    assert token_end is not None
    assert token_start <= token_end
    return (token_start, token_end + 1)


@torch.inference_mode()
def predict_next_token(model, tokenizer, prompt, k=5, batch_size=2):
    """Compute the next token."""
    if isinstance(prompt, str):
        prompt = [prompt]
    inputs = tokenizer(prompt, return_tensors="pt", padding="longest").to(model.device)
    with torch.inference_mode():
        batched_logits = []
        for i in range(0, len(inputs.input_ids), batch_size):
            batch_outputs = model(
                input_ids=inputs.input_ids[i : i + batch_size],
                attention_mask=inputs.attention_mask[i : i + batch_size],
            )
            batched_logits.append(batch_outputs.logits)

            if "cuda" in str(model.device):
                torch.cuda.empty_cache()

        logits = torch.cat(batched_logits, dim=0)

    next_token_probs = logits[:, -1].float().softmax(dim=-1)
    next_token_topk = next_token_probs.topk(dim=-1, k=k)

    predictions = []
    for token_ids, token_probs in zip(next_token_topk.indices, next_token_topk.values):
        predictions.append(
            [
                (tokenizer.decode(token_id), prob.item(), token_id.item())
                for token_id, prob in zip(token_ids, token_probs)
            ]
        )
    return predictions
