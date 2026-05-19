"""Single-turn MCQ generation with optional per-frame time captions."""

from __future__ import annotations

from typing import Any


def run_direct_mcq(
    model: Any,
    processor: Any,
    image_paths: list[str],
    prompt: str,
    max_new_tokens: int,
    frame_prefix_texts: list[str] | None = None,
) -> str:
    """Multi-image chat template plus greedy decode."""

    query_content: list[dict[str, Any]] = []
    for idx, image_path in enumerate(image_paths):
        if frame_prefix_texts is not None:
            query_content.append({"type": "text", "text": frame_prefix_texts[idx]})
        query_content.append({"type": "image", "image": image_path})
    query_content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": query_content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def build_direct_mcq_prompt(question_text: str, options: list[str]) -> str:
    from industry_infer.parsing import normalize_option

    lines = [
        "You are answering a multiple-choice question about an egocentric industrial task video.",
        "The provided images are in chronological order.",
        "Use visible evidence from the images.",
        "Do not rely on commonsense if the images do not support it.",
        "",
        "Question:",
        question_text.strip(),
        "",
        "Options:",
        *[normalize_option(str(option)) for option in options],
        "",
        "Answer with only the single letter: A, B, C, or D.",
    ]
    return "\n".join(lines)
