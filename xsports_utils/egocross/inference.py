from __future__ import annotations

from typing import Any

from egocross.support import build_support_message


def build_messages(
    image_paths: list[str],
    prompt: str,
    support_examples: list[dict[str, Any]],
    support_root: str,
    images_per_example: int,
    frame_prefix_texts: list[str] | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for example in support_examples:
        messages.extend(
            build_support_message(
                example,
                support_root,
                images_per_example=images_per_example,
            )
        )
    query_content: list[dict[str, Any]] = []
    for idx, image_path in enumerate(image_paths):
        query_content.append({"type": "image", "image": image_path})
        if frame_prefix_texts is not None and idx < len(frame_prefix_texts):
            query_content.append(
                {"type": "text", "text": frame_prefix_texts[idx]},
            )
    query_content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": query_content})
    return messages


def run_inference(
    model,
    processor,
    image_paths: list[str],
    prompt: str,
    support_examples: list[dict[str, Any]],
    support_root: str,
    images_per_example: int,
    max_new_tokens: int = 16,
    frame_prefix_texts: list[str] | None = None,
) -> str:
    messages = build_messages(
        image_paths=image_paths,
        prompt=prompt,
        support_examples=support_examples,
        support_root=support_root,
        images_per_example=images_per_example,
        frame_prefix_texts=frame_prefix_texts,
    )
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
