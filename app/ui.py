"""Gradio interface for LegalBrain AI."""

import os

import gradio as gr
import httpx


API_URL = os.getenv(
    "LEGALBRAIN_API_URL",
    "http://127.0.0.1:8000",
)

DISCLAIMER = (
    "**Important:** LegalBrain AI is an experimental demo. It may make "
    "mistakes, and its information may be outdated. Verify important legal "
    "information against current official sources. This is not legal advice."
)


def ask_legalbrain(question: str):
    if not question or not question.strip():
        return "Please enter a legal question.", ""

    try:
        response = httpx.post(
            f"{API_URL}/answer",
            json={
                "question": question.strip(),
                "max_tokens": 180,
                "temperature": 0.2,
            },
            timeout=180.0,
        )
        response.raise_for_status()
        data = response.json()

    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        return f"**Request failed:** {detail}", ""

    except httpx.RequestError:
        return (
            "Could not connect to the LegalBrain API. "
            "Make sure the backend is running on port 8000.",
            "",
        )

    metadata = (
        f"**Model:** `{data['model']}`  \n"
        f"**Adapter:** `{data['adapter']}`  \n"
        f"**Latency:** {data['latency_ms']} ms  \n"
        f"**Cache:** {'hit' if data['cache_hit'] else 'miss'}"
    )

    return data["answer"], metadata


def clear_fields():
    return "", "", ""


with gr.Blocks(title="LegalBrain AI") as demo:
    gr.Markdown(
        """
        # LegalBrain AI

        Ask a question about Indian law and receive a model-generated answer.
        This demo lets you explore a fine-tuned open-source language model.
        """
    )

    gr.Markdown(DISCLAIMER)

    question = gr.Textbox(
        label="Your legal question",
        placeholder="e.g. What does Article 32 of the Constitution of India provide?",
        lines=4,
    )

    with gr.Row():
        ask_button = gr.Button("Ask LegalBrain", variant="primary")
        clear_button = gr.Button("Clear")

    gr.Markdown("## Answer")
    answer_output = gr.Markdown()

    metadata_output = gr.Markdown()

    ask_button.click(
        fn=ask_legalbrain,
        inputs=[question],
        outputs=[answer_output, metadata_output],
    )

    question.submit(
        fn=ask_legalbrain,
        inputs=[question],
        outputs=[answer_output, metadata_output],
    )

    clear_button.click(
        fn=clear_fields,
        inputs=[],
        outputs=[question, answer_output, metadata_output],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
    )