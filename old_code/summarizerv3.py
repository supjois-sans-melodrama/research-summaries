"""
Research Paper Summarizer (Gradio, with audio)
----------------------------------------------
Upload a research paper (PDF) or paste its text. You get a written summary
(title, authors, date, main topic, abstract, experiment results, conclusion)
plus an audio version read aloud.

Run:
    pip install -r requirements.txt
    export OPENROUTER_API_KEY="your-key"     # Windows: set OPENROUTER_API_KEY=your-key
    python app.py
"""

import json
import os
import re
import tempfile
import time

import gradio as gr
import pymupdf
from gtts import gTTS
from openai import OpenAI

try:  # optional: load OPENROUTER_API_KEY from a local .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Free models change over time; see https://openrouter.ai/models?max_price=0
# Comma-separated list, tried in order. Override with SUMMARIZER_MODEL="a,b,c".
MODELS = [
    m.strip()
    for m in os.environ.get(
        "SUMMARIZER_MODEL", "inclusionai/ling-3.1-flash,apodex/apodex-1.1-mini:free"
    ).split(",")
    if m.strip()
]
RETRIES_PER_MODEL = 2  # extra attempts after the first, with a short wait
MAX_CHARS = 200_000  # safety cap on text sent to the model

SYSTEM_PROMPT = """You are an expert at reading scientific papers and summarizing them accurately.
Return ONLY a valid JSON object (no markdown fences, no commentary) with exactly these keys:

{
  "title": string,
  "authors": [string],
  "date": string,                 // publication/submission date as found in the paper, or "Not stated"
  "main_topic": string,           // 1-2 sentences: what the paper is about and the problem it addresses
  "abstract_key_points": [string],      // 3-5 concise bullets capturing the abstract
  "experiment_results": [string],       // key quantitative/qualitative results, include numbers and baselines where given
  "conclusion": [string],               // 2-4 bullets: main conclusions, limitations, future work
  "keywords": [string]
}

Rules:
- Use only information present in the paper. If something is missing, write "Not stated" (or an empty list).
- Do not invent numbers, authors, or dates.
- Keep each bullet to one or two sentences."""


# ---------- PDF + LLM ----------

def extract_pdf(path: str) -> tuple[str, dict]:
    doc = pymupdf.open(path)
    text = "\n\n".join(page.get_text("text") for page in doc)
    meta = doc.metadata or {}
    doc.close()
    return text, meta


def parse_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def llm_summarize(text: str, meta: dict) -> dict:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    hint = ""
    if meta:
        hint = (
            "PDF metadata (may be unreliable): "
            f"title={meta.get('title')!r}, author={meta.get('author')!r}, "
            f"created={meta.get('creationDate')!r}\n\n"
        )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{hint}PAPER TEXT:\n\n{text[:MAX_CHARS]}"},
    ]

    last_error = None
    for model in MODELS:
        for attempt in range(RETRIES_PER_MODEL + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    max_tokens=4000,
                    extra_body={"reasoning": {"enabled": False}},  # keep tokens for the JSON
                    messages=messages,
                )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("The model returned an empty response.")
                return parse_json(content)
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < RETRIES_PER_MODEL:
                    time.sleep(3 * (attempt + 1))  # 3s, then 6s
    raise RuntimeError(f"All models failed. Last error: {last_error}")


# ---------- Formatting ----------

def to_markdown(s: dict) -> str:
    def bullets(items):
        return "\n".join(f"- {i}" for i in items) if items else "- Not stated"

    return (
        f"# {s.get('title', 'Untitled')}\n\n"
        f"**Authors:** {', '.join(s.get('authors', [])) or 'Not stated'}  \n"
        f"**Date:** {s.get('date', 'Not stated')}\n\n"
        f"## Main Topic\n{s.get('main_topic', 'Not stated')}\n\n"
        f"## Abstract - Key Points\n{bullets(s.get('abstract_key_points'))}\n\n"
        f"## Experiment Results\n{bullets(s.get('experiment_results'))}\n\n"
        f"## Conclusion\n{bullets(s.get('conclusion'))}\n\n"
        f"**Keywords:** {', '.join(s.get('keywords', [])) or 'Not stated'}\n"
    )


def to_script(s: dict) -> str:
    """Natural-sounding narration for text-to-speech."""

    def sentences(items):
        return " ".join(i.rstrip(".") + "." for i in items) if items else "Not stated."

    authors = s.get("authors") or []
    if len(authors) > 3:
        author_line = ", ".join(authors[:3]) + f", and {len(authors) - 3} others"
    else:
        author_line = ", ".join(authors) or "unknown authors"

    return (
        f"Paper title: {s.get('title', 'Untitled')}. "
        f"By {author_line}. Date: {s.get('date', 'not stated')}. "
        f"Main topic. {s.get('main_topic', 'Not stated')} "
        f"From the abstract. {sentences(s.get('abstract_key_points'))} "
        f"Experiment results. {sentences(s.get('experiment_results'))} "
        f"Conclusion. {sentences(s.get('conclusion'))}"
    )


def make_audio(script: str) -> str:
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    out.close()
    gTTS(text=script, lang="en").save(out.name)
    return out.name


# ---------- App ----------

def run(pdf_path, pasted_text):
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise gr.Error("Set the OPENROUTER_API_KEY environment variable and restart the app.")

    text, meta = "", {}
    if pdf_path:
        text, meta = extract_pdf(pdf_path)
    elif pasted_text and pasted_text.strip():
        text = pasted_text
    if not text.strip():
        raise gr.Error("Upload a PDF (with selectable text) or paste the paper's text.")

    try:
        summary = llm_summarize(text, meta)
    except Exception as e:  # noqa: BLE001
        raise gr.Error(f"Summarization failed: {e}. Try again or pick another model.")

    try:
        audio = make_audio(to_script(summary))
    except Exception as e:  # noqa: BLE001
        gr.Warning(f"Audio generation failed ({e}); showing the text summary only.")
        audio = None

    return to_markdown(summary), audio


FONTS = [
    "Default",
    "Roboto",
    "Open Sans",
    "Lato",
    "Merriweather",
    "Lora",
    "Playfair Display",
    "Source Serif 4",
    "Inter",
    "Poppins",
    "Nunito",
    "Fira Sans",
    "Roboto Mono",
    "Atkinson Hyperlegible",
]


def font_css(name: str) -> str:
    """Return a <style> block that loads a Google Font and applies it to the summary."""
    if not name or name == "Default":
        return ""
    family = name.replace(" ", "+")
    return (
        "<style>"
        f"@import url('https://fonts.googleapis.com/css2?family={family}:wght@400;700&display=swap');"
        f"#summary-box, #summary-box * {{ font-family: '{name}', sans-serif !important; }}"
        "</style>"
    )


with gr.Blocks(title="Paper Summarizer") as demo:
    gr.Markdown("# 📄 Research Paper Summarizer\nGet a written and spoken summary of any paper.")
    with gr.Row():
        with gr.Column():
            pdf_in = gr.File(label="Research paper (PDF)", file_types=[".pdf"], type="filepath")
            text_in = gr.Textbox(label="...or paste the paper's text", lines=6)
            btn = gr.Button("Summarize", variant="primary")
        with gr.Column():
            audio_out = gr.Audio(label="Audio summary", type="filepath")
            font_dd = gr.Dropdown(
                choices=FONTS, value="Default", label="Summary font (Google Fonts)"
            )
            font_html = gr.HTML()
            md_out = gr.Markdown(elem_id="summary-box")
    btn.click(run, inputs=[pdf_in, text_in], outputs=[md_out, audio_out])
    font_dd.change(font_css, inputs=font_dd, outputs=font_html)

if __name__ == "__main__":
    demo.launch()
