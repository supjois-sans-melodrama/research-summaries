"""
Research Paper Summarizer (Gradio, with audio)
----------------------------------------------
Upload a research paper (PDF) or paste its text. You get a structured summary
(title, authors, date, main topic, abstract, experiment results, conclusion)
plus an audio version read aloud.

Run:
    pip install -r requirements.txt
    export OPENROUTER_API_KEY="your-key"     # or put it in a .env file
    python app.py
"""

import html
import json
import logging
import os
import re
import tempfile
import time

import gradio as gr
import openai
import pymupdf
from gtts import gTTS
from openai import OpenAI

log = logging.getLogger("summarizer")
logging.basicConfig(level=logging.INFO)

try:  # optional: load OPENROUTER_API_KEY from a local .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Comma-separated list, tried in order. Override with SUMMARIZER_MODEL="a,b,c".
# Free models change over time; see https://openrouter.ai/models?max_price=0
MODELS = [
    m.strip()
    for m in os.environ.get(
        "SUMMARIZER_MODEL", "inclusionai/ling-3.1-flash,apodex/apodex-1.1-mini:free"
    ).split(",")
    if m.strip()
]
RETRIES_PER_MODEL = 2  # extra attempts after the first, with a short wait
MAX_CHARS = 200_000  # safety cap on text sent to the model
MAX_FILE_MB = 25  # reject bigger PDFs before parsing them
MAX_PAGES = 150  # reject PDFs with more pages than this
REQUEST_TIMEOUT_S = 90  # per model call, so a hung provider cannot block a worker

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

class UserFacingError(Exception):
    """An error whose message is safe to show to the person using the app."""


def extract_pdf(path: str) -> tuple[str, dict]:
    if os.path.getsize(path) > MAX_FILE_MB * 1024 * 1024:
        raise UserFacingError(f"That PDF is larger than {MAX_FILE_MB} MB. Please upload a smaller file.")
    try:
        doc = pymupdf.open(path)
    except Exception as e:  # corrupt or not really a PDF
        log.warning("Could not open PDF: %s", e)
        raise UserFacingError("That file could not be read as a PDF.") from e
    try:
        if doc.needs_pass:
            raise UserFacingError("That PDF is password-protected. Please upload an unlocked copy.")
        if doc.page_count > MAX_PAGES:
            raise UserFacingError(f"That PDF has more than {MAX_PAGES} pages. Please upload a shorter paper.")
        text = "\n\n".join(page.get_text("text") for page in doc)
        meta = doc.metadata or {}
    finally:
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
        timeout=REQUEST_TIMEOUT_S,
        max_retries=0,  # we do our own retries below; the SDK default would multiply them
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

    # Worth retrying on the same model: busy/slow provider, or the model's output was unusable.
    retryable = (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
        ValueError,  # includes json.JSONDecodeError and our empty/invalid-output checks
    )

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
                summary = parse_json(content)
                if not isinstance(summary, dict):
                    raise ValueError("The model did not return a JSON object.")
                return summary
            except openai.AuthenticationError as e:
                # Wrong or revoked key: retrying or switching models cannot help.
                log.error("OpenRouter rejected the API key: %s", e)
                raise UserFacingError("The server's API key was rejected. The app owner needs to update it.") from e
            except retryable as e:
                last_error = e
                log.warning("model=%s attempt=%d failed: %r", model, attempt + 1, e)
                if attempt < RETRIES_PER_MODEL:
                    time.sleep(3 * (attempt + 1))  # 3s, then 6s
            except Exception as e:  # noqa: BLE001
                # e.g. model removed (404), no credits (402), bad request (400): try the next model now.
                last_error = e
                log.warning("model=%s unusable, moving on: %r", model, e)
                break
    log.error("All models failed. Last error: %r", last_error)
    raise UserFacingError("The summarizer is busy or unavailable right now. Please try again in a minute.")


# ---------- Formatting ----------

def _list(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def to_markdown(s: dict) -> str:
    def bullets(items):
        items = _list(items)
        return "\n".join(f"- {i}" for i in items) if items else "- Not stated"

    return (
        f"# {s.get('title') or 'Untitled'}\n\n"
        f"**Authors:** {', '.join(_list(s.get('authors'))) or 'Not stated'}  \n"
        f"**Date:** {s.get('date') or 'Not stated'}\n\n"
        f"## Main Topic\n{s.get('main_topic') or 'Not stated'}\n\n"
        f"## Abstract - Key Points\n{bullets(s.get('abstract_key_points'))}\n\n"
        f"## Experiment Results\n{bullets(s.get('experiment_results'))}\n\n"
        f"## Conclusion\n{bullets(s.get('conclusion'))}\n\n"
        f"**Keywords:** {', '.join(_list(s.get('keywords'))) or 'Not stated'}\n"
    )


def to_script(s: dict) -> str:
    """Natural-sounding narration for text-to-speech."""

    def sentences(items):
        items = _list(items)
        return " ".join(i.rstrip(".") + "." for i in items) if items else "Not stated."

    authors = _list(s.get("authors"))
    if len(authors) > 3:
        author_line = ", ".join(authors[:3]) + f", and {len(authors) - 3} others"
    else:
        author_line = ", ".join(authors) or "unknown authors"

    return (
        f"Paper title: {s.get('title') or 'Untitled'}. "
        f"By {author_line}. Date: {s.get('date') or 'not stated'}. "
        f"Main topic. {s.get('main_topic') or 'Not stated'} "
        f"From the abstract. {sentences(s.get('abstract_key_points'))} "
        f"Experiment results. {sentences(s.get('experiment_results'))} "
        f"Conclusion. {sentences(s.get('conclusion'))}"
    )


def make_audio(script: str) -> str:
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    out.close()
    gTTS(text=script, lang="en").save(out.name)
    return out.name


def save_markdown(s: dict) -> str:
    folder = tempfile.mkdtemp()
    path = os.path.join(folder, "paper_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_markdown(s))
    return path


# ---------- HTML rendering ----------

CSS = """
<style>
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
.app-header { padding: 6px 2px 14px; }
.app-header h1 { font-size: 1.5rem; font-weight: 700; margin: 0; letter-spacing: -0.01em; }
.app-header p { margin: 4px 0 0; opacity: .7; font-size: .95rem; }

.sum-wrap { display: flex; flex-direction: column; gap: 14px; }
.sum-head, .card {
  border: 1px solid var(--border-color-primary, #e5e7eb);
  border-radius: 12px;
  background: var(--background-fill-secondary, #fafafa);
}
.sum-head { padding: 18px 22px; }
.sum-title { font-size: 1.4rem; font-weight: 700; line-height: 1.3; margin: 0 0 10px; }
.sum-meta { display: flex; flex-wrap: wrap; gap: 6px 22px; font-size: .92rem; }
.sum-meta .label { opacity: .6; margin-right: 6px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
.chip {
  padding: 2px 10px; border-radius: 999px; font-size: .78rem;
  background: rgba(99, 102, 241, .12); color: #6366f1; border: 1px solid rgba(99, 102, 241, .25);
}
.sum-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.card { padding: 16px 20px; }
.card.span2 { grid-column: 1 / -1; }
.card h3 {
  margin: 0 0 8px; font-size: .72rem; font-weight: 700;
  letter-spacing: .09em; text-transform: uppercase; color: #6366f1;
}
.card p { margin: 0; line-height: 1.6; }
.card ul { margin: 0; padding-left: 18px; }
.card li { margin: 5px 0; line-height: 1.55; }
.empty {
  padding: 60px 24px; text-align: center; opacity: .6;
  border: 1px dashed var(--border-color-primary, #d1d5db); border-radius: 12px;
}
.app-header { display: flex; align-items: center; gap: 14px; padding: 4px 2px 16px;
  margin-bottom: 14px; border-bottom: 1px solid var(--border-color-primary, #e5e7eb); }
.logo { width: 40px; height: 40px; border-radius: 11px; background: #6366f1; flex: none;
  display: flex; align-items: center; justify-content: center; }
.empty-title { font-size: 1.05rem; font-weight: 600; }
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 22px; text-align: left; }
.step {
  padding: 14px 16px; border-radius: 10px;
  border: 1px solid var(--border-color-primary, #e5e7eb);
  background: var(--background-fill-secondary, #fafafa);
}
.step b {
  display: inline-flex; align-items: center; justify-content: center;
  width: 22px; height: 22px; margin-bottom: 8px; border-radius: 50%;
  background: #6366f1; color: #fff; font-size: .75rem;
}
.step p { margin: 0; font-size: .88rem; line-height: 1.45; }
@media (max-width: 860px) { .sum-grid, .steps { grid-template-columns: 1fr; } }
</style>
"""

HEADER = (
    CSS
    + '<div class="app-header"><div class="logo">'
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/>'
    '<path d="M14 3v5h5M9 13h6M9 17h4"/></svg></div>'
    "<div><h1>Research Paper Summarizer</h1>"
    "<p>Upload a paper to get a structured summary and an audio version.</p></div></div>"
)

EMPTY = (
    '<div class="empty"><div class="empty-title">Your summary will appear here</div>'
    '<div class="steps">'
    '<div class="step"><b>1</b><p>Upload a research paper as a PDF, or paste its text.</p></div>'
    '<div class="step"><b>2</b><p>Click Summarize to extract the key points.</p></div>'
    '<div class="step"><b>3</b><p>Read the cards or listen to the audio version.</p></div>'
    "</div></div>"
)


def _bullets(items) -> str:
    items = _list(items) or ["Not stated"]
    return "<ul>" + "".join(f"<li>{html.escape(i)}</li>" for i in items) + "</ul>"


def render_html(s: dict) -> str:
    esc = lambda v: html.escape(str(v or "Not stated"))  # noqa: E731
    authors = ", ".join(_list(s.get("authors"))) or "Not stated"
    chips = "".join(f'<span class="chip">{html.escape(k)}</span>' for k in _list(s.get("keywords")))

    return f"""
<div class="sum-wrap">
  <div class="sum-head">
    <h2 class="sum-title">{esc(s.get("title"))}</h2>
    <div class="sum-meta">
      <span><span class="label">Authors</span>{html.escape(authors)}</span>
      <span><span class="label">Date</span>{esc(s.get("date"))}</span>
    </div>
    {f'<div class="chips">{chips}</div>' if chips else ''}
  </div>
  <div class="sum-grid">
    <div class="card span2"><h3>Main topic</h3><p>{esc(s.get("main_topic"))}</p></div>
    <div class="card"><h3>Abstract</h3>{_bullets(s.get("abstract_key_points"))}</div>
    <div class="card"><h3>Experiment results</h3>{_bullets(s.get("experiment_results"))}</div>
    <div class="card span2"><h3>Conclusion</h3>{_bullets(s.get("conclusion"))}</div>
  </div>
</div>
"""


# ---------- Fonts ----------

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
    # Only allow names from our own list: the value is placed inside a <style> block,
    # so an arbitrary string from the browser could inject CSS or markup.
    if not name or name == "Default" or name not in FONTS:
        return ""
    family = name.replace(" ", "+")
    return (
        "<style>"
        f"@import url('https://fonts.googleapis.com/css2?family={family}:wght@400;700&display=swap');"
        f"#summary-box, #summary-box * {{ font-family: '{name}', sans-serif !important; }}"
        "</style>"
    )


# ---------- App ----------

def run(pdf_path, pasted_text):
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise gr.Error("Set the OPENROUTER_API_KEY environment variable and restart the app.")

    text, meta = "", {}
    try:
        if pdf_path:
            text, meta = extract_pdf(pdf_path)
        elif pasted_text and pasted_text.strip():
            text = pasted_text[:MAX_CHARS]
        if not text.strip():
            raise UserFacingError("Upload a PDF (with selectable text) or paste the paper's text.")
        summary = llm_summarize(text, meta)
    except UserFacingError as e:
        raise gr.Error(str(e))
    except Exception:  # noqa: BLE001
        log.exception("Unexpected error while summarizing")
        raise gr.Error("Something went wrong while summarizing. Please try again.")

    try:
        audio = make_audio(to_script(summary))
    except Exception as e:  # noqa: BLE001
        log.warning("Audio generation failed: %r", e)
        gr.Warning("Audio generation failed; showing the text summary only.")
        audio = None

    return render_html(summary), audio, save_markdown(summary)


# delete_cache=(check every N seconds, delete files older than N seconds): stops the
# generated MP3/Markdown files from piling up on a small free container.
with gr.Blocks(title="Paper Summarizer", delete_cache=(3600, 3600)) as demo:
    gr.HTML(HEADER)
    with gr.Row(equal_height=False):
        # Left: inputs and tools
        with gr.Column(scale=1, min_width=300):
            pdf_in = gr.File(label="Research paper (PDF)", file_types=[".pdf"], type="filepath")
            with gr.Accordion("Or paste text instead", open=False):
                text_in = gr.Textbox(show_label=False, lines=6, placeholder="Paste the paper's text")
            btn = gr.Button("Summarize", variant="primary", size="lg")
            audio_out = gr.Audio(label="Audio summary", type="filepath")
            with gr.Accordion("Display & export", open=False):
                font_dd = gr.Dropdown(choices=FONTS, value="Default", label="Summary font (Google Fonts)")
                md_file = gr.File(label="Download summary (.md)", interactive=False)
            font_html = gr.HTML()
        # Right: the summary
        with gr.Column(scale=3):
            summary_out = gr.HTML(EMPTY, elem_id="summary-box")

    btn.click(run, inputs=[pdf_in, text_in], outputs=[summary_out, audio_out, md_file])
    font_dd.change(font_css, inputs=font_dd, outputs=font_html)

if __name__ == "__main__":
    # Limit how many requests can wait in line, so a flood cannot pile up work (and API spend).
    demo.queue(max_size=20, default_concurrency_limit=2)

    # Optional password gate for a public deployment. Set APP_PASSWORD (and optionally
    # APP_USER) as environment variables; leave unset for an open app.
    password = os.environ.get("APP_PASSWORD")
    auth = (os.environ.get("APP_USER", "user"), password) if password else None
    demo.launch(auth=auth)
