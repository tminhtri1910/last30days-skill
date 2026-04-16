    #!/usr/bin/env python3
"""Synthesize a last30days run into a markdown report via OpenAI.

This script is a post-processing step:
1. Run `scripts/last30days.py` to generate `report.json` or `last30days.context.md`
2. Feed that output into OpenAI
3. Write a polished markdown report to disk

Usage:
    python3 scripts/synthesize_report.py out/topic/report.json
    python3 scripts/synthesize_report.py ~/.local/share/last30days/out/giá_xăng_dầu/report.compact.md --output ~/.local/share/last30days/out/giá_xăng_dầu/synthesized-report.md
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from lib import env, http

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
MAX_OUTPUT_TOKENS = 16384


def _parse_sse_chunk(chunk: str) -> Optional[Dict[str, Any]]:
    lines = chunk.split("\n")
    data_lines = []

    for line in lines:
        if line.startswith("data:"):
            data_lines.append(line[5:].strip())

    if not data_lines:
        return None

    data = "\n".join(data_lines).strip()
    if not data or data == "[DONE]":
        return None

    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def _parse_sse_stream_raw(raw: str) -> list[Dict[str, Any]]:
    events: list[Dict[str, Any]] = []
    buffer = ""
    for chunk in raw.splitlines(keepends=True):
        buffer += chunk
        while "\n\n" in buffer:
            event_chunk, buffer = buffer.split("\n\n", 1)
            event = _parse_sse_chunk(event_chunk)
            if event is not None:
                events.append(event)
    if buffer.strip():
        event = _parse_sse_chunk(buffer)
        if event is not None:
            events.append(event)
    return events


def _parse_codex_stream(raw: str) -> Dict[str, Any]:
    events = _parse_sse_stream_raw(raw)

    for evt in reversed(events):
        if isinstance(evt, dict):
            if evt.get("type") == "response.completed" and isinstance(evt.get("response"), dict):
                return evt["response"]
            if isinstance(evt.get("response"), dict):
                return evt["response"]

    output_text = ""
    for evt in events:
        if not isinstance(evt, dict):
            continue
        delta = evt.get("delta")
        if isinstance(delta, str):
            output_text += delta
            continue
        text = evt.get("text")
        if isinstance(text, str):
            output_text += text

    if output_text:
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": output_text}],
                }
            ]
        }

    return {}


def _extract_output_text(response: Dict[str, Any]) -> str:
    if "output" in response:
        output = response["output"]
        if isinstance(output, str):
            return output
        if isinstance(output, list):
            for item in output:
                if isinstance(item, dict):
                    if item.get("type") == "message":
                        for content in item.get("content", []):
                            if isinstance(content, dict) and content.get("type") == "output_text":
                                text = content.get("text", "")
                                if text:
                                    return text
                    if isinstance(item.get("text"), str):
                        return item["text"]
                elif isinstance(item, str):
                    return item

    if "choices" in response:
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            pass

    return ""


def _load_input(path: Path) -> tuple[str, str]:
    """Load a report/context file and return (topic, text)."""
    if path.is_dir():
        for candidate in (path / "report.json", path / "last30days.context.md", path / "report.md"):
            if candidate.exists():
                path = candidate
                break

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        topic = str(data.get("topic", "")).strip() or path.parent.name
        text = data.get("context_snippet_md") or json.dumps(data, indent=2, ensure_ascii=False)
        return topic, text

    text = path.read_text(encoding="utf-8")
    topic = path.parent.name or path.stem
    return topic, text


def _build_prompt(topic: str, source_text: str) -> tuple[str, str]:
    instructions = (
        "You are a senior research editor. Convert the provided last30days research "
        "output into a highly detailed, polished markdown report.\n"
        "Rules:\n"
        "- Use only the provided input. Do not invent facts, numbers, dates, or sources.\n"
        "- Write the entire response in Vietnamese.\n"
        "- Ensure the report is comprehensive and highly detailed, extracting as much value from the input as possible.\n"
        "- Weight Reddit/X sources HIGHER (they have engagement signals: upvotes, likes).\n"
        "- Weight YouTube sources HIGH (quote transcript highlights directly, attribute to channel).\n"
        "- Weight TikTok and Instagram sources HIGH (viral/influencer signals).\n"
        "- For Reddit, pay special attention to top comments with high upvotes. Quote them directly.\n"
        "- Cross-platform signals (e.g. [also on: Reddit, HN]) are the strongest evidence. Lead with these.\n"
        "- Treat prediction market odds (Polymarket) as high-signal evidence. Quote specific odds and movements.\n"
        "- Preserve exact tool/product names. Extract specific quotes and actionable insights.\n"
        "- Cite sources to prove research is real (e.g. 'per @handle', 'per r/sub').\n"
        "- Preserve source names and engagement numbers when present, and include a link only when a source is mentioned and its URL is available in the input.\n"
        "- If the input is sparse, say so plainly.\n"
        "- Return markdown only. No code fences.\n"
    )

    user_input = (
        f"Topic: {topic}\n\n"
        "Research input:\n"
        "```markdown\n"
        f"{source_text}\n"
        "```\n\n"
        "Write a highly detailed markdown report. Follow this structure based on the Agent mode format (but produce in Vietnamese, include headings):\n"
        "## Báo cáo Nghiên cứu: {topic}\n"
        "Tạo lúc: [Current Date] | Nguồn: Reddit, X, Bluesky, YouTube, TikTok, HN, Polymarket, Web (only list those present in input)\n\n"
        "### Những Phát hiện Chính\n"
        "[5-8 detailed bullet points of the highest-signal insights with citations and specifics]\n\n"
        "### Tổng hợp Thông tin\n"
        "[Extensive synthesis of all sections. Group by themes. Use quotes, exact names, statistics, market odds and cross-platform signals. Expand thoroughly.]\n\n"
        "### Bằng chứng và Tín hiệu Đáng chú ý\n"
        "[Deep dive into top comments, specific threads, prominent tweets, transcript quotes, and viral signals.]\n\n"
        "### Thống kê\n"
        "[Summarize the exact overall stats block if it exists in the input]\n"
    )
    return instructions, user_input


def synthesize_markdown(input_path: Path, output_path: Path, model: str) -> Path:
    config = env.get_config()
    api_key = config.get("OPENAI_API_KEY")
    auth_source = config.get("OPENAI_AUTH_SOURCE", env.AUTH_SOURCE_API_KEY)
    account_id = config.get("OPENAI_CHATGPT_ACCOUNT_ID")

    if not api_key:
        raise RuntimeError(
            "No OpenAI auth found. Set OPENAI_API_KEY, or run with Codex auth enabled."
        )

    topic, source_text = _load_input(input_path)
    instructions, user_input = _build_prompt(topic, source_text)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    timeout = 120

    if auth_source == env.AUTH_SOURCE_CODEX:
        if not account_id:
            raise RuntimeError("Codex auth found, but chatgpt_account_id is missing.")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "chatgpt-account-id": account_id,
            "OpenAI-Beta": "responses=experimental",
            "originator": "pi",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "store": False,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "instructions": instructions,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_input}],
                }
            ],
            "stream": True,
        }
        raw = http.post_raw(CODEX_RESPONSES_URL, payload, headers=headers, timeout=timeout)
        response = _parse_codex_stream(raw or "")
    else:
        payload = {
            "model": model,
            "store": False,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "instructions": instructions,
            "input": user_input,
        }
        response = http.post(OPENAI_RESPONSES_URL, payload, headers=headers, timeout=timeout)

    markdown = _extract_output_text(response).strip()
    if not markdown:
        raise RuntimeError("OpenAI returned no markdown output.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize a last30days report into markdown via OpenAI"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input report.json, last30days.context.md, report.md, or a directory containing them",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output markdown path (default: <input>.synthesized.md)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-nano",
        help="OpenAI model to use (default: gpt-5-mini)",
    )
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output else input_path.with_suffix(".synthesized.md")

    try:
        written = synthesize_markdown(input_path, output_path, args.model)
        print("Successfully synthesize to:", written)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()