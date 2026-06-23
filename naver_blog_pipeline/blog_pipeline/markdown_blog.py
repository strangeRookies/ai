from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .llm_client import LLMClient

FENCE_RE: Final = re.compile(r"^(```|~~~)")
HEADING_RE: Final = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
IMAGE_RE: Final = re.compile(r"!\[[^\]]*]\([^)]+\)")
LIST_RE: Final = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
TABLE_RE: Final = re.compile(r"^\s*\|.+\|\s*$")
INLINE_CODE_RE: Final = re.compile(r"(`[^`\n]+`)")
LINK_RE: Final = re.compile(r"(\[[^\]]+\]\([^)]+\))")


@dataclass(frozen=True, slots=True)
class BlogDraft:
    title: str
    markdown: str
    sections: tuple[str, ...]
    summary_items: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MarkdownBlock:
    text: str
    kind: str


def build_blog_draft(
    source_markdown: str,
    source_name: str = "note.md",
    llm_client: LLMClient | None = None,
    humanize: bool = True,
) -> BlogDraft:
    body = _strip_frontmatter(source_markdown)
    
    # Path A: Full Document generation via LLM
    if humanize and llm_client is not None and llm_client.is_enabled():
        full_post_markdown = _protect_and_generate_full(body, llm_client)
        return BlogDraft(
            title=source_name,
            markdown=full_post_markdown + "\n",
            sections=(),
            summary_items=(),
        )

    # Path B: Fallback programmatic layout
    blocks = _split_blocks(body)
    title = _extract_title(blocks, source_name)
    content_blocks = _remove_first_h1(blocks)
    sections = _section_titles(content_blocks)
    summary_items = _summary_items(content_blocks, sections, title)

    lines: list[str] = [
        "---",
        f'title: "{_escape_frontmatter(title)}"',
        "status: draft",
        "generated_by: blog_pipeline",
        "---",
        "",
        f"# {title}",
        "",
        "## 도입부",
        "",
        _protect_and_humanize(_intro_for(title, sections), llm_client, humanize),
        "",
    ]

    if len(sections) >= 3:
        lines.extend(["## 목차", ""])
        lines.extend(f"- {section}" for section in sections)
        lines.append("")

    lines.extend(["## 본문", ""])
    lines.extend(_render_content_blocks(content_blocks, llm_client, humanize))
    lines.extend(["", "## 핵심 정리", ""])
    lines.extend(f"- {item}" for item in summary_items)
    lines.extend(
        [
            "",
            "## 마무리",
            "",
            _protect_and_humanize(_closing_for(title), llm_client, humanize),
            "",
        ],
    )

    return BlogDraft(
        title=title,
        markdown=_collapse_blank_lines("\n".join(lines)).strip() + "\n",
        sections=sections,
        summary_items=summary_items,
    )


def write_blog_draft(
    input_path: Path,
    output_path: Path,
    llm_client: LLMClient | None = None,
    humanize: bool = True,
) -> BlogDraft:
    source = input_path.read_text(encoding="utf-8")
    draft = build_blog_draft(source, source_name=input_path.name, llm_client=llm_client, humanize=humanize)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(draft.markdown, encoding="utf-8")
    return draft


def _strip_frontmatter(markdown: str) -> str:
    normalized = markdown.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        return normalized
    end = normalized.find("\n---\n", 4)
    if end == -1:
        return normalized
    return normalized[end + 5 :]


def _split_blocks(markdown: str) -> tuple[MarkdownBlock, ...]:
    blocks: list[MarkdownBlock] = []
    current: list[str] = []
    in_fence = False
    fence_marker = ""

    for line in markdown.splitlines():
        fence_match = FENCE_RE.match(line)
        if fence_match is not None:
            marker = fence_match.group(1)
            if not in_fence:
                _flush_paragraph(current, blocks)
                current.append(line)
                in_fence = True
                fence_marker = marker
                continue
            current.append(line)
            if marker == fence_marker:
                blocks.append(MarkdownBlock("\n".join(current), "code"))
                current = []
                in_fence = False
                fence_marker = ""
            continue

        if in_fence:
            current.append(line)
            continue

        if line.strip() == "":
            _flush_paragraph(current, blocks)
            continue
        current.append(line)

    _flush_paragraph(current, blocks)
    return tuple(blocks)


def _flush_paragraph(current: list[str], blocks: list[MarkdownBlock]) -> None:
    if not current:
        return
    text = "\n".join(current).strip()
    kind = _block_kind(text)
    blocks.append(MarkdownBlock(text, kind))
    current.clear()


def _block_kind(text: str) -> str:
    first = text.splitlines()[0]
    if HEADING_RE.match(first) is not None:
        return "heading"
    if IMAGE_RE.search(text) is not None:
        return "image"
    if LIST_RE.match(first) is not None:
        return "list"
    if TABLE_RE.match(first) is not None:
        return "table"
    if first.startswith(">"):
        return "quote"
    return "paragraph"


def _extract_title(blocks: tuple[MarkdownBlock, ...], source_name: str) -> str:
    for block in blocks:
        match = HEADING_RE.match(block.text)
        if match is not None and len(match.group(1)) == 1:
            return _clean_heading(match.group(2))
    source_title = Path(source_name).stem.replace("_", " ").replace("-", " ").strip()
    return f"{source_title or '정리본'} 블로그 정리"


def _remove_first_h1(blocks: tuple[MarkdownBlock, ...]) -> tuple[MarkdownBlock, ...]:
    result: list[MarkdownBlock] = []
    removed = False
    for block in blocks:
        match = HEADING_RE.match(block.text)
        if not removed and match is not None and len(match.group(1)) == 1:
            removed = True
            continue
        result.append(block)
    return tuple(result)


def _section_titles(blocks: tuple[MarkdownBlock, ...]) -> tuple[str, ...]:
    titles: list[str] = []
    for block in blocks:
        match = HEADING_RE.match(block.text)
        if match is not None and len(match.group(1)) in {2, 3}:
            titles.append(_clean_heading(match.group(2)))
    return tuple(titles)


def _summary_items(
    blocks: tuple[MarkdownBlock, ...],
    sections: tuple[str, ...],
    title: str,
) -> tuple[str, ...]:
    if sections:
        return tuple(f"{section} 내용을 중심으로 확인합니다." for section in sections[:5])

    for block in blocks:
        if block.kind == "list":
            items = [_clean_list_item(line) for line in block.text.splitlines()]
            cleaned = tuple(item for item in items if item)
            if cleaned:
                return cleaned[:5]

    return (f"{title}의 핵심 흐름을 한 번에 파악할 수 있게 정리했습니다.",)


def _render_content_blocks(
    blocks: tuple[MarkdownBlock, ...],
    llm_client: LLMClient | None = None,
    humanize: bool = True,
) -> list[str]:
    rendered: list[str] = []
    for block in blocks:
        if block.kind == "heading":
            rendered.append(_shift_heading(block.text))
        elif block.kind == "paragraph":
            rendered.append(_protect_and_humanize(block.text, llm_client, humanize))
        else:
            rendered.append(block.text)
        rendered.append("")
    return rendered


def _protect_and_humanize(text: str, llm_client: LLMClient | None = None, humanize: bool = True) -> str:
    if not humanize:
        return text

    if llm_client is None or not llm_client.is_enabled():
        return humanize_paragraph(text)

    placeholders: list[tuple[str, str]] = []

    def repl_img(match: re.Match[str]) -> str:
        token = f"__IMG_{len(placeholders)}__"
        placeholders.append((token, match.group(0)))
        return token

    def repl_link(match: re.Match[str]) -> str:
        token = f"__LINK_{len(placeholders)}__"
        placeholders.append((token, match.group(0)))
        return token

    def repl_code(match: re.Match[str]) -> str:
        token = f"__CODE_{len(placeholders)}__"
        placeholders.append((token, match.group(0)))
        return token

    # 1. Protect images first since they contain ![...]
    processed = IMAGE_RE.sub(repl_img, text)
    # 2. Protect links
    processed = LINK_RE.sub(repl_link, processed)
    # 3. Protect inline code
    processed = INLINE_CODE_RE.sub(repl_code, processed)

    # Send to LLM
    humanized = llm_client.humanize(processed)

    # Check if all placeholders are present in the response
    missing = False
    for token, _ in placeholders:
        if token.lower() not in humanized.lower():
            missing = True
            break

    if missing:
        print(
            "[markdown_blog] Warning: LLM omitted or corrupted placeholders. "
            "Falling back to regex softening for this paragraph.",
            flush=True,
        )
        return humanize_paragraph(text)

    # Restore placeholders
    for token, original in placeholders:
        humanized = humanized.replace(token, original)

    return humanized


def _protect_and_generate_full(text: str, llm_client: LLMClient) -> str:
    placeholders: list[tuple[str, str]] = []

    def repl(match: re.Match[str], prefix: str) -> str:
        token = f"__{prefix}_{len(placeholders)}__"
        placeholders.append((token, match.group(0)))
        return token

    # Protect code fences
    processed = re.sub(r"(```.*?```|~~~.*?~~~)", lambda m: repl(m, "FENCE"), text, flags=re.DOTALL)
    processed = IMAGE_RE.sub(lambda m: repl(m, "IMG"), processed)
    processed = LINK_RE.sub(lambda m: repl(m, "LINK"), processed)
    processed = INLINE_CODE_RE.sub(lambda m: repl(m, "CODE"), processed)

    # Call LLM
    full_post = llm_client.generate_full_post(processed)

    # Restore placeholders
    for token, original in placeholders:
        full_post = full_post.replace(token, original)

    return full_post


def _shift_heading(text: str) -> str:
    match = HEADING_RE.match(text)
    if match is None:
        return text
    level = min(len(match.group(1)) + 1, 6)
    return f"{'#' * level} {_clean_heading(match.group(2))}"


def humanize_paragraph(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    normalized = " ".join(line for line in lines if line)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    
    # 1. AI 관용구 제거
    ai_phrases = [
        r"결론적으로,?\s*",
        r"시사하는 바가 크다\.?\s*",
        r"이에 대해 알아봅시다\.?\s*",
        r"요약하자면,?\s*",
        r"마지막으로,?\s*",
        r"첫째,?\s*",
        r"둘째,?\s*",
        r"셋째,?\s*",
    ]
    for phrase in ai_phrases:
        normalized = re.sub(phrase, "", normalized)
        
    # 2. 번역투 및 피동 표현 제거
    normalized = re.sub(r"(\S+)에 있어서\s+", r"\1에서 ", normalized)
    normalized = re.sub(r"(\S+)[을를] 통해\s+", r"\1(으)로 ", normalized)
    normalized = re.sub(r"(\S+)에 의해\s+", r"\1(으)로 ", normalized)
    normalized = re.sub(r"(\S+)되어지다", r"\1되다", normalized)
    normalized = re.sub(r"(\S+)되어집니다", r"\1됩니다", normalized)

    # 3. 어미 자연화
    replacements = {
        "한다.": "합니다.",
        "이다.": "입니다.",
        "된다.": "됩니다.",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
        
    return normalized.strip()


def _intro_for(title: str, sections: tuple[str, ...]) -> str:
    if sections:
        joined = ", ".join(sections[:3])
        return f"이 글에서는 {title}를 중심으로 {joined} 흐름을 차근차근 정리합니다."
    return f"이 글에서는 {title}의 핵심 내용을 블로그 독자가 읽기 쉽게 정리합니다."


def _closing_for(title: str) -> str:
    return (
        f"{title}를 적용할 때는 위 내용을 기준으로 먼저 흐름을 확인하고, "
        "코드나 명령어, 설정값은 원본 맥락에 맞게 다시 점검해 주세요."
    )


def _clean_heading(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip("#")).strip()


def _clean_list_item(line: str) -> str:
    return re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", line).strip()


def _escape_frontmatter(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _collapse_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text)
