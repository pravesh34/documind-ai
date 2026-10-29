"""Text extraction and page segmentation for PDFs and pasted text."""

from __future__ import annotations

import io
import re

from app.core.errors import AppError
from app.domain import DocumentPage


def split_text_into_pages(text: str, *, max_page_chars: int = 3_000) -> list[DocumentPage]:
    """Split raw text into logical pages.

    Form feeds (PDF page breaks) are honoured. Otherwise blank-line separated
    blocks are grouped into roughly ``max_page_chars`` sized pages.
    """
    normalised = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalised:
        return []

    form_feed_pages = [page.strip() for page in normalised.split("\f") if page.strip()]
    if len(form_feed_pages) > 1:
        return [
            DocumentPage(page_number=index + 1, text=page, char_count=len(page))
            for index, page in enumerate(form_feed_pages)
        ]

    blocks = [block.strip() for block in re.split(r"\n{2,}", normalised) if block.strip()]
    if len(blocks) <= 1:
        return [DocumentPage(page_number=1, text=normalised, char_count=len(normalised))]

    pages: list[str] = []
    current = ""
    for block in blocks:
        if current and len(current) + len(block) + 2 > max_page_chars:
            pages.append(current)
            current = block
        else:
            current = f"{current}\n\n{block}" if current else block
    if current:
        pages.append(current)

    return [
        DocumentPage(page_number=index + 1, text=page, char_count=len(page))
        for index, page in enumerate(pages)
    ]


def extract_pdf_pages(data: bytes) -> list[DocumentPage]:
    try:
        from pypdf import PdfReader
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise AppError(
            500,
            "PDF_SUPPORT_UNAVAILABLE",
            "PDF extraction support is not installed.",
            expose=False,
        ) from error

    try:
        reader = PdfReader(io.BytesIO(data))
        pages: list[DocumentPage] = []
        for index, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            pages.append(
                DocumentPage(
                    page_number=index + 1,
                    text=text,
                    char_count=len(text),
                )
            )
    except Exception as error:  # noqa: BLE001 - corrupt upload
        raise AppError(
            422,
            "PDF_PARSE_FAILED",
            "The uploaded PDF could not be parsed. It may be encrypted or corrupt.",
        ) from error

    if not any(page.text.strip() for page in pages):
        raise AppError(
            422,
            "PDF_HAS_NO_TEXT",
            "The PDF contains no extractable text (it may be a scanned image). "
            "Run OCR first or paste the text content instead.",
        )

    return pages if pages else [DocumentPage(page_number=1, text="", char_count=0)]


def detect_mime_type(filename: str, content_type: str | None) -> str:
    if content_type and content_type not in {"application/octet-stream", ""}:
        return content_type
    lowered = filename.lower()
    if lowered.endswith(".pdf"):
        return "application/pdf"
    if lowered.endswith((".md", ".markdown")):
        return "text/markdown"
    if lowered.endswith((".txt", ".text")):
        return "text/plain"
    return "application/octet-stream"
