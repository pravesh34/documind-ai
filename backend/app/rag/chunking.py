"""Page-aware chunking built on LangChain text splitters."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings
from app.core.utils import token_count
from app.domain import ChunkRecord, DocumentPage

SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    "! ",
    "? ",
    "; ",
    ", ",
    " ",
    "",
]


@dataclass(slots=True)
class ChunkDraft:
    text: str
    page_number: int
    chunk_index: int
    start_char: int
    end_char: int


def build_splitter(settings: Settings) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=SEPARATORS,
        keep_separator=True,
        add_start_index=True,
        strip_whitespace=True,
        length_function=len,
    )


def split_pages(pages: list[DocumentPage], settings: Settings) -> list[ChunkDraft]:
    splitter = build_splitter(settings)
    drafts: list[ChunkDraft] = []
    index = 0

    for page in pages:
        text = page.text.strip()
        if not text:
            continue

        documents = splitter.create_documents([text])
        for document in documents:
            chunk_text = document.page_content.strip()
            if not chunk_text:
                continue

            start = int(document.metadata.get("start_index", 0) or 0)
            if text[start : start + len(chunk_text)] != chunk_text:
                # Fallback if whitespace stripping shifted the offset.
                located = text.find(chunk_text)
                start = located if located >= 0 else start

            drafts.append(
                ChunkDraft(
                    text=chunk_text,
                    page_number=page.page_number,
                    chunk_index=index,
                    start_char=start,
                    end_char=start + len(chunk_text),
                )
            )
            index += 1

    return drafts


def to_chunk_records(
    drafts: list[ChunkDraft],
    *,
    tenant_id: str,
    document_id: str,
    document_title: str,
) -> list[ChunkRecord]:
    return [
        ChunkRecord(
            chunk_id=f"{document_id}:{draft.chunk_index}",
            tenant_id=tenant_id,
            document_id=document_id,
            document_title=document_title,
            page_number=draft.page_number,
            chunk_index=draft.chunk_index,
            text=draft.text,
            token_count=token_count(draft.text),
            start_char=draft.start_char,
            end_char=draft.end_char,
        )
        for draft in drafts
    ]
