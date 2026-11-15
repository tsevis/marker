"""Decide whether a PDF carries its own text, so OCR can be skipped safely.

OCR is the slowest stage of a conversion, and a PDF that already has a text
layer does not need it. Skipping it on a scan, though, produces an empty
document, so the GUI only offers the choice where a sample of the pages shows
real text.

The decision (`report_from_counts`) is kept apart from the file reading
(`inspect_text_layer`) so it can be reasoned about without a PDF.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# How many pages to look at. Spread over the whole document, because a book
# can have typeset front matter and a scanned body.
SAMPLE_PAGES = 40

# Fewer characters than this is a page number or a watermark, not the page.
MIN_CHARS_PER_PAGE = 50

# Share of sampled pages that must carry text. Below 1.0 so that an image-only
# cover or a few plates do not hide an otherwise fully typeset book.
SUGGEST_SHARE = 0.9


@dataclass(frozen=True)
class TextLayerReport:
    """What a sample of a PDF's pages says about its text layer."""

    pages: int
    checked: int
    with_text: int

    @property
    def suggest_skip_ocr(self) -> bool:
        if self.checked <= 0:
            return False
        return self.with_text / self.checked >= SUGGEST_SHARE

    def hint(self) -> str:
        """One line for the GUI, saying what was found and what it means."""
        if self.checked <= 0:
            return "Could not read any pages; OCR stays on."
        found = f"Text found on {self.with_text} of {self.checked} pages checked."
        if not self.suggest_skip_ocr:
            return f"{found} This PDF needs OCR."
        if self.with_text == self.checked:
            return f"{found} Skipping OCR is faster."
        return f"{found} Skipping OCR is faster; pages without text come out empty."


def sample_indices(page_count: int, limit: int = SAMPLE_PAGES) -> tuple[int, ...]:
    """Up to `limit` page indices, evenly spread from the first page to the last."""
    if page_count <= 0 or limit <= 0:
        return ()
    if page_count <= limit:
        return tuple(range(page_count))
    if limit == 1:
        return (0,)
    step = (page_count - 1) / (limit - 1)
    return tuple(round(i * step) for i in range(limit))


def report_from_counts(page_count: int, char_counts: Sequence[int]) -> TextLayerReport:
    """Build the report from the character count of each sampled page."""
    with_text = sum(1 for count in char_counts if count >= MIN_CHARS_PER_PAGE)
    return TextLayerReport(page_count, len(char_counts), with_text)


def inspect_text_layer(pdf: str) -> TextLayerReport:
    """Sample `pdf` and report on its text layer.

    Raises whatever PDFium raises for a file it cannot open; the caller decides
    what an unreadable file means for the user.
    """
    import pypdfium2

    document = pypdfium2.PdfDocument(pdf)
    try:
        page_count = len(document)
        counts = [
            _page_char_count(document, index) for index in sample_indices(page_count)
        ]
    finally:
        document.close()
    return report_from_counts(page_count, counts)


def _page_char_count(document, index: int) -> int:
    page = document[index]
    try:
        textpage = page.get_textpage()
        try:
            return textpage.count_chars()
        finally:
            textpage.close()
    finally:
        page.close()
