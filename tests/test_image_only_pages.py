"""A cover or plate with no text of its own must still reach the output."""

from types import SimpleNamespace

import pytest
from PIL import Image

from marker.builders.document import DocumentBuilder
from marker.builders.ocr import OcrBuilder
from marker.providers.pdf import PdfProvider
from marker.schema import BlockTypes
from marker.schema.labels import block_type_to_surya_label

pytestmark = pytest.mark.cpu

PAGE_PIXELS = (288, 216)  # a 4 x 3 inch page at 72 DPI


def image_pdf(tmp_path, *pages):
    path = tmp_path / "pages.pdf"
    pages[0].save(path, "PDF", resolution=72, save_all=True, append_images=pages[1:])
    return path


def cover():
    """Half dark, half light: a page that is clearly not blank."""
    image = Image.new("RGB", PAGE_PIXELS, "white")
    image.paste("navy", (0, 0, PAGE_PIXELS[0] // 2, PAGE_PIXELS[1]))
    return image


def bare_document(path, config=None):
    provider = PdfProvider(str(path), {"pdftext_workers": 1})
    document = DocumentBuilder(config).build_document(provider)
    for page in document.pages:
        page.children = []
        page.structure = []
    return document


def ocr_block(block_type, html, bbox):
    x0, y0, x1, y1 = bbox
    return SimpleNamespace(
        label=block_type_to_surya_label(block_type),
        html=html,
        polygon=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        error=False,
        skipped=False,
    )


def structure_types(document, page):
    return [block.block_type for block in page.structure_blocks(document)]


# --- OCR on: the model says where the image is --------------------------------


def test_an_image_the_ocr_model_locates_is_kept_as_a_picture(tmp_path):
    """Catches the cover reaching the Markdown as an empty `![]()`."""
    document = bare_document(image_pdf(tmp_path, cover()))
    page = document.pages[0]
    image = page.get_image(highres=False)
    result = SimpleNamespace(
        blocks=[
            ocr_block(BlockTypes.SectionHeader, "<img/>", (10, 10, 200, 150)),
            ocr_block(BlockTypes.Text, "<p>INTRODUCTION BY</p>", (10, 160, 200, 200)),
        ]
    )

    OcrBuilder(recognition_model=None).replace_page_structure(
        document, [page], [image], [result]
    )

    assert structure_types(document, page) == [BlockTypes.Picture, BlockTypes.Text]
    picture = page.structure_blocks(document)[0]
    assert picture.html is None


# --- OCR off: nothing reads the page, so keep the page itself -----------------


def test_with_ocr_off_a_textless_cover_becomes_a_full_page_picture(tmp_path):
    document = bare_document(image_pdf(tmp_path, cover(), cover()))
    first, second = document.pages

    DocumentBuilder({"disable_ocr": True}).add_image_only_page_pictures(document, [0])

    assert structure_types(document, first) == [BlockTypes.Picture]
    picture = first.structure_blocks(document)[0]
    assert picture.polygon.bbox == first.polygon.bbox
    assert structure_types(document, second) == []


def test_a_blank_textless_page_gets_no_picture(tmp_path):
    """Blank versos are textless too; they must not become white pictures."""
    blank = Image.new("RGB", PAGE_PIXELS, "white")
    document = bare_document(image_pdf(tmp_path, blank))

    DocumentBuilder({"disable_ocr": True}).add_image_only_page_pictures(document, [0])

    assert structure_types(document, document.pages[0]) == []


def test_a_partial_picture_on_such_a_page_gives_way_to_the_whole_page(tmp_path):
    """On the Bresson cover the OCR model located only the title lettering;
    exporting that crop instead of the cover is the bug this catches."""
    document = bare_document(image_pdf(tmp_path, cover()))
    page = document.pages[0]
    image = page.get_image(highres=False)
    result = SimpleNamespace(
        blocks=[
            ocr_block(BlockTypes.SectionHeader, "<img/>", (10, 10, 200, 150)),
            ocr_block(BlockTypes.Text, "<p>INTRODUCTION BY</p>", (10, 160, 200, 200)),
        ]
    )
    OcrBuilder(recognition_model=None).replace_page_structure(
        document, [page], [image], [result]
    )

    DocumentBuilder().add_image_only_page_pictures(document, [0])

    assert structure_types(document, page) == [BlockTypes.Picture, BlockTypes.Text]
    picture = page.structure_blocks(document)[0]
    assert picture.polygon.bbox == page.polygon.bbox


def pages_with_text(count, textless):
    document = SimpleNamespace(
        pages=[SimpleNamespace(page_id=i) for i in range(count)]
    )
    provider = SimpleNamespace(
        page_lines={i: ([] if i in textless else ["line"]) for i in range(count)}
    )
    return document, provider


def test_only_the_textless_pages_of_a_text_document_are_chosen():
    document, provider = pages_with_text(20, textless={0, 7})

    chosen = DocumentBuilder({"disable_ocr": True}).image_only_page_ids(
        document, provider
    )

    assert chosen == [0, 7]


def test_a_scanned_document_is_not_turned_into_page_pictures():
    document, provider = pages_with_text(20, textless=set(range(20)))

    chosen = DocumentBuilder({"disable_ocr": True}).image_only_page_ids(
        document, provider
    )

    assert chosen == []


def test_the_textless_pages_are_chosen_with_ocr_on_too():
    document, provider = pages_with_text(20, textless={0})

    assert DocumentBuilder().image_only_page_ids(document, provider) == [0]
