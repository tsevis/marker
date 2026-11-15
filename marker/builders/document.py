from typing import Annotated, List, Sequence

from marker.builders import BaseBuilder
from marker.builders.layout import LayoutBuilder
from marker.builders.line import LineBuilder
from marker.builders.ocr import OcrBuilder
from marker.providers.pdf import PdfProvider
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.polygon import PolygonBox
from marker.schema.registry import get_block_class


class DocumentBuilder(BaseBuilder):
    """
    Constructs a Document given a PdfProvider, LayoutBuilder, and OcrBuilder.
    """

    lowres_image_dpi: Annotated[
        int,
        "DPI setting for low-resolution page images used for layout detection.",
    ] = 96
    highres_image_dpi: Annotated[
        int,
        "DPI setting for high-resolution page images used for OCR.",
    ] = 192
    image_export_dpi: Annotated[
        int,
        "Resolution for exported pictures, independent of OCR. 0 uses the existing extraction mode.",
    ] = 0
    disable_ocr: Annotated[
        bool,
        "Disable OCR processing.",
    ] = False

    keep_image_only_pages: Annotated[
        bool,
        "Keep a page that has no text layer (a cover, a plate) as a full-page",
        "picture, in a document whose other pages do have one.",
    ] = True
    image_only_page_max_share: Annotated[
        float,
        "Largest share of textless pages for which they are kept as pictures.",
        "Above it the document is a scan, not a text PDF with a few plates.",
    ] = 0.25
    blank_page_tone_range: Annotated[
        int,
        "A page whose darkest and lightest pixels differ by less than this",
        "(0-255) is blank and is not kept as a picture.",
    ] = 16
    image_block_types: Annotated[
        Sequence[BlockTypes],
        "Partial pictures that a full-page picture replaces on such a page.",
    ] = (
        BlockTypes.Picture,
        BlockTypes.Figure,
    )

    # Block types whose default-path processors crop the high-res image:
    # TableProcessor (Table/TableOfContents/Form) and EquationProcessor
    # (Equation/ChemicalBlock). Figures/Pictures/Diagrams are only consumed by
    # opt-in LLM processors, which are served by the per-page lazy loader, so
    # they are deliberately excluded here (a figure-only page skips the render).
    highres_block_types: Annotated[
        Sequence[BlockTypes],
        "Block types whose presence on a page requires the high-res image.",
    ] = (
        BlockTypes.Table,
        BlockTypes.Form,
        BlockTypes.TableOfContents,
        BlockTypes.Equation,
        BlockTypes.ChemicalBlock,
    )

    def __call__(
        self,
        provider: PdfProvider,
        layout_builder: LayoutBuilder,
        line_builder: LineBuilder,
        ocr_builder: OcrBuilder,
    ):
        document = self.build_document(provider)
        layout_builder(document, provider)
        line_builder(document, provider)
        # Now that layout + the OCR decision are known, render high-res only for
        # the pages that actually need it (one batched pass). Anything missed
        # (e.g. opt-in LLM processors touching a clean page) falls back to the
        # per-page lazy loader set in build_document.
        self.render_highres(document, provider)
        if not self.disable_ocr:
            ocr_builder(document, provider)
        self.add_image_only_page_pictures(
            document, self.image_only_page_ids(document, provider)
        )
        return document

    def image_only_page_ids(self, document: Document, provider: PdfProvider) -> List[int]:
        """Pages to keep as full-page pictures: textless pages of a text PDF.

        Applies with OCR on as well: OCR reads the lettering on a cover, but
        the picture it locates there can be a fragment (only the title box).
        """
        if not self.keep_image_only_pages:
            return []
        pages = document.pages
        textless = [
            page.page_id for page in pages if not provider.page_lines.get(page.page_id)
        ]
        if not pages or len(textless) / len(pages) > self.image_only_page_max_share:
            return []
        return textless

    def add_image_only_page_pictures(self, document: Document, page_ids: Sequence[int]):
        """Put a full-page Picture at the top of each listed page.

        Blank pages are skipped. Partial pictures already on the page are
        dropped from its structure, since the whole page now covers them.
        """
        wanted = set(page_ids)
        picture_cls = get_block_class(BlockTypes.Picture)
        for page in document.pages:
            if page.page_id not in wanted or self.page_is_blank(page):
                continue
            kept = []
            for block in page.structure_blocks(document):
                if block.block_type in self.image_block_types:
                    block.removed = True
                else:
                    kept.append(block.id)
            picture = page.add_block(
                picture_cls, PolygonBox.from_bbox(page.polygon.bbox)
            )
            page.structure = [picture.id, *kept]

    def page_is_blank(self, page: PageGroup) -> bool:
        darkest, lightest = page.get_image(highres=False).convert("L").getextrema()
        return lightest - darkest < self.blank_page_tone_range

    def page_needs_highres(self, page: PageGroup) -> bool:
        if page.text_extraction_method == "surya":
            return True
        highres_types = set(self.highres_block_types)
        return any(block.block_type in highres_types for block in (page.children or []))

    def render_highres(self, document: Document, provider: PdfProvider):
        needed = [
            page
            for page in document.pages
            if page.highres_image is None and self.page_needs_highres(page)
        ]
        if not needed:
            return
        images = provider.get_images(
            [page.page_id for page in needed], self.highres_image_dpi
        )
        for page, image in zip(needed, images):
            page.highres_image = image

    def build_document(self, provider: PdfProvider):
        PageGroupClass: PageGroup = get_block_class(BlockTypes.Page)
        lowres_images = provider.get_images(provider.page_range, self.lowres_image_dpi)

        def make_loader(prov: PdfProvider):
            # Non-persistent: get_images opens/renders/closes per call.
            return lambda page_id: prov.get_images([page_id], self.highres_image_dpi)[0]

        loader = make_loader(provider)
        initial_pages = []
        for i, p in enumerate(provider.page_range):
            page = PageGroupClass(
                page_id=p,
                lowres_image=lowres_images[i],
                highres_image=None,
                polygon=provider.get_page_bbox(p),
                refs=provider.get_page_refs(p),
                pdftext_page=getattr(provider, "raw_pdftext_pages", {}).get(p),
            )
            page._highres_loader = loader
            if self.image_export_dpi:
                def export_loader(page_id, page=page, prov=provider):
                    if self.image_export_dpi == self.highres_image_dpi:
                        return page.get_image(highres=True)
                    if self.image_export_dpi == self.lowres_image_dpi:
                        return page.get_image(highres=False)
                    return prov.get_images([page_id], self.image_export_dpi)[0]

                page._export_loader = export_loader
            initial_pages.append(page)
        DocumentClass: Document = get_block_class(BlockTypes.Document)
        return DocumentClass(filepath=provider.filepath, pages=initial_pages)
