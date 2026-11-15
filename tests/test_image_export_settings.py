import io
import json
import queue
import threading
from types import SimpleNamespace

import pytest
from PIL import Image

import convert_gui
import marker_worker
from marker.builders.document import DocumentBuilder
from marker.config.parser import ConfigParser
from marker.image_export import ImageExportSettings, scaled_image_size
from marker.output import save_output
from marker.providers.pdf import PdfProvider
from marker.renderers.markdown import MarkdownOutput, MarkdownRenderer
from marker.schema import BlockTypes
from marker.schema.polygon import PolygonBox
from marker.schema.registry import get_block_class

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("values", [
    {"dpi": 0}, {"dpi": 601}, {"dpi": True}, {"dpi": "300"},
    {"scale_percent": 0}, {"scale_percent": 401}, {"scale_percent": 50.5},
    {"max_dimension": -1}, {"max_dimension": 20001},
])
def test_invalid_settings_are_rejected(values):
    with pytest.raises(ValueError):
        ImageExportSettings.from_dict(values)


def test_preferences_survive_restart_and_invalid_files_fall_back(tmp_path):
    path = tmp_path / "settings.json"
    options = ImageExportSettings(300, 75, 2400)
    convert_gui.save_image_settings(path, options)
    assert convert_gui.load_image_settings(path) == (options, "")
    assert list(tmp_path.iterdir()) == [path]
    path.write_text('{"dpi": -1}')
    fallback, warning = convert_gui.load_image_settings(path)
    assert fallback == ImageExportSettings()
    assert "using defaults" in warning
    path.write_text('{"dpi":')
    assert convert_gui.load_image_settings(path)[1]


def test_failed_save_preserves_previous_preferences(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    convert_gui.save_image_settings(path, ImageExportSettings())

    def fail(*args):
        raise OSError("Cannot replace settings")

    monkeypatch.setattr(type(path), "replace", fail)
    with pytest.raises(OSError):
        convert_gui.save_image_settings(path, ImageExportSettings(300))
    assert convert_gui.load_image_settings(path) == (ImageExportSettings(), "")
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("size,scale,limit,expected", [
    ((1200, 900), 50, 0, (600, 450)),
    ((1200, 900), 200, 1600, (1600, 1200)),
    ((900, 1200), 100, 600, (450, 600)),
    ((1200, 900), 50, 2000, (600, 450)),
    ((1, 10), 10, 0, (1, 1)),
])
def test_scaling_preserves_proportions_and_caps_after_scaling(size, scale, limit, expected):
    assert scaled_image_size(size, scale, limit) == expected


@pytest.fixture
def photo_pdf(tmp_path):
    # Two differently coloured 4 × 3 inch pages, no external documents/models.
    path = tmp_path / "photos.pdf"
    first = Image.new("RGB", (288, 216), "red")
    second = Image.new("RGB", (288, 216), "blue")
    first.save(path, "PDF", resolution=72, save_all=True, append_images=[second])
    return path


def photo_document(path, config):
    provider = PdfProvider(str(path), {"pdftext_workers": 1})
    document = DocumentBuilder(config).build_document(provider)
    for page in document.pages:
        # A 2 × 1.5 inch crop, offset from the top-left of each page.
        picture = get_block_class(BlockTypes.Picture)(
            page_id=page.page_id, block_id=0,
            polygon=PolygonBox.from_bbox([72, 54, 216, 162]),
        )
        page.children = [picture]
        page.structure = [picture.id]
    return document


@pytest.mark.parametrize("options,expected", [
    (ImageExportSettings(), (384, 288)),
    (ImageExportSettings(96), (192, 144)),
    (ImageExportSettings(300), (600, 450)),
    (ImageExportSettings(600, 50), (600, 450)),
    (ImageExportSettings(300, 200, 800), (800, 600)),
])
def test_real_pdf_crop_is_saved_at_selected_size(photo_pdf, tmp_path, options, expected):
    config = options.converter_config()
    document = photo_document(photo_pdf, config)
    renderer = MarkdownRenderer(config)
    images = {}
    for page in document.pages:
        picture = page.children[0]
        crop = renderer.extract_image(document, picture.id)
        # PDFium rounds page sizes up and Pillow rounds crop coordinates;
        # fractional pixel boundaries can contribute one extra pixel.
        assert all(abs(actual - target) <= 1 for actual, target in zip(crop.size, expected))
        if options.max_dimension:
            assert max(crop.size) <= options.max_dimension
        # Repeated access reuses the current page; another page cannot get its pixels.
        assert renderer.extract_image(document, picture.id).size == crop.size
        assert crop.getpixel((0, 0))[0 if page.page_id == 0 else 2] > 200
        images[f"page-{page.page_id}.jpeg"] = crop
        assert page.get_image(highres=False).size == (384, 288)
        assert page.get_image(highres=True).size == (768, 576)
    rendered = MarkdownOutput(markdown="Photos", images=images, metadata={})
    save_output(rendered, str(tmp_path), "export")
    for name in images:
        with Image.open(tmp_path / name) as saved:
            assert saved.size == images[name].size
            assert saved.format == "JPEG"


def test_unconfigured_renderer_keeps_existing_export_dimensions(photo_pdf):
    document = photo_document(photo_pdf, {})
    picture = document.pages[0].children[0]
    assert MarkdownRenderer().extract_image(document, picture.id).size == (384, 288)
    assert MarkdownRenderer({"image_extraction_mode": "lowres"}).extract_image(
        document, picture.id
    ).size == (192, 144)


def test_client_sends_settings_with_each_request(tmp_path):
    proc = SimpleNamespace(
        stdin=io.BytesIO(), stdout=io.BytesIO(b'{"ok": true, "out": "doc.md"}\n'),
        poll=lambda: None,
    )
    client = convert_gui.MarkerWorker()
    client.proc = proc
    options = ImageExportSettings(300, 50, 2400)
    assert client.convert("doc.pdf", str(tmp_path), options) == (True, "doc.md")
    assert json.loads(proc.stdin.getvalue())["image_settings"] == options.to_dict()


def test_single_and_batch_keep_the_settings_captured_at_start(tmp_path):
    selected = ImageExportSettings(300, 50, 2400)
    calls = []
    app = SimpleNamespace(image_settings=selected)

    def convert(pdf, out_dir, options, skip_ocr=False):
        calls.append((pdf, options))
        # Simulate someone changing the Settings tab while a file is running.
        app.image_settings = ImageExportSettings(600)
        return True, "doc.md"

    app.worker = SimpleNamespace(start=lambda: (True, "cpu"), convert=convert)
    tab = SimpleNamespace(app=app, events=queue.Queue(), cancel=threading.Event())
    convert_gui.SingleTab._run(tab, "single.pdf", str(tmp_path), selected)
    convert_gui.BatchTab._run(
        tab, [tmp_path / "first.pdf", tmp_path / "second.pdf"], str(tmp_path), selected
    )
    assert len(calls) == 3
    assert all(options == selected for _, options in calls)


def test_worker_applies_per_request_settings_to_real_exports(photo_pdf, tmp_path, monkeypatch):
    class ExportConverter:
        """Supply detected photo blocks without loading layout/OCR models."""
        def __init__(self, config, **kwargs):
            self.config = config

        def __call__(self, path):
            return MarkdownRenderer(self.config)(photo_document(path, self.config))

    monkeypatch.setattr(ConfigParser, "get_converter_cls", lambda _: ExportConverter)
    for options, expected in [
        (ImageExportSettings(300, 50), (300, 225)),
        (ImageExportSettings(), (384, 288)),
    ]:
        output = marker_worker.convert_one(
            {}, str(photo_pdf), str(tmp_path / f"out-{options.dpi}"),
            image_settings=options.to_dict(),
        )
        output_path = type(photo_pdf)(output)
        assert output_path.exists()
        exported_images = list(output_path.parent.glob("*.jpeg"))
        assert len(exported_images) == 2
        for path in exported_images:
            assert path.name in output_path.read_text()
            with Image.open(path) as saved:
                assert all(abs(actual - target) <= 1 for actual, target in zip(saved.size, expected))
