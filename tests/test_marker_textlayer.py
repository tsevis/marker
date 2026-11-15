import io
import json

import convert_gui
import marker_worker
from marker.image_export import ImageExportSettings
from marker_textlayer import report_from_counts, sample_indices

FULL_PAGE = 1500  # characters on an ordinary page of prose


# --- when to suggest skipping OCR --------------------------------------------


def test_a_book_with_one_image_only_cover_still_gets_the_suggestion():
    report = report_from_counts(20, [0] + [FULL_PAGE] * 19)

    assert report.suggest_skip_ocr
    assert (report.checked, report.with_text) == (20, 19)


def test_a_scan_never_gets_the_suggestion():
    """Skipping OCR on a scan would produce an empty document."""
    assert not report_from_counts(20, [0] * 20).suggest_skip_ocr


def test_a_half_scanned_document_does_not_get_the_suggestion():
    counts = [FULL_PAGE] * 10 + [0] * 10

    assert not report_from_counts(20, counts).suggest_skip_ocr


def test_stray_characters_on_a_scan_are_not_a_text_layer():
    """A stamped page number or watermark is text, but not the page's text."""
    assert not report_from_counts(20, [4] * 20).suggest_skip_ocr


def test_a_document_with_no_pages_gets_no_suggestion():
    assert not report_from_counts(0, []).suggest_skip_ocr


def test_the_sample_reaches_the_end_of_a_long_document():
    """Text front matter followed by a scanned body must not look like text."""
    indices = sample_indices(400, 40)

    assert len(indices) == 40
    assert indices[0] == 0
    assert indices[-1] >= 390


def test_a_short_document_is_checked_in_full():
    assert sample_indices(7, 40) == (0, 1, 2, 3, 4, 5, 6)


# --- carrying the choice to the converter ------------------------------------


class RecordingProc:
    """A worker that records the request and answers that it succeeded."""

    def __init__(self):
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(b'{"ok": true, "out": "/tmp/doc.md"}\n')

    def poll(self):
        return None


def test_the_gui_sends_the_skip_ocr_choice_to_the_worker(tmp_path):
    client = convert_gui.MarkerWorker()
    client.proc = RecordingProc()

    client.convert("doc.pdf", str(tmp_path), skip_ocr=True)

    assert json.loads(client.proc.stdin.getvalue())["skip_ocr"] is True


def test_skip_ocr_turns_ocr_off_in_the_converter_config():
    config = marker_worker.converter_options(
        "out", ImageExportSettings(), skip_ocr=True
    )

    assert config["disable_ocr"] is True
