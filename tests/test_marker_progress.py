import io

import convert_gui

from marker_progress import (
    ProgressEvent,
    ProgressReporter,
    plan_total,
    progress_converter_cls,
    wrap_processors,
)


class FakeProcessor:
    """Stands in for a marker processor: callable, with attributes to forward."""

    def __init__(self, name, calls):
        self.name = name
        self.calls = calls
        self.block_types = ("Table",)

    def __call__(self, document):
        self.calls.append(self.name)
        return document

    @property
    def __class__name__(self):  # pragma: no cover - attribute forwarding probe
        return self.name


def named(cls_name, calls):
    processor = FakeProcessor(cls_name, calls)
    processor.__class__ = type(cls_name, (FakeProcessor,), {})
    return processor


# --- ProgressEvent -----------------------------------------------------------


def test_percentage_is_the_share_of_steps_completed():
    assert ProgressEvent("Rendering", 1, 4).pct == 25
    assert ProgressEvent("Rendering", 4, 4).pct == 100


def test_percentage_never_divides_by_zero():
    assert ProgressEvent("Rendering", 0, 0).pct == 0


def test_payload_matches_what_the_gui_reads():
    payload = ProgressEvent("Tables", 3, 10).as_payload()
    assert payload == {
        "desc": "Tables",
        "cur": "3",
        "total": "10",
        "pct": "30",
        "unit": "steps",
    }


# --- ProgressReporter --------------------------------------------------------


def test_each_step_advances_the_count():
    seen = []
    reporter = ProgressReporter(3, seen.append)

    reporter.step("one")
    reporter.step("two")

    assert [(e.desc, e.cur, e.total) for e in seen] == [("one", 1, 3), ("two", 2, 3)]


def test_the_count_never_runs_past_the_total():
    """A pipeline that reports more phases than planned must not exceed 100%."""
    seen = []
    reporter = ProgressReporter(1, seen.append)

    reporter.step("one")
    reporter.step("surplus")

    assert [e.cur for e in seen] == [1, 1]
    assert seen[-1].pct == 100


def test_a_failing_sink_never_breaks_the_conversion():
    def explode(event):
        raise RuntimeError("the GUI went away")

    reporter = ProgressReporter(2, explode)

    reporter.step("one")  # must not raise


# --- wrap_processors ---------------------------------------------------------


def test_wrapping_reports_each_processor_by_name_as_it_runs():
    calls, seen = [], []
    reporter = ProgressReporter(2, seen.append)
    processors = [named("TableProcessor", calls), named("TextProcessor", calls)]

    for processor in wrap_processors(processors, reporter):
        processor("document")

    assert calls == ["TableProcessor", "TextProcessor"]
    assert [e.desc for e in seen] == ["Table", "Text"]


def test_wrapping_leaves_the_original_list_untouched():
    reporter = ProgressReporter(1, lambda event: None)
    original = [named("TableProcessor", [])]

    wrapped = wrap_processors(original, reporter)

    assert wrapped is not original
    assert original[0].__class__.__name__ == "TableProcessor"
    assert wrapped[0] is not original[0]


def test_a_wrapped_processor_still_looks_like_the_processor():
    reporter = ProgressReporter(1, lambda event: None)
    wrapped = wrap_processors([named("TableProcessor", [])], reporter)[0]

    assert wrapped.block_types == ("Table",)


def test_a_wrapped_processor_returns_what_the_processor_returned():
    reporter = ProgressReporter(1, lambda event: None)

    class Returns:
        def __call__(self, document):
            return "rendered"

    wrapped = wrap_processors([Returns()], reporter)[0]

    assert wrapped("document") == "rendered"


# --- plan_total --------------------------------------------------------------


def test_the_plan_counts_every_processor_plus_the_fixed_phases():
    """Analysis, rendering and saving bracket the processor pipeline."""

    class Converter:
        processor_list = [1, 2, 3]

    assert plan_total(Converter()) == 6


# --- progress_converter_cls --------------------------------------------------


class FakeConverter:
    def __init__(self):
        self.processor_list = []
        self.built = []

    def build_document(self, filepath):
        self.built.append(filepath)
        for processor in self.processor_list:
            processor("document")
        return "document"


def test_analysis_is_reported_before_the_work_and_rendering_after():
    seen = []
    cls = progress_converter_cls(FakeConverter)
    converter = cls()
    converter.processor_list = [named("TableProcessor", [])]
    converter.attach_progress(ProgressReporter(plan_total(converter), seen.append))

    converter.build_document("doc.pdf")

    assert [e.desc for e in seen] == ["Analyzing", "Table", "Rendering"]


def test_the_underlying_conversion_still_runs():
    cls = progress_converter_cls(FakeConverter)
    converter = cls()
    converter.attach_progress(ProgressReporter(3, lambda event: None))

    assert converter.build_document("doc.pdf") == "document"
    assert converter.built == ["doc.pdf"]


def test_a_converter_without_progress_attached_still_converts():
    """Attaching progress is optional; the subclass must not require it."""
    cls = progress_converter_cls(FakeConverter)
    converter = cls()

    assert converter.build_document("doc.pdf") == "document"


def test_attach_progress_wraps_the_processors_it_finds():
    seen = []
    cls = progress_converter_cls(FakeConverter)
    converter = cls()
    converter.processor_list = [named("TextProcessor", [])]
    converter.attach_progress(ProgressReporter(4, seen.append))

    assert converter.processor_list[0].block_types == ("Table",)
    converter.build_document("doc.pdf")
    assert "Text" in [e.desc for e in seen]


# --- the GUI's side of the protocol ------------------------------------------
#
# Importing convert_gui opens no window: it only defines constants at module
# scope, and App().mainloop() sits behind __main__.


class FakeProc:
    """A worker whose stdout is a canned sequence of lines."""

    def __init__(self, lines):
        self.stdout = io.BytesIO(b"".join(line.encode() for line in lines))

    def poll(self):
        return None


def client_reading(lines):
    client = convert_gui.MarkerWorker()
    client.proc = FakeProc(lines)
    seen = []
    client.emit = lambda kind, payload: seen.append((kind, payload))
    return client, seen


def test_progress_lines_are_forwarded_and_the_reply_still_arrives():
    client, seen = client_reading(
        [
            '{"progress": {"desc": "Analyzing", "pct": "5", "cur": "1", "total": "20"}}\n',
            '{"progress": {"desc": "Table", "pct": "50", "cur": "10", "total": "20"}}\n',
            '{"ok": true, "out": "/tmp/doc.md"}\n',
        ]
    )

    reply = client._read_reply()

    assert reply == {"ok": True, "out": "/tmp/doc.md"}
    assert [p["desc"] for _, p in seen] == ["Analyzing", "Table"]
    assert all(kind == "progress" for kind, _ in seen)


def test_a_reply_with_no_progress_is_returned_untouched():
    client, seen = client_reading(['{"ok": true, "out": "/tmp/doc.md"}\n'])

    assert client._read_reply() == {"ok": True, "out": "/tmp/doc.md"}
    assert seen == []


def test_a_worker_that_dies_mid_progress_reports_an_error():
    client, seen = client_reading(
        ['{"progress": {"desc": "Analyzing", "pct": "5", "cur": "1", "total": "20"}}\n']
    )
    client.tail = ["CUDA out of memory"]

    reply = client._read_reply()

    assert "CUDA out of memory" in reply["error"]
    assert len(seen) == 1  # the progress it did report was still forwarded


def test_garbage_on_stdout_does_not_hang_the_read():
    client, _ = client_reading(["this is not json\n"])

    assert "Unexpected worker output" in client._read_reply()["error"]


def test_a_bare_json_value_is_rejected_rather_than_treated_as_a_reply():
    client, _ = client_reading(["[1, 2, 3]\n"])

    assert "Unexpected worker output" in client._read_reply()["error"]


def test_the_progress_payload_the_worker_sends_is_what_the_gui_renders():
    """The worker's payload must carry every key _apply_progress reads."""
    payload = ProgressEvent("Table", 3, 10).as_payload()

    assert int(payload["pct"]) == 30
    assert payload["cur"] == "3" and payload["total"] == "10"
    assert payload["unit"] == "steps"
