"""Real per-phase progress for a single marker conversion.

marker 2.0.0 runs in fast mode by default on CPU and MPS, and that path emits
no per-stage tqdm bars: on a text PDF the recognition stages never run at all.
The GUI used to scrape those bars off stderr, so its progress bar sat at zero
for the whole conversion.

Rather than parse output that is no longer produced, report the phases the
converter actually performs -- the analysis pass, every processor in its
pipeline, then rendering and saving -- and push them to the GUI as ordinary
worker events.

Nothing here mutates what it is given: `wrap_processors` returns a new list,
and the converter subclass is built fresh from whatever class marker resolved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Sequence

# Phases that bracket the processor pipeline: the analysis pass that builds the
# document, the render at the end, and the save the worker performs afterwards.
FIXED_PHASES = 3

ANALYZE_DESC = "Analyzing"
RENDER_DESC = "Rendering"
SAVE_DESC = "Saving"

# Processor class names are pipeline jargon; the GUI shows a plain noun.
_PROCESSOR_SUFFIX = "Processor"


@dataclass(frozen=True)
class ProgressEvent:
    """One completed phase of a conversion."""

    desc: str
    cur: int
    total: int

    @property
    def pct(self) -> int:
        if self.total <= 0:
            return 0
        return min(100, int(self.cur * 100 / self.total))

    def as_payload(self) -> dict:
        """The shape the GUI's progress bar reads."""
        return {
            "desc": self.desc,
            "cur": str(self.cur),
            "total": str(self.total),
            "pct": str(self.pct),
            "unit": "steps",
        }


class ProgressReporter:
    """Counts completed phases and hands each one to a sink.

    The sink reaches the GUI over a pipe that may already be closed, so a sink
    that raises is swallowed: progress reporting must never take a conversion
    down with it.
    """

    def __init__(self, total: int, sink: Callable[[ProgressEvent], None]):
        self._total = max(0, int(total))
        self._sink = sink
        self._cur = 0

    def step(self, desc: str) -> None:
        # A pipeline may report more phases than were planned (an LLM processor
        # that fans out, say). Cap rather than let the bar run past 100%.
        self._cur = min(self._cur + 1, self._total)
        try:
            self._sink(ProgressEvent(desc, self._cur, self._total))
        except Exception:  # noqa: BLE001 - a dead GUI must not fail the file
            pass


def friendly_name(processor: Any) -> str:
    """'TableProcessor' -> 'Table'; anything unnamed keeps its class name."""
    name = type(processor).__name__
    if name.endswith(_PROCESSOR_SUFFIX) and len(name) > len(_PROCESSOR_SUFFIX):
        return name[: -len(_PROCESSOR_SUFFIX)]
    return name


class _ReportingProcessor:
    """Calls a processor, reporting it first, and otherwise stays out of the way.

    marker reads attributes off its processors (block types, config), so this
    forwards everything it does not define itself.
    """

    def __init__(self, processor: Any, reporter: ProgressReporter):
        self._processor = processor
        self._reporter = reporter
        self._desc = friendly_name(processor)

    def __call__(self, *args, **kwargs):
        self._reporter.step(self._desc)
        return self._processor(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._processor, name)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<progress {self._processor!r}>"


def wrap_processors(
    processors: Iterable[Any], reporter: ProgressReporter
) -> List[_ReportingProcessor]:
    """Return a NEW list of reporting processors; the input is left alone."""
    return [_ReportingProcessor(processor, reporter) for processor in processors]


def plan_total(converter: Any) -> int:
    """How many phases a conversion with this converter will report."""
    processors: Sequence[Any] = getattr(converter, "processor_list", None) or ()
    return len(processors) + FIXED_PHASES


def progress_converter_cls(base_cls: type) -> type:
    """Subclass `base_cls` so that it reports the phases around its processors.

    The subclass only brackets `build_document`; it never reimplements it, so
    marker keeps full ownership of the pipeline and upstream changes to it
    carry through untouched.
    """

    class ProgressConverter(base_cls):  # type: ignore[misc, valid-type]
        _reporter: ProgressReporter | None = None

        def attach_progress(self, reporter: ProgressReporter):
            """Report through `reporter`, including one step per processor."""
            self._reporter = reporter
            self.processor_list = wrap_processors(self.processor_list, reporter)
            return self

        def build_document(self, filepath):
            if self._reporter is not None:
                self._reporter.step(ANALYZE_DESC)
            document = super().build_document(filepath)
            # build_document returns straight into rendering in __call__.
            if self._reporter is not None:
                self._reporter.step(RENDER_DESC)
            return document

    ProgressConverter.__name__ = f"Progress{base_cls.__name__}"
    return ProgressConverter
