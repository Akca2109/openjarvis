"""Safe interactive confirmation for tool calls in CLI surfaces.

Confirmation prompts embed tool arguments chosen by the model, so they are
attacker-influenced text shown at the exact moment the user makes a security
decision. Everything displayed here is therefore treated as untrusted:

* Rich markup is never interpreted (``[conceal]`` shows up literally).
* ANSI/ESC sequences and C0/C1 control characters are removed, as are
  invisible Unicode format characters (bidi overrides, zero-width chars).
* Newlines and other whitespace are collapsed so a second line cannot pose as
  the approval target.
* The prompt is bounded, with a visible truncation marker.

The answer defaults to *No*; EOF and Ctrl-C deny. A process-wide lock
serializes prompts so concurrent tool calls cannot interleave their questions
and answers. Nothing here logs the prompt, which carries raw tool arguments.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from typing import Callable, Optional

import click
from rich.console import Console
from rich.markup import escape
from rich.text import Text

# Long enough for a realistic shell command, short enough to read in full.
MAX_CONFIRM_PROMPT_CHARS = 1000

# One lock for every interactive confirmation in the process.
_CONFIRM_LOCK = threading.Lock()

# Terminal escape sequences, removed as a whole so their parameters do not
# leak into the display as stray text.
_ESCAPE_SEQUENCE_RE = re.compile(
    "|".join(
        (
            # String sequences: ESC ] / P / X / ^ / _ (or C1 forms) up to BEL/ST.
            r"(?:\x1b[\]PX^_]|[\x90\x98\x9d\x9e\x9f])[^\x07\x1b\x9c]*"
            r"(?:\x07|\x1b\\|\x9c)",
            # CSI: ESC [ or C1 CSI, parameters, intermediates, final byte.
            r"(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]",
            # Other ESC sequences (nF / Fp / Fe / Fs).
            r"\x1b[ -/]*[0-~]",
        )
    )
)
_WHITESPACE_RUN_RE = re.compile(r"\s+")
_HORIZONTAL_WHITESPACE_RUN_RE = re.compile(r"[^\S\n]+")


def _neutralize_char(char: str, *, single_line: bool) -> str:
    if char == "\n" and not single_line:
        return char
    category = unicodedata.category(char)
    if category == "Cc" or category in ("Zl", "Zp"):
        # C0/C1 controls (incl. CR, TAB, ESC remnants) and line separators.
        return " "
    if category == "Cf":
        # Invisible format characters: bidi overrides, zero-width chars.
        return ""
    return char


def sanitize_terminal_text(
    value: object,
    *,
    single_line: bool = True,
    max_chars: Optional[int] = None,
) -> str:
    """Return *value* as plain text that cannot drive the terminal.

    With ``single_line`` every whitespace run (including newlines) collapses
    to one space; otherwise newlines are kept and other whitespace runs
    collapse. The result is not Rich-escaped — use :func:`safe_rich_text`
    when interpolating into a markup string.
    """
    text = _ESCAPE_SEQUENCE_RE.sub("", str(value))
    text = "".join(_neutralize_char(c, single_line=single_line) for c in text)
    if single_line:
        text = _WHITESPACE_RUN_RE.sub(" ", text).strip()
    else:
        text = _HORIZONTAL_WHITESPACE_RUN_RE.sub(" ", text)
    if max_chars is not None and len(text) > max_chars:
        hidden = len(text) - max_chars
        text = f"{text[:max_chars]} …[truncated {hidden} more chars]"
    return text


def safe_rich_text(
    value: object,
    *,
    single_line: bool = True,
    max_chars: Optional[int] = None,
) -> str:
    """Sanitize *value* and escape it for use inside a Rich markup string."""
    return escape(
        sanitize_terminal_text(value, single_line=single_line, max_chars=max_chars)
    )


def confirm_tool_call(
    prompt: str,
    *,
    console: Optional[Console] = None,
    input_fn: Optional[Callable[[], str]] = None,
    max_chars: int = MAX_CONFIRM_PROMPT_CHARS,
) -> bool:
    """Ask the user to approve a ``requires_confirmation`` tool call.

    With *console*, prints ``Confirm: <prompt> [y/N]`` through Rich and reads
    the answer with *input_fn* (default :func:`input`). Without it, uses
    :func:`click.confirm` on stderr so stdout stays clean for ``--json``.
    Only an explicit ``y``/``yes`` approves; EOF and Ctrl-C deny.
    """
    safe_prompt = sanitize_terminal_text(prompt, max_chars=max_chars)
    with _CONFIRM_LOCK:
        if console is None:
            try:
                return click.confirm(f"\n{safe_prompt}", default=False, err=True)
            except (click.Abort, EOFError, KeyboardInterrupt):
                return False
        read = input_fn if input_fn is not None else input
        console.print(
            Text.assemble(("Confirm:", "yellow"), " ", safe_prompt, " [y/N] "),
            end="",
        )
        try:
            answer = read()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return False
    return answer.strip().lower() in ("y", "yes")


__all__ = [
    "MAX_CONFIRM_PROMPT_CHARS",
    "confirm_tool_call",
    "safe_rich_text",
    "sanitize_terminal_text",
]
