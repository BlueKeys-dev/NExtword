#!/usr/bin/env python3
"""Watch input.txt, show a last-sentence rewrite, and apply it with Tab."""

from __future__ import annotations

import argparse
import re
import select
import sys
import termios
import threading
import time
import tty
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
INPUT_FILE = HERE / "input.txt"
LOG_FILE = HERE / "predict.log"

# minicpm = MiniCPM5-1B, MLX 4-bit. Granite stays available as --model granite.
DEFAULT_MODEL_ALIAS = "minicpm"
MODEL_ALIASES = {
    "minicpm": "openbmb/MiniCPM5-1B-MLX",
    "minicpm5": "openbmb/MiniCPM5-1B-MLX",
    "granite": "mlx-community/granite-4.0-h-350m-bf16",
    "graphite": "mlx-community/granite-4.0-h-350m-bf16",
    "granite-8bit": "mlx-community/granite-4.0-h-350m-8bit",
    "granite-4bit": "mlx-community/granite-4.0-h-350m-4bit",
    "smol": "mlx-community/SmolLM2-360M-Instruct",
}
DEBOUNCE_SEC = 0.1
MAX_CONTEXT_CHARS = 24000
MAX_TOKENS = 12
MAX_REFINE_TOKENS = 64
MAX_EXTRA_WORDS = 8
MAX_PARTIAL_EXTRA_WORDS = 3
TAIL_WORDS = 12
OVERLAY_HEIGHT = 6
SELECTION = re.compile(r"\[\[(.*?)\]\]", re.DOTALL)
REFINE_SYSTEM = (
    "Use the rest of the file as topic context. Refine only the text inside [[ ]]. "
    "Fix grammar, spelling, and spacing. Keep the same meaning. "
    "Reply with only the refined inner text, no brackets."
)
LIST_SYSTEM = (
    "Complete the numbered line. The reply must start with the same number and the same prefix, "
    "expanded into one item that fits the topic. Output only that line."
)
LIST_SHOTS = (
    ("Shopping list\n1. mil", "1. milk"),
    ("Cities to visit\n2. par", "2. Paris"),
    ("Paint colors\n3. yel", "3. yellow"),
)
NEXT_ITEM_SYSTEM = "Write only the next numbered item. Do not repeat an earlier item."
NEXT_ITEM_SHOTS = (
    ("Shopping list\n1. milk\n2. bread", "3. eggs"),
    ("Cities\n1. Paris\n2. Tokyo", "3. London"),
)
REFINE_SHOTS = (
    ("recommand The best movie in the world", "Recommend the best movie in the world."),
    ("chess is quite popular in  the india", "Chess is quite popular in India."),
)
WORD = re.compile(r"[A-Za-z']+")
TOKEN = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*|\d+", re.UNICODE)
FILLER = frozenset({"the", "a", "an", "of", "to", "and"})
ARTICLES = frozenset({"a", "an", "the"})
FUNCTION_WORDS = ARTICLES | FILLER | frozenset(
    {
        "is",
        "am",
        "are",
        "be",
        "was",
        "were",
        "it",
        "in",
        "on",
        "for",
        "you",
        "me",
        "my",
        "some",
        "can",
        "do",
        "we",
        "he",
        "she",
        "they",
        "this",
        "that",
        "with",
        "at",
        "if",
        "so",
        "or",
        "but",
        "not",
        "i",
        "your",
        "his",
        "her",
        "what",
        "how",
        "who",
    }
)
TITLES = frozenset(
    {
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "sr",
        "jr",
        "eg",
        "ie",
        "vs",
        "etc",
        "inc",
        "no",
        "us",
    }
)


def log(line: str) -> None:
    print(line)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def token_before_punct(text: str, index: int) -> str:
    cursor = index - 1
    while cursor >= 0 and text[cursor] in " \t":
        cursor -= 1
    end = cursor + 1
    while cursor >= 0 and (text[cursor].isalnum() or text[cursor] in ".'"):
        cursor -= 1
    return text[cursor + 1 : end]


def is_sentence_end(text: str, index: int) -> bool:
    if index < 0 or index >= len(text):
        return False
    mark = text[index]
    if mark not in ".!?":
        return False
    if index + 1 < len(text) and text[index + 1] not in " \t\n":
        return False
    if mark in "!?":
        return True
    token = token_before_punct(text, index)
    bare = token.replace(".", "").lower()
    if bare in TITLES:
        return False
    if bare.isdigit():
        return False
    if len(bare) == 1 and bare.isalpha():
        return False
    return True


def first_sentence(text: str) -> str:
    for index, char in enumerate(text):
        if char in ".!?" and is_sentence_end(text, index):
            return text[: index + 1]
    return text


def split_active_sentence(text: str) -> tuple[str, str]:
    """Frozen prefix plus the unfinished sentence after the last real sentence end."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")
    if not text.strip():
        return "", ""

    last_end = -1
    for index, char in enumerate(text):
        if is_sentence_end(text, index):
            last_end = index

    if last_end == -1:
        if "\n" not in text:
            return "", text
        earlier, last = text.rsplit("\n", 1)
        return earlier + "\n", last

    if last_end == len(text) - 1:
        return text, ""

    frozen = text[: last_end + 1]
    rest = text[last_end + 1 :]
    skipped = len(rest) - len(rest.lstrip(" \t\n"))
    frozen += rest[:skipped]
    return frozen, rest[skipped:]


def split_cursor(fragment: str) -> tuple[str, str]:
    if "|" not in fragment:
        return fragment, ""
    left, right = fragment.split("|", 1)
    return left, right


def last_selection(text: str) -> re.Match[str] | None:
    match: re.Match[str] | None = None
    for found in SELECTION.finditer(text):
        match = found
    return match


def selection_surrounding(text: str, match: re.Match[str]) -> str:
    return text[: match.start()] + text[match.end() :]


def split_trailing_ws(fragment: str) -> tuple[str, str]:
    core = fragment.rstrip(" \t")
    return core, fragment[len(core) :]


def tail_window(fragment: str, limit: int = TAIL_WORDS) -> tuple[str, str]:
    matches = list(TOKEN.finditer(fragment))
    if len(matches) <= limit:
        return "", fragment
    start = matches[-limit].start()
    return fragment[:start], fragment[start:]


def prepare_fragment(fragment: str) -> tuple[str, str, str]:
    core, trail = split_trailing_ws(fragment)
    head, tail = tail_window(core)
    return head, tail, trail


def apply_fragment(path: Path, suggestion: str, mode: str = "sentence") -> str:
    raw = path.read_text(encoding="utf-8") if path.exists() else ""
    if mode == "selection":
        match = last_selection(raw)
        if match is None:
            return raw.rstrip("\n")
        written = raw[: match.start()] + suggestion + raw[match.end() :]
    elif suggestion.startswith("\n"):
        frozen, fragment = split_active_sentence(raw)
        written = frozen + fragment.rstrip(" \t") + suggestion
    else:
        frozen, _ = split_active_sentence(raw)
        written = frozen + suggestion
    if not written.endswith("\n"):
        written += "\n"
    path.write_text(written, encoding="utf-8")
    return written.rstrip("\n")


def letters_in(text: str) -> str:
    return "".join(char for char in text if char.isalpha())


def too_small_fragment(fragment: str) -> bool:
    working, _ = split_cursor(fragment)
    if not letters_in(working):
        return True
    words = TOKEN.findall(working)
    return len(words) == 1 and len(words[0]) <= 2


def last_word_partial(fragment: str) -> bool:
    stripped = fragment.rstrip()
    if not stripped:
        return False
    if stripped.endswith("-"):
        return True
    if stripped[-1] in "'’":
        return True
    if fragment.endswith((" ", "\t")):
        return False
    return bool(TOKEN.findall(fragment) or letters_in(fragment))


def ready_for_next_line(fragment: str) -> bool:
    """A trailing space after a content word means the line is done."""
    if not fragment.endswith((" ", "\t")):
        return False
    words = TOKEN.findall(fragment)
    if not words:
        return False
    return words[-1].lower() not in FUNCTION_WORDS


def must_extend_last_word(fragment: str) -> bool:
    """A mid-word prefix must grow. Function words like 'is' may take a next word."""
    if not last_word_partial(fragment):
        return False
    words = TOKEN.findall(fragment)
    if not words:
        return False
    return words[-1].lower() not in FUNCTION_WORDS


def extra_word_count(fragment: str, generated: str) -> int:
    return max(0, len(TOKEN.findall(generated)) - len(TOKEN.findall(fragment)))


def extra_after_last_word(fragment: str, suggestion: str) -> int:
    frag_words = TOKEN.findall(fragment)
    sug_words = [word.lower() for word in TOKEN.findall(suggestion)]
    if not frag_words or not sug_words:
        return extra_word_count(fragment, suggestion)
    last = frag_words[-1].lower()
    match_at = None
    for index, word in enumerate(sug_words):
        if word.startswith(last):
            match_at = index
    if match_at is None:
        return extra_word_count(fragment, suggestion)
    return max(0, len(sug_words) - match_at - 1)


def levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, char in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char != other))
            )
        previous = current
    return previous[-1]


def is_digit_word_swap(left: str, right: str) -> bool:
    return left.isdigit() != right.isdigit() and (left.isdigit() or right.isdigit())


def is_spelling_fix(typed: str, suggested: str) -> bool:
    a, b = typed.lower(), suggested.lower()
    if a == b:
        return True
    compact = b.replace(" ", "")
    if min(len(a), max(len(b), len(compact))) < 3:
        return False
    if levenshtein(a, b) <= 2:
        return True
    return levenshtein(a, compact) <= 2


def last_word_prefix_ok(fragment: str, suggestion: str) -> bool:
    if not last_word_partial(fragment):
        return True
    typed = TOKEN.findall(fragment)
    suggested = TOKEN.findall(suggestion)
    if not typed or not suggested:
        return False
    needle = typed[-1]
    nlow = needle.lower()
    for index, word in enumerate(suggested):
        wlow = word.lower()
        if wlow.startswith(nlow) or (nlow.endswith("-") and wlow.startswith(nlow.rstrip("-"))):
            return True
        if is_spelling_fix(needle, word) and len(wlow) >= len(nlow):
            return True
        if index + 1 < len(suggested) and is_spelling_fix(needle, word + " " + suggested[index + 1]):
            return True
    if fragment.rstrip().endswith("-"):
        return any(word.lower().startswith(nlow) or word.lower().startswith(nlow + "-") for word in suggested)
    return False


def polish_punctuation(fragment: str, suggestion: str) -> str:
    suggestion = first_sentence(suggestion.split("\n", 1)[0].split("\r", 1)[0].strip())
    keep_question = fragment.rstrip().endswith("?")
    keep_exclaim = fragment.rstrip().endswith("!")
    unfinished = last_word_partial(fragment) or not fragment.rstrip().endswith((".", "!", "?"))
    if unfinished and suggestion.endswith(".") and not suggestion.endswith("..."):
        suggestion = suggestion[:-1].rstrip()
    if keep_question and not suggestion.endswith("?"):
        suggestion = suggestion.rstrip(".!") + "?"
    elif keep_exclaim and not suggestion.endswith("!"):
        suggestion = suggestion.rstrip(".") + "!"
    return suggestion


LIST_PREFIX = re.compile(r"^(\d+)\.(\s*)")


def reconstruct(fragment: str, words: list[str]) -> str:
    matches = list(TOKEN.finditer(fragment))
    if not matches or not words:
        return " ".join(words)
    listed = LIST_PREFIX.match(fragment)
    if listed and words[0].isdigit():
        rest = " ".join(words[1:])
        gap = listed.group(2) or " "
        body = f"{listed.group(1)}.{gap}{rest}".rstrip()
        trail = fragment[matches[-1].end() :]
        if trail.strip() and not rest:
            body += trail
        return body
    if len(words) == len(matches):
        pieces: list[str] = []
        last = 0
        for match, word in zip(matches, words):
            pieces.append(fragment[last : match.start()])
            pieces.append(word)
            last = match.end()
        trail = fragment[last:]
        if words and trail and all(char in "'’-" or char.isspace() for char in trail):
            absorbed = set(words[-1]) & set("'’-")
            if absorbed:
                trail = "".join(char for char in trail if char not in absorbed)
        pieces.append(trail)
        return "".join(pieces)
    lead = fragment[: matches[0].start()]
    trail = fragment[matches[-1].end() :]
    joined = " ".join(words)
    if len(words) > len(matches):
        return lead + joined + (trail if not trail.strip() else "")
    return lead + joined + trail


def choose_equal_case(typed: str, suggested: str, *, first: bool) -> str:
    if typed.lower() != suggested.lower():
        return typed
    if first and typed[:1].islower() and suggested[:1].isupper():
        if any(char.isupper() for char in typed[1:]):
            return typed
        return suggested[:1] + typed[1:]
    return typed


def choose_token(typed: str, suggested: str, *, first: bool, partial: bool) -> str:
    a, b = typed.lower(), suggested.lower()
    if is_digit_word_swap(typed, suggested):
        return typed
    if partial and b.startswith(a):
        return suggested
    if a == b:
        return choose_equal_case(typed, suggested, first=first)
    if not first and typed.isupper() and typed.isalpha():
        return typed
    if first and a[1:] == b[1:] and typed[:1].islower() and suggested[:1].isupper():
        return suggested[:1] + typed[1:]
    if is_spelling_fix(typed, suggested) and not is_digit_word_swap(typed, suggested):
        if typed.isupper() and typed.isalpha():
            return suggested.upper() if suggested.isalpha() else suggested
        if typed[:1].isupper():
            return suggested[:1].upper() + suggested[1:]
        return suggested
    return typed


def token_compatible(typed: str, suggested: str, partial: bool) -> bool:
    if not suggested:
        return False
    a, b = typed.lower(), suggested.lower()
    if a == b:
        return True
    if partial and b.startswith(a):
        return True
    if is_spelling_fix(typed, suggested) and not is_digit_word_swap(typed, suggested):
        return True
    return False


def take_insert(word: str, *, at_end: bool, extra_used: int, cap: int) -> tuple[str | None, int]:
    if word.lower() in ARTICLES:
        return word, extra_used
    if at_end and extra_used < cap:
        return word, extra_used + 1
    return None, extra_used


def last_partial_match(pslice: list[str], last: str) -> int | None:
    found: int | None = None
    needle = last.lower()
    for index, word in enumerate(pslice):
        low = word.lower()
        if low.startswith(needle) and low != needle:
            found = index
        elif is_spelling_fix(last, word) and len(low) >= len(needle) and low != needle:
            found = index
    return found


def align_replace(
    tslice: list[str],
    pslice: list[str],
    *,
    i1: int,
    typed_len: int,
    partial: bool,
    extra_used: int,
    cap: int,
    at_end: bool,
) -> tuple[list[str], int]:
    if partial and tslice and pslice and i1 + len(tslice) == typed_len:
        match_at = last_partial_match(pslice, tslice[-1])
        article_before = len(tslice) >= 2 and tslice[-2].lower() in ARTICLES
        if match_at is not None and article_before:
            prev = pslice[match_at - 1].lower() if match_at else ""
            if prev not in ARTICLES:
                match_at = None
        if match_at is not None:
            head, extra_used = align_replace(
                tslice[:-1],
                pslice[:match_at],
                i1=i1,
                typed_len=i1 + len(tslice) - 1,
                partial=False,
                extra_used=extra_used,
                cap=cap,
                at_end=False,
            )
            out = head + [pslice[match_at]]
            for word in pslice[match_at + 1 :]:
                taken, extra_used = take_insert(word, at_end=True, extra_used=extra_used, cap=cap)
                if taken is not None:
                    out.append(taken)
            return out, extra_used
    out: list[str] = []
    ti = 0
    pi = 0
    while ti < len(tslice) or pi < len(pslice):
        tword = tslice[ti] if ti < len(tslice) else None
        if tword is not None and pi + 1 < len(pslice):
            joined = pslice[pi] + " " + pslice[pi + 1]
            compact = pslice[pi] + pslice[pi + 1]
            if is_spelling_fix(tword, joined) or tword.lower() == compact.lower():
                out.extend(pslice[pi : pi + 2])
                ti += 1
                pi += 2
                continue
        if tword is not None and pi < len(pslice) and pslice[pi].lower() in ARTICLES:
            nxt = pslice[pi + 1] if pi + 1 < len(pslice) else ""
            if token_compatible(
                tword,
                nxt,
                partial=partial and i1 + ti == typed_len - 1,
            ):
                out.append(pslice[pi])
                pi += 1
                continue
        if tword is not None and pi < len(pslice):
            out.append(
                choose_token(
                    tword,
                    pslice[pi],
                    first=i1 + ti == 0,
                    partial=partial and i1 + ti == typed_len - 1,
                )
            )
            ti += 1
            pi += 1
            continue
        if tword is not None:
            out.append(tword)
            ti += 1
            continue
        taken, extra_used = take_insert(pslice[pi], at_end=at_end, extra_used=extra_used, cap=cap)
        if taken is not None:
            out.append(taken)
        pi += 1
    return out, extra_used


def apply_word_edits(fragment: str, suggestion: str) -> str:
    typed = TOKEN.findall(fragment)
    proposed = TOKEN.findall(suggestion)
    if not typed:
        return suggestion
    a = [word.lower() for word in typed]
    b = [word.lower() for word in proposed]
    out: list[str] = []
    extra_used = 0
    partial = last_word_partial(fragment)
    cap = MAX_PARTIAL_EXTRA_WORDS if partial else MAX_EXTRA_WORDS
    opcodes = SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
    for tag, i1, i2, j1, j2 in opcodes:
        tslice = typed[i1:i2]
        pslice = proposed[j1:j2]
        at_end = i2 == len(typed)
        if tag == "equal":
            for offset, (typed_word, proposed_word) in enumerate(zip(tslice, pslice)):
                out.append(choose_equal_case(typed_word, proposed_word, first=i1 + offset == 0))
            continue
        if tag == "delete":
            out.extend(tslice)
            continue
        if tag == "insert":
            for word in pslice:
                if word.lower() in ARTICLES:
                    out.append(word)
                elif at_end and extra_used < cap:
                    out.append(word)
                    extra_used += 1
            continue
        aligned, extra_used = align_replace(
            tslice,
            pslice,
            i1=i1,
            typed_len=len(typed),
            partial=partial,
            extra_used=extra_used,
            cap=cap,
            at_end=at_end,
        )
        out.extend(aligned)
    while extra_used and out and out[-1].lower() in FILLER:
        out.pop()
        extra_used -= 1
    return reconstruct(fragment, out)


def strip_trailing_filler(suggestion: str) -> str:
    words = list(TOKEN.finditer(suggestion))
    while words and words[-1].group().lower() in FILLER:
        suggestion = suggestion[: words[-1].start()].rstrip(" ,;:")
        words = list(TOKEN.finditer(suggestion))
    return suggestion.strip()


def trim_extra_words(fragment: str, suggestion: str, limit: int | None = None) -> str:
    if limit is None:
        limit = MAX_PARTIAL_EXTRA_WORDS if last_word_partial(fragment) else MAX_EXTRA_WORDS
    marks = trailing_marks(suggestion)
    typed = TOKEN.findall(fragment)
    matches = list(TOKEN.finditer(suggestion))
    if last_word_partial(fragment) and typed:
        last = typed[-1].lower()
        match_at = None
        for index, match in enumerate(matches):
            if match.group().lower().startswith(last):
                match_at = index
        if match_at is not None:
            end_at = min(len(matches) - 1, match_at + limit)
            suggestion = suggestion[: matches[end_at].end()]
            suggestion = strip_trailing_filler(suggestion)
            if marks and marks not in ".!?" and not suggestion.endswith(marks):
                suggestion += marks
            return suggestion
    keep = len(typed) + limit
    if len(matches) > keep:
        suggestion = suggestion[: matches[keep - 1].end()]
    suggestion = strip_trailing_filler(suggestion)
    if marks and marks not in ".!?" and not suggestion.endswith(marks):
        suggestion += marks
    return suggestion


def should_stop_next_line(generated: str) -> bool:
    if "\n" in generated or "\r" in generated:
        return True
    for index, char in enumerate(generated):
        if char in ".!?" and is_sentence_end(generated, index):
            return True
    if len(TOKEN.findall(generated)) >= MAX_EXTRA_WORDS and generated[-1] in " \t,;:":
        return True
    return False


def should_stop_generation(fragment: str, generated: str) -> bool:
    if "\n" in generated or "\r" in generated:
        return True
    if (
        LIST_PREFIX.match(fragment.strip())
        and last_word_partial(fragment)
        and not last_word_completed(fragment, generated)
    ):
        gen = generated.strip().rstrip(".!?")
        frag = fragment.strip().rstrip(".!?")
        if fold_tokens(gen) == fold_tokens(frag) or gen.lower() == frag.lower():
            return False
    for index, char in enumerate(generated):
        if char in ".!?" and is_sentence_end(generated, index):
            return True
    limit = MAX_PARTIAL_EXTRA_WORDS if last_word_partial(fragment) else MAX_EXTRA_WORDS
    if extra_word_count(fragment, generated) < limit:
        return False
    return (not generated) or generated[-1].isspace() or generated[-1] in ",;:"


def extra_sentence(suggestion: str) -> bool:
    stripped = suggestion.strip()
    for index, char in enumerate(stripped):
        if char in ".!?" and is_sentence_end(stripped, index):
            if stripped[index + 1 :].strip():
                return True
    return False


def copies_earlier(frozen: str, suggestion: str) -> bool:
    haystack = suggestion.strip().lower()
    if not haystack:
        return False
    for line in frozen.split("\n"):
        line = line.strip()
        if len(line) < 8:
            continue
        if line.lower() in haystack:
            return True
    return False


def trailing_marks(text: str) -> str:
    matches = list(TOKEN.finditer(text))
    if not matches:
        return text.strip()
    return text[matches[-1].end() :].strip()


def fold_tokens(text: str) -> list[str]:
    return [word.lower() for word in TOKEN.findall(text)]


def last_word_completed(fragment: str, suggestion: str) -> bool:
    typed = TOKEN.findall(fragment)
    suggested = TOKEN.findall(suggestion)
    if not typed or not suggested:
        return False
    last = typed[-1]
    for word in suggested:
        if word.lower() == last.lower():
            continue
        if word.lower().startswith(last.lower()):
            return True
        if is_spelling_fix(last, word) and len(word) >= len(last):
            return True
    return False


def score_suggestion(
    should_show: bool,
    accept: list[str],
    reject_if: list[str],
    suggestion: str | None,
    raw: str = "",
) -> str:
    shown = bool(suggestion and suggestion.strip())

    def norm(text: str) -> str:
        return " ".join(text.split()).rstrip(".!?").lower()

    accepted = {norm(item) for item in accept}
    if shown:
        if not should_show:
            return "wrong"
        got = suggestion or ""
        if any(flag in got for flag in reject_if):
            return "wrong"
        if accept and norm(got) in accepted:
            return "solid"
        return "flawed"
    if not should_show:
        return "hidden-ok"
    if raw and accept and norm(raw) in accepted:
        return "hidden-miss"
    if raw and any(flag in raw for flag in reject_if):
        return "hidden-ok"
    return "hidden-ok"


def accept_suggestion(frozen: str, fragment: str, suggestion: str) -> str | None:
    if suggestion.startswith("\n"):
        return accept_next_line(frozen, fragment, suggestion[1:])
    body, after_cursor = split_cursor(fragment)
    if too_small_fragment(body):
        return None
    head, tail, trail = prepare_fragment(body)
    if not tail.strip():
        return None
    polished = polish_punctuation(tail, suggestion)
    if not polished:
        return None
    if not last_word_prefix_ok(tail, polished):
        return None
    edited = apply_word_edits(tail, polished)
    extra_marks = trailing_marks(polished)
    if extra_marks and extra_marks not in trailing_marks(edited) and extra_marks not in ".!?":
        edited = edited.rstrip() + extra_marks
    edited = polish_punctuation(tail, edited)
    edited = trim_extra_words(tail, edited)
    if not edited.strip():
        return None
    if extra_sentence(edited):
        return None
    if copies_earlier(frozen, edited):
        return None
    if edited.rstrip(".!?") == tail.strip().rstrip(".!?"):
        return None
    if fold_tokens(edited) == fold_tokens(tail):

        def marks(text: str) -> str:
            return TOKEN.sub("", text.rstrip(".!?")).replace(" ", "")

        if marks(edited) == marks(tail):
            return None
    elif must_extend_last_word(tail) and not last_word_completed(tail, edited):
        return None
    result = head + edited + trail
    if after_cursor:
        result = result + after_cursor
    return result


def accept_next_line(frozen: str, fragment: str, generated: str) -> str | None:
    line = generated.split("\n", 1)[0].split("\r", 1)[0].strip()
    if not TOKEN.findall(line):
        return None
    if line.lower() == fragment.strip().lower():
        return None
    if copies_earlier(frozen, line):
        return None
    return "\n" + line


def clean_refine_output(text: str) -> str:
    text = last_output_line(text).strip()
    if text.startswith("[[") and text.endswith("]]") and len(text) >= 4:
        text = text[2:-2].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def accept_refine(selected: str, suggestion: str, surrounding: str = "") -> str | None:
    text = clean_refine_output(suggestion)
    if not text or not letters_in(text):
        return None
    if "\n" not in selected:
        text = first_sentence(text.split("\n", 1)[0].split("\r", 1)[0].strip())
    if not text:
        return None
    if extra_sentence(text) and not extra_sentence(selected):
        text = first_sentence(text)
    if copies_earlier(surrounding, text):
        return None
    if text.strip() == selected.strip():
        return None
    return text


def resolve_model(name: str | None = None) -> str:
    raw = (name or DEFAULT_MODEL_ALIAS).strip() or DEFAULT_MODEL_ALIAS
    return MODEL_ALIASES.get(raw.lower(), raw)


def load_model(name: str | None = None) -> tuple[Any, Any, Any, Callable[..., Any]]:
    from mlx_lm import load, stream_generate  # type: ignore[import-not-found]
    from mlx_lm.sample_utils import make_sampler  # type: ignore[import-not-found]

    model_id = resolve_model(name)
    loaded = load(model_id, trust_remote_code=False)
    model, tokenizer = loaded[0], loaded[1]
    sampler = make_sampler(temp=0)

    for _ in stream_generate(model, tokenizer, "warmup", max_tokens=2, sampler=sampler):
        pass

    return model, tokenizer, sampler, stream_generate


def context_tail(frozen: str, limit: int = TAIL_WORDS) -> str:
    frozen = frozen.strip()
    if not frozen:
        return ""
    matches = list(TOKEN.finditer(frozen))
    if len(matches) <= limit:
        return frozen
    return frozen[matches[-limit].start() :]


def clip_context(text: str, limit: int = MAX_CONTEXT_CHARS, keep: str = "") -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) <= limit:
        return text
    if keep:
        idx = text.find(keep)
        if idx >= 0:
            extra = max(0, limit - len(keep))
            start = max(0, idx - extra // 2)
            end = min(len(text), start + limit)
            start = max(0, end - limit)
            chunk = text[start:end]
            if start > 0:
                nl = chunk.find("\n")
                if 0 <= nl < 200:
                    chunk = chunk[nl + 1 :]
            return chunk
    clipped = text[-limit:]
    nl = clipped.find("\n")
    if 0 <= nl < 200:
        clipped = clipped[nl + 1 :]
    return clipped


def format_user_prompt(fragment: str, frozen: str = "") -> str:
    """Document to continue. No chat roles: the file itself is the context."""
    whole = (frozen + fragment).replace("\r\n", "\n").replace("\r", "\n")
    return clip_context(whole, keep=fragment)


def join_continuation(fragment: str, generated: str) -> str:
    suffix = generated.split("\n", 1)[0].split("\r", 1)[0]
    if not suffix:
        return ""
    typed = fragment.strip().lower()
    continued = suffix.strip().lower()
    if typed and continued.startswith(typed):
        return suffix.strip()
    return fragment + suffix


def last_output_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def drop_frozen_prefix(frozen: str, text: str) -> str:
    if not frozen.strip() or not text:
        return text
    candidates = [frozen, frozen.strip(), frozen.rstrip()]
    for prefix in candidates:
        if prefix and text.startswith(prefix):
            return text[len(prefix) :].lstrip(" \n")
    return last_output_line(text)


def build_prompt(tokenizer: Any, fragment: str, frozen: str = "") -> str:
    del tokenizer
    return format_user_prompt(fragment, frozen)


def topic_list(fragment: str, frozen: str) -> bool:
    return bool(frozen.strip()) and bool(LIST_PREFIX.match(fragment.strip()))


def list_item_closed(fragment: str) -> bool:
    """A numbered line whose last word is already a whole word should get a next item."""
    if not LIST_PREFIX.match(fragment.strip()):
        return False
    words = TOKEN.findall(fragment)
    if len(words) < 4:
        return False
    return len(words[-1]) >= 6


def next_list_number(fragment: str) -> str:
    match = LIST_PREFIX.match(fragment.strip())
    if match is None:
        return "1"
    return str(int(match.group(1)) + 1)


def renumber_item(line: str, number: str) -> str:
    line = strip_think(line).strip()
    if not line:
        return ""
    match = LIST_PREFIX.match(line)
    rest = line[match.end() :].strip() if match else line
    if not TOKEN.findall(rest):
        return ""
    return f"{number}. {rest}"


def strip_think(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.replace("<think>", "").strip()


def build_list_prompt(tokenizer: Any, fragment: str, frozen: str) -> str:
    whole = clip_context(frozen + fragment, keep=fragment)
    return chat_prompt(tokenizer, LIST_SYSTEM, LIST_SHOTS, whole)


def build_next_item_prompt(tokenizer: Any, fragment: str, frozen: str) -> str:
    whole = clip_context((frozen + fragment).rstrip() + "\n", keep=fragment)
    return chat_prompt(tokenizer, NEXT_ITEM_SYSTEM, NEXT_ITEM_SHOTS, whole)


def chat_prompt(tokenizer: Any, system: str, shots: tuple[tuple[str, str], ...], user: str) -> str:
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for shot_user, assistant in shots:
        messages.append({"role": "user", "content": shot_user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": user})
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def predict(
    model: Any,
    tokenizer: Any,
    sampler: Any,
    stream_generate: Callable[..., Any],
    fragment: str,
    frozen: str = "",
) -> str:
    body = split_cursor(fragment)[0]
    if not body.strip() or too_small_fragment(body):
        return ""

    topic = topic_list(body, frozen)
    closed = list_item_closed(body)
    next_line = ready_for_next_line(body) or closed
    if topic and closed:
        prompt = build_next_item_prompt(tokenizer, body, clip_context(frozen, keep=body))
    elif topic:
        prompt = build_list_prompt(tokenizer, body, clip_context(frozen, keep=body))
    else:
        prompt = build_prompt(
            tokenizer,
            body[-MAX_CONTEXT_CHARS:],
            clip_context(frozen, keep=body),
        )
        if next_line:
            prompt = prompt.rstrip("\n") + "\n"
    generated = ""
    stream = stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=24 if next_line or topic else MAX_TOKENS,
        sampler=sampler,
    )
    try:
        for response in stream:
            generated += response.text
            if topic or next_line:
                if should_stop_next_line(generated):
                    break
            elif should_stop_generation(body, generated):
                break
            if response.finish_reason == "stop":
                break
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()

    if topic and closed:
        line = renumber_item(last_output_line(strip_think(generated)), next_list_number(body))
        return f"\n{line}" if line else ""

    if topic:
        return last_output_line(strip_think(generated))

    if next_line:
        line = generated.split("\n", 1)[0].split("\r", 1)[0].strip()
        return f"\n{line}" if line else ""

    text = join_continuation(body, generated)
    text = first_sentence(text.split("\r", 1)[0].strip())
    return drop_frozen_prefix(frozen, text)


def refine_max_tokens(selected: str) -> int:
    return min(MAX_REFINE_TOKENS, max(16, len(TOKEN.findall(selected)) + 8))


def should_stop_refine(selected: str, generated: str) -> bool:
    if "\n" in generated or "\r" in generated:
        return True
    if extra_sentence(generated):
        return True
    if "\n" not in selected:
        for index, char in enumerate(generated):
            if char in ".!?" and is_sentence_end(generated, index):
                return True
    limit = max(8, len(TOKEN.findall(selected)) + 6)
    if len(TOKEN.findall(generated)) < limit:
        return False
    return (not generated) or generated[-1].isspace() or generated[-1] in ".!?,;:"


def build_refine_prompt(tokenizer: Any, selected: str, file_text: str = "") -> str:
    messages: list[dict[str, str]] = [{"role": "system", "content": REFINE_SYSTEM}]
    for shot_user, assistant in REFINE_SHOTS:
        messages.append({"role": "user", "content": shot_user})
        messages.append({"role": "assistant", "content": assistant})
    source = file_text.replace("\r\n", "\n").replace("\r", "\n") if file_text else selected
    marker = f"[[{selected}]]"
    ctx = clip_context(source, keep=marker if marker in source else selected)
    if file_text.strip() and "[[" in file_text:
        user = (
            f"File:\n{ctx}\n\n"
            "Refine only the text inside [[ ]]. Reply with only the refined inner text, no brackets."
        )
    else:
        user = selected[-MAX_CONTEXT_CHARS:]
    messages.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def predict_refine(
    model: Any,
    tokenizer: Any,
    sampler: Any,
    stream_generate: Callable[..., Any],
    selected: str,
    file_text: str = "",
) -> str:
    if not selected.strip() or not letters_in(selected):
        return ""
    prompt = build_refine_prompt(tokenizer, selected, file_text)
    generated = ""
    stream = stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=refine_max_tokens(selected),
        sampler=sampler,
    )
    try:
        for response in stream:
            generated += response.text
            if should_stop_refine(selected, generated):
                break
            if response.finish_reason == "stop":
                break
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
    return generated


@dataclass
class OverlayState:
    fragment: str = ""
    suggestion: str = ""
    status: str = "waiting"
    mode: str = "sentence"
    dirty: bool = True
    lock: threading.Lock = field(default_factory=threading.Lock)


def clip(text: str, width: int = 78) -> str:
    text = text.replace("\n", "⏎")
    return text if len(text) <= width else text[: width - 1] + "…"


def render(state: OverlayState) -> None:
    with state.lock:
        fragment = state.fragment
        suggestion = state.suggestion
        status = state.status
        mode = state.mode
        state.dirty = False
    title = "selection refine" if mode == "selection" else "next line" if mode == "next" else "sentence rewrite"
    lines = [
        f"┌─ {title}",
        f"│ now  {clip(fragment)}",
        f"│ tab  {clip(suggestion) if suggestion else '(none)'}",
        "│ Tab apply · Esc clear · Ctrl+C quit",
        f"│ {clip(status)}",
        "└────────────────────",
    ]
    sys.stdout.write(f"\033[{OVERLAY_HEIGHT}A\033[J" if hasattr(render, "drawn") else "")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()
    render.drawn = True  # type: ignore[attr-defined]


def set_state(state: OverlayState, **changes: str) -> None:
    with state.lock:
        for key, value in changes.items():
            setattr(state, key, value)
        state.dirty = True


def watch_file(
    state: OverlayState,
    model: Any,
    tokenizer: Any,
    sampler: Any,
    stream_generate: Callable[..., Any],
    stop: threading.Event,
) -> None:
    last_text = ""
    last_mtime = 0.0
    pending_at = 0.0
    while not stop.is_set():
        try:
            mtime = INPUT_FILE.stat().st_mtime
        except FileNotFoundError:
            stop.wait(0.1)
            continue

        if mtime != last_mtime:
            last_mtime = mtime
            pending_at = time.monotonic()

        if pending_at and (time.monotonic() - pending_at) >= DEBOUNCE_SEC:
            pending_at = 0.0
            text = INPUT_FILE.read_text(encoding="utf-8")
            if text == last_text:
                continue
            last_text = text
            selected_match = last_selection(text)
            if not text.strip():
                set_state(state, fragment="", suggestion="", status="empty file", mode="sentence")
                continue
            if selected_match is not None:
                selected = selected_match.group(1)
                if not letters_in(selected):
                    set_state(
                        state,
                        fragment="[[ ]]",
                        suggestion="",
                        status="empty selection",
                        mode="selection",
                    )
                    continue
                set_state(state, fragment=selected, status="predicting…", mode="selection")
                started = time.perf_counter()
                rewritten = accept_refine(
                    selected,
                    predict_refine(model, tokenizer, sampler, stream_generate, selected, text),
                    selection_surrounding(text, selected_match),
                ) or ""
                ms = round((time.perf_counter() - started) * 1000, 1)
                status = f"+{ms}ms" if rewritten else f"+{ms}ms rejected"
                set_state(
                    state,
                    fragment=selected,
                    suggestion=rewritten,
                    status=status,
                    mode="selection",
                )
                stamp = datetime.now().isoformat(timespec="seconds")
                with LOG_FILE.open("a", encoding="utf-8") as f:
                    f.write(f"[{stamp}] +{ms}ms selection\n  now: {selected!r}\n  tab: {rewritten!r}\n")
                continue
            frozen, fragment = split_active_sentence(text)
            if not fragment.strip():
                set_state(state, fragment="", suggestion="", status="sentence complete", mode="sentence")
                continue
            if too_small_fragment(fragment):
                set_state(
                    state,
                    fragment=fragment,
                    suggestion="",
                    status="too little text",
                    mode="sentence",
                )
                continue

            set_state(state, fragment=fragment, status="predicting…", mode="sentence")
            started = time.perf_counter()
            offered = accept_suggestion(
                frozen,
                fragment,
                predict(model, tokenizer, sampler, stream_generate, fragment, frozen),
            ) or ""
            mode = "next" if offered.startswith("\n") else "sentence"
            rewritten = offered[1:].lstrip() if mode == "next" else offered
            ms = round((time.perf_counter() - started) * 1000, 1)
            status = f"+{ms}ms" if rewritten else f"+{ms}ms rejected"
            set_state(
                state,
                fragment=fragment,
                suggestion=rewritten,
                status=status,
                mode=mode,
            )
            stamp = datetime.now().isoformat(timespec="seconds")
            with LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(f"[{stamp}] +{ms}ms\n  now: {fragment!r}\n  tab: {rewritten!r}\n")

        stop.wait(0.05)


def read_key(timeout: float) -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return None
    return sys.stdin.read(1)


def run_overlay(model: Any, tokenizer: Any, sampler: Any, stream_generate: Callable[..., Any]) -> None:
    state = OverlayState(status="watching input.txt")
    stop = threading.Event()
    watcher = threading.Thread(
        target=watch_file,
        args=(state, model, tokenizer, sampler, stream_generate, stop),
        daemon=True,
    )
    watcher.start()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        print("Focus this terminal. Edit input.txt, wrap a span in [[ ]] to refine it, then Tab to apply.")
        render(state)
        while True:
            with state.lock:
                dirty = state.dirty
            if dirty:
                render(state)
            key = read_key(0.05)
            if key is None:
                continue
            if key == "\x03":
                raise KeyboardInterrupt
            if key == "\x1b":
                set_state(state, suggestion="", status="cleared")
                render(state)
                continue
            if key != "\t":
                continue
            with state.lock:
                suggestion = state.suggestion
                mode = state.mode
            if not suggestion:
                set_state(state, status="nothing to apply")
                render(state)
                continue
            if mode == "selection" and last_selection(
                INPUT_FILE.read_text(encoding="utf-8") if INPUT_FILE.exists() else ""
            ) is None:
                set_state(state, status="selection gone")
                render(state)
                continue
            apply_fragment(
                INPUT_FILE,
                "\n" + suggestion if mode == "next" else suggestion,
                mode,
            )
            set_state(
                state,
                fragment=suggestion,
                suggestion="",
                status="applied selection" if mode == "selection" else "applied sentence",
                mode="sentence",
            )
            render(state)
            with LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(f"applied {mode}: {suggestion!r}\n")
    finally:
        stop.set()
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write("\n")
        sys.stdout.flush()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch input.txt and suggest a last-line rewrite.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ALIAS,
        help=(
            "Model alias or Hugging Face / MLX repo id. "
            "Aliases: minicpm (default, MiniCPM5-1B 4-bit), granite, graphite, "
            "smol, granite-8bit, granite-4bit."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    model_id = resolve_model(args.model)
    if not INPUT_FILE.exists():
        INPUT_FILE.write_text("", encoding="utf-8")
    LOG_FILE.touch(exist_ok=True)

    log(f"=== session start {datetime.now(timezone.utc).isoformat()} ===")
    log(f"Watching: {INPUT_FILE}")
    log(f"Model: {model_id}")
    log("Mode: continue the file (no chat template)")

    model, tokenizer, sampler, stream_generate = load_model(model_id)
    log("Model ready.")

    if not sys.stdin.isatty():
        raise SystemExit("Run this in a terminal so Tab can apply the rewrite.")

    run_overlay(model, tokenizer, sampler, stream_generate)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
        try:
            with LOG_FILE.open("a", encoding="utf-8") as f:
                f.write("=== session stop ===\n")
        except OSError:
            pass
        raise SystemExit(0)
