#!/usr/bin/env python3
"""Write case files and evaluate sentence rewrites with at most 3 workers."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CASES_DIR = HERE / "cases"
REPORT = HERE / "cases_report.jsonl"
MAX_WORKERS = 3
_bundle: tuple[object, object, object, object] | None = None

SOLID_19 = {
    "01_partial_student.txt",
    "03_newline_plan.txt",
    "05_partial_deg.txt",
    "07_weather_tod.txt",
    "08_amaz.txt",
    "09_article_stu.txt",
    "12_same_line_scien.txt",
    "20_bana.txt",
    "22_long_frozen.txt",
    "24_walki.txt",
    "29_crlf.txt",
    "31_double_space_after_period.txt",
    "37_mr_library.txt",
    "38_quotes.txt",
    "41_blank_line_then_partial.txt",
    "42_question_then_partial.txt",
    "44_many_frozen.txt",
    "48_comma_list_end.txt",
    "49_tab_indent.txt",
}


def spec(
    text: str,
    kind: str,
    *,
    should_show: bool,
    accept: list[str] | None = None,
    reject_if: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "text": text,
        "kind": kind,
        "should_show": should_show,
        "accept": accept or [],
        "reject_if": reject_if or [],
    }


CASES: dict[str, dict[str, Any]] = {
    "01_partial_student.txt": spec(
        "hello, I am mayank. I am science stude",
        "open",
        should_show=True,
        accept=["I am a science student", "I am a science student."],
    ),
    "02_complete_knownledge.txt": spec("I am here to get a knownledge.", "complete", should_show=False),
    "03_newline_plan.txt": spec(
        "hello, I am mayank. I am science stude\nI am here to get a knownledge.\nso my plan is to make some ",
        "open",
        should_show=True,
        accept=[
            "so my plan is to make some plans",
            "so my plan is to make some progress",
            "so my plan is to make some money",
            "so my plan is to make some pasta",
            "So my plan is to make some plans",
            "So my plan is to make some progress",
            "So my plan is to make some money",
            "So my plan is to make some pasta",
        ],
    ),
    "04_grammar_tense.txt": spec(
        "he go to school yesterday and learn alot of sciense",
        "open",
        should_show=True,
        accept=[
            "He go to school yesterday and learned a lot of science",
            "he go to school yesterday and learned a lot of science",
            "He go to school yesterday and learn a lot of science",
            "he go to school yesterday and learn a lot of science",
        ],
        reject_if=["went"],
    ),
    "05_partial_deg.txt": spec(
        "I am here to get deg",
        "open",
        should_show=True,
        accept=["I am here to get a knownledge", "I am here to get a knownledge."],
    ),
    "06_short_so.txt": spec(
        "hello, I am mayank. I am science stude\nI am here to get a knownledge.\nso ",
        "open",
        should_show=False,
    ),
    "07_weather_tod.txt": spec(
        "The weather today is",
        "open",
        should_show=True,
        accept=[
            "The weather today is sunny",
            "The weather today is sunny.",
            "The weather today is nice",
            "The weather today is cold",
            "The weather today is warm",
        ],
    ),
    "08_amaz.txt": spec(
        "This is amaz",
        "open",
        should_show=True,
        accept=["This is amazing", "This is amazing."],
    ),
    "09_article_stu.txt": spec(
        "I am science stu",
        "open",
        should_show=True,
        accept=["I am a science student", "I am a science student."],
    ),
    "10_well_kno.txt": spec(
        "It is a well-kno",
        "open",
        should_show=True,
        accept=["It is a well-known", "It is a well-known fact"],
        reject_if=["well-known."],
    ),
    "11_cant.txt": spec(
        "I can'",
        "open",
        should_show=True,
        accept=["I can't", "I cannot", "I can't believe", "I can't wait"],
        reject_if=["can't."],
    ),
    "12_same_line_scien.txt": spec(
        "Hello, I am Mayank. I study scien",
        "open",
        should_show=True,
        accept=["I study science", "I study science."],
    ),
    "13_finished_then_new.txt": spec(
        "I am here to get a knownledge.\nif I am going to be a writ",
        "open",
        should_show=True,
        accept=["if I am going to be a writer", "If I am going to be a writer"],
        reject_if=["be writing"],
    ),
    "14_book_is.txt": spec(
        "the best to write a book is ",
        "open",
        should_show=True,
        accept=[
            "the best to write a book is to start",
            "the best to write a book is to outline",
            "the best to write a book is to write",
            "the best to write a book is practice",
        ],
        reject_if=["hard work"],
    ),
    "15_writer.txt": spec("if I am going to be a writer", "open", should_show=False),
    "16_career.txt": spec(
        "if I am going to be a writer\nand leave my student's caree",
        "open",
        should_show=True,
        accept=["and leave my student's career", "and leave my student's career."],
        reject_if=["student's care."],
    ),
    "17_lectu.txt": spec(
        "I learned alot from this lectu",
        "open",
        should_show=True,
        accept=[
            "I learned a lot from this lecture",
            "I learned a lot from this lecture.",
        ],
    ),
    "18_their_tabl.txt": spec(
        "I put the book over their on the tabl",
        "open",
        should_show=True,
        accept=[
            "I put the book over their on the table",
            "I put the book over there on the table",
        ],
        reject_if=["over the table"],
    ),
    "19_done.txt": spec("Done.", "complete", should_show=False),
    "20_bana.txt": spec(
        "I bought apples, oranges, and bana",
        "open",
        should_show=True,
        accept=["I bought apples, oranges, and bananas", "I bought apples, oranges, and bananas."],
    ),
    "21_caps_stud.txt": spec(
        "HELLO I AM A STUD",
        "open",
        should_show=True,
        accept=["HELLO I AM A student", "HELLO I AM A STUDENT"],
        reject_if=["Hello,"],
    ),
    "22_long_frozen.txt": spec(
        "Hello, I am Mayank. I am a science student. I am here to get a knownledge.\nSo my plan is to make some progr",
        "open",
        should_show=True,
        accept=["So my plan is to make some progress", "So my plan is to make some progress."],
    ),
    "23_sunny_complete.txt": spec("The weather today is sunny.", "complete", should_show=False),
    "24_walki.txt": spec(
        "She was walki",
        "open",
        should_show=True,
        accept=["She was walking", "She was walking."],
    ),
    "25_question_tod.txt": spec(
        "What time is it goin",
        "open",
        should_show=True,
        accept=["What time is it going", "What time is it going?"],
        reject_if=["going."],
    ),
    "26_empty.txt": spec("", "empty", should_show=False),
    "27_spaces_only.txt": spec("    ", "empty", should_show=False),
    "28_newlines_only.txt": spec("\n\n\n", "empty", should_show=False),
    "29_crlf.txt": spec(
        "Hello, I am Mayank.\r\nI study scien",
        "open",
        should_show=True,
        accept=["I study science", "I study science."],
    ),
    "30_trailing_spaces.txt": spec(
        "I am science stude   ",
        "open",
        should_show=True,
        accept=["I am a science student   ", "I am a science student."],
    ),
    "31_double_space_after_period.txt": spec(
        "Hello, I am Mayank.  I study scien",
        "open",
        should_show=True,
        accept=["I study science", "I study science."],
    ),
    "32_question_complete.txt": spec("How are you?", "complete", should_show=False),
    "33_exclaim_complete.txt": spec("This is great!", "complete", should_show=False),
    "34_ellipsis_complete.txt": spec("Wait...", "complete", should_show=False),
    "35_numbers.txt": spec(
        "I have 3 appl",
        "open",
        should_show=True,
        accept=["I have 3 apples", "I have 3 apples."],
        reject_if=["three"],
    ),
    "36_abbreviation_dr.txt": spec(
        "Dr. Smith is wai",
        "open",
        should_show=False,
        reject_if=["awake"],
    ),
    "37_mr_library.txt": spec(
        "Mr. Jones went to the librar",
        "open",
        should_show=True,
        accept=["Mr. Jones went to the library", "Mr. Jones went to the library."],
    ),
    "38_quotes.txt": spec(
        'She said "hello',
        "open",
        should_show=True,
        accept=['She said "hello"', 'She said "hello.', 'She said "hello'],
    ),
    "39_hyphen_end.txt": spec(
        "It is a well-",
        "open",
        should_show=True,
        accept=["It is a well-known", "It is a well-known fact"],
        reject_if=["well-known."],
    ),
    "40_single_letter.txt": spec("I", "open", should_show=False),
    "41_blank_line_then_partial.txt": spec(
        "I am here to get a knownledge.\n\nso my plan is to make some",
        "open",
        should_show=True,
        accept=[
            "so my plan is to make some plans",
            "so my plan is to make some progress",
            "so my plan is to make some money",
            "so my plan is to make some pasta",
        ],
    ),
    "42_question_then_partial.txt": spec(
        "Are you ready? I am waiti",
        "open",
        should_show=True,
        accept=["I am waiting", "I am waiting."],
        reject_if=["shipment"],
    ),
    "43_curly_apostrophe.txt": spec(
        "I can’t belie",
        "open",
        should_show=True,
        accept=["I can’t believe", "I can't believe"],
        reject_if=["believe."],
    ),
    "44_many_frozen.txt": spec(
        "Hello, I am Mayank. I am a science student. I am here to get a knownledge.\n"
        "The weather today is sunny. I bought apples, oranges, and bananas.\n"
        "Now I want to fin",
        "open",
        should_show=True,
        accept=["Now I want to finish", "Now I want to find"],
    ),
    "45_long_open.txt": spec(
        "the best way to write a book is to sit down every morning with tea and start typing even if the first draft is ",
        "open",
        should_show=True,
        accept=[
            "the best way to write a book is to sit down every morning with tea and start typing even if the first draft is bad",
            "the best way to write a book is to sit down every morning with tea and start typing even if the first draft is messy",
            "the best way to write a book is to sit down every morning with tea and start typing even if the first draft is rough",
            "the best way to write a book is to sit down every morning with tea and start typing even if the first draft is terrible",
        ],
    ),
    "46_digits_only.txt": spec("12345", "open", should_show=False),
    "47_url_like.txt": spec("Visit https://example.com/docs/get-star", "open", should_show=False),
    "48_comma_list_end.txt": spec(
        "red, green, blu",
        "open",
        should_show=True,
        accept=["red, green, blue", "red, green, blue."],
    ),
    "49_tab_indent.txt": spec(
        "I am here to get a knownledge.\n\tThen I will start writ",
        "open",
        should_show=True,
        accept=["Then I will start writing", "Then I will start writing."],
        reject_if=["be writing"],
    ),
    "50_space_after_complete.txt": spec("Done.\n ", "empty", should_show=False),
    "51_how_are_you_doin.txt": spec(
        "how are you doin",
        "open",
        should_show=True,
        accept=["how are you doing", "How are you doing"],
        reject_if=["doing."],
    ),
    "52_wait_what.txt": spec(
        "wait what",
        "open",
        should_show=True,
        accept=["wait what happened", "wait what is", "Wait what happened"],
        reject_if=["what."],
    ),
    "53_who_are_you_talki.txt": spec(
        "Who are you talki",
        "open",
        should_show=True,
        accept=["Who are you talking", "Who are you talking to", "Who are you talking about"],
        reject_if=["talking."],
    ),
    "54_name_mayan.txt": spec(
        "my name is Mayan",
        "open",
        should_show=True,
        accept=["my name is Mayank", "My name is Mayank"],
        reject_if=["Mayan."],
    ),
    "55_phone_555.txt": spec(
        "call me at 555-01",
        "open",
        should_show=True,
        accept=["call me at 555-0123", "call me at 555-0100", "Call me at 555-0123"],
        reject_if=["five", "555-01."],
    ),
    "56_iphone_gre.txt": spec(
        "iPhone is gre",
        "open",
        should_show=True,
        accept=["iPhone is great", "iPhone is green"],
        reject_if=["Iphone"],
    ),
    "57_nasa_launche.txt": spec(
        "NASA launche",
        "open",
        should_show=True,
        accept=["NASA launched", "NASA launches", "NASA launcher"],
        reject_if=["Nasa"],
    ),
    "58_emoji_end.txt": spec(
        "I am so happy 😊 and excit",
        "open",
        should_show=True,
        accept=["I am so happy 😊 and excited", "I am so happy 😊 and exciting"],
    ),
    "59_cursor_mid.txt": spec(
        "The ca| sat on the mat",
        "open",
        should_show=True,
        accept=["The cat sat on the mat", "The car sat on the mat"],
    ),
    "60_french_etudia.txt": spec(
        "Je suis étudia",
        "open",
        should_show=True,
        accept=["Je suis étudiant", "Je suis étudiante", "Je suis étudiant."],
    ),
    "61_code_rang.txt": spec(
        "for i in rang",
        "open",
        should_show=True,
        accept=["for i in range", "for i in range(", "for i in range 1 10"],
        reject_if=["rang."],
    ),
    "62_can_you_help_wi.txt": spec(
        "Can you help me wi",
        "open",
        should_show=True,
        accept=["Can you help me with", "Can you help me with that"],
        reject_if=["with."],
    ),
    "63_wow_incred.txt": spec(
        "wow that is incred",
        "open",
        should_show=True,
        accept=["wow that is incredible", "Wow that is incredible"],
        reject_if=["incredible."],
    ),
    "64_email_hello.txt": spec(
        "email me at hello@",
        "open",
        should_show=True,
        accept=["email me at hello@example.com", "email me at hello@gmail.com"],
    ),
    "65_question_nam.txt": spec(
        "What is your nam",
        "open",
        should_show=True,
        accept=["What is your name", "What is your name?"],
        reject_if=["name."],
    ),
    "66_list_inter.txt": spec(
        "The best movie in the world \n1. inter",
        "open",
        should_show=True,
        accept=["1. Interstellar", "1. Interstellar.", "1. interesting"],
    ),
}


def write_cases() -> list[Path]:
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, item in CASES.items():
        path = CASES_DIR / name
        text = item["text"]
        path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def _worker_init(model_id: str) -> None:
    global _bundle
    sys.path.insert(0, str(HERE))
    from predictor import load_model

    _bundle = load_model(model_id)


def evaluate_case(name: str) -> dict[str, object]:
    sys.path.insert(0, str(HERE))
    from predictor import (
        accept_suggestion,
        apply_fragment,
        copies_earlier,
        extra_after_last_word,
        extra_sentence,
        extra_word_count,
        last_word_partial,
        predict,
        score_suggestion,
        split_active_sentence,
        too_small_fragment,
    )

    if _bundle is None:
        raise RuntimeError("worker model was not initialized")
    item = CASES[name]
    text = item["text"]
    kind = item["kind"]
    frozen, fragment = split_active_sentence(text)
    started = time.perf_counter()
    raw = ""
    suggestion = None
    skip_model = not fragment.strip() or too_small_fragment(fragment)
    if not skip_model:
        model, tokenizer, sampler, stream_generate = _bundle
        raw = predict(model, tokenizer, sampler, stream_generate, fragment, frozen)
        suggestion = accept_suggestion(frozen, fragment, raw)
    ms = round((time.perf_counter() - started) * 1000, 1)

    failures: list[str] = []
    warnings: list[str] = []
    applied = None
    if kind == "complete":
        if fragment.strip():
            failures.append("expected empty fragment for a finished sentence")
        if suggestion:
            failures.append("must not suggest after a finished sentence")
    elif kind == "empty":
        if fragment.strip():
            failures.append("expected no fragment for empty input")
        if suggestion:
            failures.append("must not suggest on empty input")
    else:
        if not fragment.strip():
            failures.append("expected an open fragment")
        elif too_small_fragment(fragment):
            if suggestion:
                failures.append("must not suggest on a tiny fragment")
        elif suggestion:
            if extra_sentence(suggestion):
                failures.append("accepted a second sentence")
            if copies_earlier(frozen, suggestion):
                failures.append("accepted a copy of earlier text")
            if extra_word_count(fragment, suggestion) > 8:
                failures.append("accepted more than eight extra words")
            if last_word_partial(fragment) and extra_after_last_word(fragment, suggestion) > 3:
                failures.append("accepted too many words after a partial last word")
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "input.txt"
                path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
                applied = apply_fragment(path, suggestion)
                if not applied.startswith(frozen):
                    failures.append("apply changed frozen prefix")
                spliced = applied[len(frozen) :]
                if spliced.rstrip() != suggestion.rstrip() and spliced != suggestion:
                    failures.append("apply did not splice the suggestion")
        else:
            warnings.append("open sentence produced no suggestion")
        if raw and extra_sentence(raw) and suggestion:
            failures.append("raw extra sentence leaked through")
        if raw and copies_earlier(frozen, raw) and suggestion:
            failures.append("raw echo leaked through")

    bucket = score_suggestion(
        item["should_show"],
        item["accept"],
        item["reject_if"],
        suggestion,
        raw,
    )

    status = "FAIL" if failures else "PASS"
    return {
        "name": name,
        "kind": kind,
        "status": status,
        "ms": ms,
        "frozen": frozen,
        "fragment": fragment,
        "raw": raw,
        "suggestion": suggestion,
        "applied": applied,
        "failures": failures,
        "warnings": warnings,
        "bucket": bucket,
        "should_show": item["should_show"],
    }


def summarize(results: list[dict[str, object]]) -> dict[str, object]:
    counts: dict[str, int] = {}
    for item in results:
        bucket = str(item.get("bucket") or "?")
        counts[bucket] = counts.get(bucket, 0) + 1
    model_ms = [float(item["ms"]) for item in results if float(item.get("ms") or 0) > 0]
    model_ms.sort()

    def pct(p: float) -> float | None:
        if not model_ms:
            return None
        index = min(len(model_ms) - 1, max(0, round((p / 100) * (len(model_ms) - 1))))
        return model_ms[index]

    solid_regressed = [
        str(item["name"])
        for item in results
        if str(item["name"]) in SOLID_19 and item.get("bucket") != "solid"
    ]
    return {
        "counts": counts,
        "solid": counts.get("solid", 0),
        "wrong": counts.get("wrong", 0),
        "hidden_miss": counts.get("hidden-miss", 0),
        "n_model": len(model_ms),
        "p50": pct(50),
        "p95": pct(95),
        "max": model_ms[-1] if model_ms else None,
        "solid_regressed": solid_regressed,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate last-line rewrite cases.")
    parser.add_argument(
        "--model",
        default="granite",
        help="Model alias or repo id (same as predictor.py --model).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(HERE))
    from predictor import resolve_model

    args = parse_args(argv)
    model_id = resolve_model(args.model)
    write_cases()
    names = sorted(CASES)
    print(f"Running {len(names)} cases with {MAX_WORKERS} workers model={model_id}")
    ctx = multiprocessing.get_context("spawn")
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(model_id,),
    ) as pool:
        futures = {pool.submit(evaluate_case, name): name for name in names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "name": name,
                    "kind": CASES[name]["kind"],
                    "status": "FAIL",
                    "ms": 0,
                    "failures": [f"worker crashed: {exc}"],
                    "warnings": [],
                    "traceback": traceback.format_exc(),
                    "bucket": "?",
                }
            results.append(result)
            mark = "FAIL" if result["status"] == "FAIL" else "PASS"
            warn = f" warn={result.get('warnings')}" if result.get("warnings") else ""
            print(f"{mark} {name} +{result.get('ms')}ms [{result.get('bucket')}]{warn}")
            if result.get("failures"):
                print(f"    {result['failures']}")
            if result.get("suggestion"):
                print(f"    tab {result['suggestion']!r}")
            elif result.get("raw"):
                print(f"    raw {result['raw']!r}")

    results.sort(key=lambda item: str(item["name"]))
    REPORT.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results),
        encoding="utf-8",
    )
    failed = [item for item in results if item["status"] != "PASS"]
    stats = summarize(results)
    print("\n| Case | Typed | Got | Bucket |")
    print("|---|---|---|---|")
    for item in results:
        typed = str(item.get("fragment") or item.get("frozen") or "")
        got = item.get("suggestion") or item.get("raw") or ""
        print(
            f"| {item['name']} | {typed[:40]!s} | {str(got)[:40]} | {item.get('bucket', '')} |"
        )
    print(
        f"\n{len(results) - len(failed)} wiring-pass, {len(failed)} failed. "
        f"solid={stats['solid']} wrong={stats['wrong']} hidden-miss={stats['hidden_miss']} "
        f"p95={stats['p95']} max={stats['max']} regressed={stats['solid_regressed']}. "
        f"Report: {REPORT}"
    )
    targets_ok = (
        stats["solid"] >= 24
        and stats["wrong"] <= 2
        and stats["hidden_miss"] <= 1
        and not stats["solid_regressed"]
        and (stats["p95"] is None or float(stats["p95"]) <= 500)
    )
    print(f"targets={'PASS' if targets_ok else 'MISS'} {stats['counts']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
