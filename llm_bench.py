#!/usr/bin/env python3
"""Small OpenAI-compatible LLM inference benchmark.

Targets vLLM and SGLang servers that expose the OpenAI Chat Completions API.
Uses only the Python standard library for benchmarking. Plotting is optional
and uses matplotlib when available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
MODELS_PATH = "/v1/models"


@dataclass
class RequestResult:
    request_id: str
    ok: bool
    status_code: int | None
    error: str | None
    start_s: float
    completed_s: float
    latency_s: float
    ttft_s: float | None
    decode_s: float | None
    tpot_s: float | None
    output_tokens: int
    token_count_approx: bool
    tokens_per_second: float | None
    output_chars: int
    output_text: str


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be 0 or greater")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def positive_int_list(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError("must be a comma-separated list of positive integers")

    parsed: list[int] = []
    for part in parts:
        parsed.append(positive_int(part))
    return parsed


def normalize_base_url(raw_url: str) -> str:
    raw_url = raw_url.strip()
    parsed = urllib.parse.urlsplit(raw_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("--url must include a scheme and host, for example http://localhost:8000")

    path = parsed.path.rstrip("/")
    if path == "/v1":
        path = ""
    elif path.endswith("/v1"):
        path = path[:-3].rstrip("/")

    normalized = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
    return normalized.rstrip("/")


def make_api_url(base_url: str, path: str) -> str:
    return f"{base_url}{path}"


def make_headers(api_key: str | None, json_body: bool = True) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if json_body:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", "replace")
    except Exception:
        body = ""
    body = body.strip()
    if len(body) > 500:
        body = body[:500] + "..."
    return body


def describe_exception(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        body = read_error_body(exc)
        if body:
            return f"HTTP {exc.code}: {body}"
        return f"HTTP {exc.code}: {exc.reason}"
    if isinstance(exc, urllib.error.URLError):
        return f"connection error: {exc.reason}"
    if isinstance(exc, socket.timeout):
        return "request timed out"
    return str(exc) or exc.__class__.__name__


def request_json(method: str, url: str, api_key: str | None, timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        method=method,
        headers=make_headers(api_key, json_body=False),
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = getattr(resp, "status", 200)
        if status < 200 or status >= 300:
            raise RuntimeError(f"HTTP {status}")
        data = resp.read()
    return json.loads(data.decode("utf-8"))


def detect_model(base_url: str, api_key: str | None, timeout: float) -> str:
    url = make_api_url(base_url, MODELS_PATH)
    try:
        payload = request_json("GET", url, api_key, timeout)
    except Exception as exc:
        raise RuntimeError(
            f"could not fetch {url}. Provide --model explicitly or check that the server is reachable. "
            f"Details: {describe_exception(exc)}"
        ) from exc

    models = payload.get("data")
    if not isinstance(models, list) or not models:
        raise RuntimeError(f"{url} returned no models. Provide --model explicitly.")

    first = models[0]
    if not isinstance(first, dict) or not first.get("id"):
        raise RuntimeError(f"{url} returned an unexpected model payload. Provide --model explicitly.")
    return str(first["id"])


def synthetic_prompt(word_count: int) -> str:
    words = [
        "latency",
        "throughput",
        "benchmark",
        "inference",
        "server",
        "tokens",
        "request",
        "response",
        "deterministic",
        "measurement",
        "compute",
        "memory",
        "scheduler",
        "cache",
        "batch",
        "decode",
    ]
    body = " ".join(words[i % len(words)] for i in range(max(1, word_count)))
    return (
        "You are helping measure an LLM inference server. Read the following "
        "synthetic workload text and then answer with one concise paragraph.\n\n"
        f"{body}\n\n"
        "Summarize the workload in plain language."
    )


def estimate_output_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text.split()))


def parse_stream_chunk(raw_data: str) -> tuple[str, int | None]:
    chunk = json.loads(raw_data)
    text_parts: list[str] = []
    usage_tokens: int | None = None

    usage = chunk.get("usage")
    if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
        try:
            usage_tokens = int(usage["completion_tokens"])
        except (TypeError, ValueError):
            usage_tokens = None

    choices = chunk.get("choices") or []
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str) and content:
                    text_parts.append(content)

    return "".join(text_parts), usage_tokens


def completion_tokens_from_payload(payload: dict[str, Any]) -> int | None:
    usage = payload.get("usage")
    if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
        try:
            return int(usage["completion_tokens"])
        except (TypeError, ValueError):
            return None
    return None


def message_text_from_payload(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not isinstance(choices, list) or not choices:
        return ""

    first = choices[0]
    if not isinstance(first, dict):
        return ""

    message = first.get("message") or {}
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]

    text = first.get("text")
    if isinstance(text, str):
        return text
    return ""


def build_chat_body(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stream: bool,
) -> bytes:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    return json.dumps(payload).encode("utf-8")


def send_chat_request(
    request_id: str,
    base_url: str,
    api_key: str | None,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stream: bool,
    timeout: float,
    benchmark_started: float,
) -> RequestResult:
    url = make_api_url(base_url, CHAT_COMPLETIONS_PATH)
    body = build_chat_body(model, prompt, max_tokens, temperature, stream)
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers=make_headers(api_key, json_body=True),
    )

    start = time.perf_counter()
    status_code: int | None = None
    ttft_s: float | None = None
    output_parts: list[str] = []
    usage_tokens: int | None = None

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status_code = getattr(resp, "status", 200)
            if status_code < 200 or status_code >= 300:
                raise RuntimeError(f"HTTP {status_code}")

            if stream:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", "replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        text, chunk_tokens = parse_stream_chunk(data)
                    except json.JSONDecodeError:
                        continue
                    if chunk_tokens is not None:
                        usage_tokens = chunk_tokens
                    if text:
                        if ttft_s is None:
                            ttft_s = time.perf_counter() - start
                        output_parts.append(text)
            else:
                payload = json.loads(resp.read().decode("utf-8"))
                output_parts.append(message_text_from_payload(payload))
                usage_tokens = completion_tokens_from_payload(payload)

        completed = time.perf_counter()
        output_text = "".join(output_parts)
        approx = usage_tokens is None
        output_tokens = usage_tokens if usage_tokens is not None else estimate_output_tokens(output_text)
        latency_s = completed - start
        decode_s = None if ttft_s is None else max(0.0, latency_s - ttft_s)
        tpot_s = None
        if decode_s is not None and output_tokens > 0:
            tpot_s = decode_s / output_tokens
        tokens_per_second = output_tokens / latency_s if latency_s > 0 else None
        return RequestResult(
            request_id=request_id,
            ok=True,
            status_code=status_code,
            error=None,
            start_s=start - benchmark_started,
            completed_s=completed - benchmark_started,
            latency_s=latency_s,
            ttft_s=ttft_s,
            decode_s=decode_s,
            tpot_s=tpot_s,
            output_tokens=output_tokens,
            token_count_approx=approx,
            tokens_per_second=tokens_per_second,
            output_chars=len(output_text),
            output_text=output_text,
        )
    except Exception as exc:
        completed = time.perf_counter()
        latency_s = completed - start
        if isinstance(exc, urllib.error.HTTPError):
            status_code = exc.code
        return RequestResult(
            request_id=request_id,
            ok=False,
            status_code=status_code,
            error=describe_exception(exc),
            start_s=start - benchmark_started,
            completed_s=completed - benchmark_started,
            latency_s=latency_s,
            ttft_s=ttft_s,
            decode_s=None,
            tpot_s=None,
            output_tokens=0,
            token_count_approx=False,
            tokens_per_second=None,
            output_chars=0,
            output_text="",
        )


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def stats(values: list[float]) -> dict[str, float | int | None]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    return {
        "count": len(clean),
        "min": min(clean),
        "max": max(clean),
        "mean": mean(clean),
        "p50": percentile(clean, 50),
        "p90": percentile(clean, 90),
        "p95": percentile(clean, 95),
        "p99": percentile(clean, 99),
    }


def summarize_results(
    results: list[RequestResult],
    duration_s: float,
    server: str,
    model: str,
    concurrency: int,
    stream: bool,
) -> dict[str, Any]:
    total = len(results)
    successful = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    total_output_tokens = sum(r.output_tokens for r in successful)
    error_counts: dict[str, int] = {}
    for result in failed:
        message = result.error or "unknown error"
        error_counts[message] = error_counts.get(message, 0) + 1

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "server": server,
        "endpoint": make_api_url(server, CHAT_COMPLETIONS_PATH),
        "model": model,
        "stream": stream,
        "total_requests": total,
        "concurrency": concurrency,
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "error_rate": (len(failed) / total) if total else 0.0,
        "errors": dict(sorted(error_counts.items(), key=lambda item: item[1], reverse=True)),
        "duration_s": duration_s,
        "requests_per_second": (len(successful) / duration_s) if duration_s > 0 else 0.0,
        "total_output_tokens": total_output_tokens,
        "output_tokens_per_second": (total_output_tokens / duration_s) if duration_s > 0 else 0.0,
        "token_counts_approximate": any(r.token_count_approx for r in successful),
        "stats": {
            "ttft_s": stats([r.ttft_s for r in successful if r.ttft_s is not None]),
            "latency_s": stats([r.latency_s for r in successful]),
            "decode_s": stats([r.decode_s for r in successful if r.decode_s is not None]),
            "tpot_s": stats([r.tpot_s for r in successful if r.tpot_s is not None]),
            "output_tokens": stats([float(r.output_tokens) for r in successful]),
            "tokens_per_second_per_request": stats(
                [r.tokens_per_second for r in successful if r.tokens_per_second is not None]
            ),
        },
    }
    return summary


def fmt_float(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f} s"


def fmt_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def stat_line(label: str, block: dict[str, Any], suffix: str = "") -> str:
    values = [block.get("p50"), block.get("p90"), block.get("p95"), block.get("p99")]
    if suffix == "s":
        rendered = " / ".join(fmt_seconds(v) for v in values)
    else:
        rendered = " / ".join(fmt_float(v) for v in values)
    return f"{label} p50 / p90 / p95 / p99: {rendered}"


def min_mean_max_line(label: str, block: dict[str, Any], suffix: str = "") -> str:
    values = [block.get("min"), block.get("mean"), block.get("max")]
    if suffix == "s":
        rendered = " / ".join(fmt_seconds(v) for v in values)
    else:
        rendered = " / ".join(fmt_float(v) for v in values)
    return f"{label} min / mean / max: {rendered}"


def print_benchmark_summary(summary: dict[str, Any]) -> None:
    stats_by_name = summary["stats"]
    approx_note = " (some counts approximate)" if summary["token_counts_approximate"] else ""

    print()
    print("Benchmark Summary")
    print("-----------------")
    print(f"Server: {summary['server']}")
    print(f"Model: {summary['model']}")
    print(f"Requests: {summary['total_requests']}")
    print(f"Concurrency: {summary['concurrency']}")
    print(f"Successful: {summary['successful_requests']}")
    print(f"Failed: {summary['failed_requests']}")
    print(f"Error rate: {summary['error_rate'] * 100:.2f}%")
    print(f"Duration: {fmt_seconds(summary['duration_s'])}")
    print()
    print("Throughput")
    print("----------")
    print(f"Requests/sec: {fmt_rate(summary['requests_per_second'])}")
    print(f"Output tokens/sec: {fmt_rate(summary['output_tokens_per_second'])}")
    print(f"Total output tokens: {summary['total_output_tokens']}{approx_note}")
    print()
    print("Latency")
    print("-------")
    print(stat_line("TTFT", stats_by_name["ttft_s"], suffix="s"))
    print(stat_line("E2E", stats_by_name["latency_s"], suffix="s"))
    print(stat_line("TPOT", stats_by_name["tpot_s"], suffix="s"))
    print(min_mean_max_line("TTFT", stats_by_name["ttft_s"], suffix="s"))
    print(min_mean_max_line("E2E", stats_by_name["latency_s"], suffix="s"))
    print(min_mean_max_line("Output tokens", stats_by_name["output_tokens"]))
    print(min_mean_max_line("Tokens/sec per request", stats_by_name["tokens_per_second_per_request"]))
    if not summary["stream"]:
        print()
        print("TTFT and TPOT are only available in streaming mode.")
    if summary.get("errors"):
        print()
        print("Errors")
        print("------")
        for error, count in list(summary["errors"].items())[:5]:
            print(f"{count}x {error}")


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def result_sort_key(result: RequestResult) -> tuple[tuple[int, int | str], ...]:
    parts = result.request_id.replace("-", " ").split()
    if not parts:
        return ((1, result.request_id),)
    key: list[tuple[int, int | str]] = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part))
    return tuple(key)


def save_json(path: str, summary: dict[str, Any], results: list[RequestResult]) -> None:
    payload = {
        "summary": summary,
        "results": [asdict(result) for result in sorted(results, key=result_sort_key)],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def save_csv(path: str, results: list[RequestResult]) -> None:
    fieldnames = list(RequestResult.__dataclass_fields__.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted(results, key=result_sort_key):
            writer.writerow(asdict(result))


def save_sweep_json(path: str, summary: dict[str, Any], runs: list[tuple[dict[str, Any], list[RequestResult]]]) -> None:
    payload = {
        "summary": summary,
        "runs": [
            {
                "summary": run_summary,
                "results": [asdict(result) for result in sorted(run_results, key=result_sort_key)],
            }
            for run_summary, run_results in runs
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def save_sweep_csv(path: str, runs: list[tuple[dict[str, Any], list[RequestResult]]]) -> None:
    fieldnames = ["concurrency"] + list(RequestResult.__dataclass_fields__.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run_summary, run_results in runs:
            concurrency = run_summary.get("concurrency")
            for result in sorted(run_results, key=result_sort_key):
                row = asdict(result)
                row["concurrency"] = concurrency
                writer.writerow(row)


def maybe_save_outputs(args: argparse.Namespace, summary: dict[str, Any], results: list[RequestResult]) -> None:
    if args.save_json or args.save_csv or args.plot:
        ensure_output_dir(args.out)

    if args.save_json:
        path = os.path.join(args.out, "summary.json")
        save_json(path, summary, results)
        print(f"Saved JSON: {path}")

    if args.save_csv:
        path = os.path.join(args.out, "results.csv")
        save_csv(path, results)
        print(f"Saved CSV: {path}")

    if args.plot:
        save_plots(args.out, summary, results)


def maybe_save_sweep_outputs(
    args: argparse.Namespace,
    summary: dict[str, Any],
    runs: list[tuple[dict[str, Any], list[RequestResult]]],
) -> None:
    if args.save_json or args.save_csv or args.plot:
        ensure_output_dir(args.out)

    if args.save_json:
        path = os.path.join(args.out, "summary.json")
        save_sweep_json(path, summary, runs)
        print(f"Saved JSON: {path}")

    if args.save_csv:
        path = os.path.join(args.out, "results.csv")
        save_sweep_csv(path, runs)
        print(f"Saved CSV: {path}")

    if args.plot:
        save_plots(args.out, summary)


def configure_matplotlib() -> Any | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots.")
        return None

    plt.rcParams.update(
        {
            "figure.figsize": (8.5, 5),
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "normal",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )
    return plt


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def seconds_stat_to_ms(summary: dict[str, Any], metric: str, stat_name: str) -> float | None:
    value = summary.get("stats", {}).get(metric, {}).get(stat_name)
    if not is_finite_number(value):
        return None
    return float(value) * 1000.0


def first_finite_metric(summary: dict[str, Any], names: list[str]) -> float | None:
    for name in names:
        value = summary.get(name)
        if is_finite_number(value):
            return float(value)
    return None


def benchmark_summary_to_plot_row(summary: dict[str, Any]) -> dict[str, float | None] | None:
    concurrency = summary.get("concurrency")
    if not is_finite_number(concurrency):
        return None

    return {
        "max_concurrency": float(concurrency),
        "request_throughput": first_finite_metric(summary, ["requests_per_second"]),
        "output_token_throughput": first_finite_metric(summary, ["output_tokens_per_second"]),
        "total_token_throughput": first_finite_metric(
            summary,
            [
                "total_tokens_per_second",
                "tokens_per_second",
                "all_tokens_per_second",
            ],
        ),
        "mean_ttft_ms": seconds_stat_to_ms(summary, "ttft_s", "mean"),
        "p99_ttft_ms": seconds_stat_to_ms(summary, "ttft_s", "p99"),
        "mean_tpot_ms": seconds_stat_to_ms(summary, "tpot_s", "mean"),
        "p99_tpot_ms": seconds_stat_to_ms(summary, "tpot_s", "p99"),
    }


def benchmark_plot_rows(summary: dict[str, Any]) -> list[dict[str, float | None]]:
    if summary.get("kind") == "concurrency_sweep":
        rows = [
            row
            for run_summary in summary.get("runs", [])
            if isinstance(run_summary, dict)
            for row in [benchmark_summary_to_plot_row(run_summary)]
            if row is not None
        ]
        rows.sort(key=lambda row: float(row["max_concurrency"] or 0.0))
        return rows

    row = benchmark_summary_to_plot_row(summary)
    return [] if row is None else [row]


def plot_line_metric(
    plt: Any,
    rows: list[dict[str, float | None]],
    x_col: str,
    y_series: list[tuple[str, str]],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str,
) -> str | None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    plotted_count = 0
    x_ticks: set[float] = set()

    for label, column in y_series:
        points = [
            (float(row[x_col]), float(row[column]))
            for row in rows
            if is_finite_number(row.get(x_col)) and is_finite_number(row.get(column))
        ]
        if not points:
            continue

        points.sort(key=lambda item: item[0])
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        x_ticks.update(x_values)
        ax.plot(x_values, y_values, marker="o", linewidth=2, label=label)
        plotted_count += 1

    if plotted_count == 0:
        plt.close(fig)
        return None

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if 0 < len(x_ticks) <= 12:
        ax.set_xticks(sorted(x_ticks))
    if plotted_count > 1:
        ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_plots(output_dir: str, summary: dict[str, Any], results: list[RequestResult] | None = None) -> None:
    plt = configure_matplotlib()
    if plt is None:
        return

    if results is not None and not any(r.ok for r in results):
        print("No successful requests to plot, skipping plots.")
        return

    rows = benchmark_plot_rows(summary)
    if not rows:
        print("No benchmark summary metrics found to plot, skipping plots.")
        return

    saved: list[str] = []
    for path in [
        plot_line_metric(
            plt,
            rows,
            "max_concurrency",
            [("request throughput", "request_throughput")],
            "vLLM Request Throughput — Higher is better",
            "Max concurrency",
            "Requests / second",
            os.path.join(output_dir, "request_throughput.png"),
        ),
        plot_line_metric(
            plt,
            rows,
            "max_concurrency",
            [
                ("output tokens / second", "output_token_throughput"),
                ("total tokens / second", "total_token_throughput"),
            ],
            "vLLM Token Throughput — Higher is better",
            "Max concurrency",
            "Tokens / second",
            os.path.join(output_dir, "token_throughput.png"),
        ),
        plot_line_metric(
            plt,
            rows,
            "max_concurrency",
            [
                ("mean TTFT", "mean_ttft_ms"),
                ("p99 TTFT", "p99_ttft_ms"),
            ],
            "vLLM Time To First Token — Lower is better",
            "Max concurrency",
            "Latency (ms)",
            os.path.join(output_dir, "ttft_latency.png"),
        ),
        plot_line_metric(
            plt,
            rows,
            "max_concurrency",
            [
                ("mean TPOT", "mean_tpot_ms"),
                ("p99 TPOT", "p99_tpot_ms"),
            ],
            "vLLM Time Per Output Token — Lower is better",
            "Max concurrency",
            "Latency (ms)",
            os.path.join(output_dir, "tpot_latency.png"),
        ),
    ]:
        if path is not None:
            saved.append(path)

    if not saved:
        print("No plottable metrics found, skipping plots.")
        return

    print("Saved plots:")
    for path in saved:
        print(f"  {path}")


def run_warmup(args: argparse.Namespace, base_url: str, model: str, prompt: str) -> None:
    if args.warmup <= 0:
        return
    print(f"Running {args.warmup} warmup request(s)...")
    started = time.perf_counter()
    for i in range(args.warmup):
        result = send_chat_request(
            request_id=f"warmup-{i + 1}",
            base_url=base_url,
            api_key=args.api_key,
            model=model,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=args.stream,
            timeout=args.timeout,
            benchmark_started=started,
        )
        if not result.ok:
            print(f"Warmup request {i + 1} failed: {result.error}", file=sys.stderr)


def run_benchmark(args: argparse.Namespace, base_url: str, model: str) -> tuple[dict[str, Any], list[RequestResult]]:
    prompt = args.prompt if args.prompt is not None else synthetic_prompt(args.prompt_len)
    run_warmup(args, base_url, model, prompt)

    print(f"Running benchmark: {args.requests} request(s), concurrency {args.concurrency}...")
    benchmark_started = time.perf_counter()
    results: list[RequestResult] = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                send_chat_request,
                str(i + 1),
                base_url,
                args.api_key,
                model,
                prompt,
                args.max_tokens,
                args.temperature,
                args.stream,
                args.timeout,
                benchmark_started,
            )
            for i in range(args.requests)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    duration_s = time.perf_counter() - benchmark_started
    results.sort(key=lambda r: int(r.request_id))
    summary = summarize_results(results, duration_s, base_url, model, args.concurrency, args.stream)
    return summary, results


def run_concurrency_sweep(
    args: argparse.Namespace,
    base_url: str,
    model: str,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], list[RequestResult]]]]:
    concurrency_values = args.concurrency_sweep or []
    print(f"Running concurrency sweep: {', '.join(str(value) for value in concurrency_values)}")
    started = time.perf_counter()
    runs: list[tuple[dict[str, Any], list[RequestResult]]] = []

    for index, concurrency in enumerate(concurrency_values, start=1):
        print()
        print(f"Sweep {index}/{len(concurrency_values)}: concurrency {concurrency}")
        run_args = argparse.Namespace(**vars(args))
        run_args.concurrency = concurrency
        run_args.concurrency_sweep = None
        run_summary, run_results = run_benchmark(run_args, base_url, model)
        runs.append((run_summary, run_results))

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "concurrency_sweep",
        "server": base_url,
        "endpoint": make_api_url(base_url, CHAT_COMPLETIONS_PATH),
        "model": model,
        "stream": args.stream,
        "requests_per_concurrency": args.requests,
        "warmup_per_concurrency": args.warmup,
        "prompt_len": None if args.prompt is not None else args.prompt_len,
        "custom_prompt": args.prompt is not None,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "concurrency_sweep": concurrency_values,
        "duration_s": time.perf_counter() - started,
        "runs": [run_summary for run_summary, _ in runs],
    }
    return summary, runs


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def print_sweep_summary(summary: dict[str, Any]) -> None:
    print()
    print("Concurrency Sweep Summary")
    print("-------------------------")
    print(f"Server: {summary['server']}")
    print(f"Model: {summary['model']}")
    print(f"Concurrency values: {', '.join(str(value) for value in summary['concurrency_sweep'])}")
    print(f"Requests per concurrency: {summary['requests_per_concurrency']}")
    print(f"Duration: {fmt_seconds(summary['duration_s'])}")
    print()
    print("conc | req/s | out tok/s | mean TTFT ms | p99 TTFT ms | mean TPOT ms | p99 TPOT ms | error %")
    print("-----+-------+-----------+--------------+-------------+--------------+-------------+--------")
    for run_summary in summary["runs"]:
        ttft_mean_ms = seconds_stat_to_ms(run_summary, "ttft_s", "mean")
        ttft_p99_ms = seconds_stat_to_ms(run_summary, "ttft_s", "p99")
        tpot_mean_ms = seconds_stat_to_ms(run_summary, "tpot_s", "mean")
        tpot_p99_ms = seconds_stat_to_ms(run_summary, "tpot_s", "p99")
        print(
            f"{run_summary['concurrency']:>4} | "
            f"{fmt_rate(run_summary['requests_per_second']):>5} | "
            f"{fmt_rate(run_summary['output_tokens_per_second']):>9} | "
            f"{fmt_ms(ttft_mean_ms):>12} | "
            f"{fmt_ms(ttft_p99_ms):>11} | "
            f"{fmt_ms(tpot_mean_ms):>12} | "
            f"{fmt_ms(tpot_p99_ms):>11} | "
            f"{run_summary['error_rate'] * 100:>6.2f}"
        )


def isolation_mismatch(output_text: str, expected: str, other: str) -> bool:
    upper = output_text.upper()
    return expected not in upper or other in upper


def print_isolation_summary(summary: dict[str, Any]) -> None:
    print()
    print("Isolation Test Summary")
    print("----------------------")
    print(f"Server: {summary['server']}")
    print(f"Model: {summary['model']}")
    print(f"Pairs: {summary['pairs']}")
    print(f"Total checks: {summary['total_checks']}")
    print(f"Successful checks: {summary['successful_checks']}")
    print(f"Failed checks: {summary['failed_checks']}")
    print(f"Mismatches: {summary['mismatches']}")
    print(f"Mismatch rate: {summary['mismatch_rate'] * 100:.2f}%")
    print(f"Failure rate: {summary['failure_rate'] * 100:.2f}%")
    print()
    print("This is a smoke test only, not proof of KV cache correctness.")


def run_isolation_test(args: argparse.Namespace, base_url: str, model: str) -> tuple[dict[str, Any], list[RequestResult]]:
    prompt_a = "The secret color is BLUE. Answer only the secret color."
    prompt_b = "The secret color is RED. Answer only the secret color."
    pairs = args.requests
    worker_count = max(2, args.concurrency)
    benchmark_started = time.perf_counter()
    labeled_results: list[tuple[str, str, str, RequestResult]] = []

    print(f"Running isolation test: {pairs} pair(s), concurrency {worker_count}...")
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = []
        for i in range(pairs):
            futures.append(
                (
                    "BLUE",
                    "RED",
                    executor.submit(
                        send_chat_request,
                        f"{i + 1:04d}-A",
                        base_url,
                        args.api_key,
                        model,
                        prompt_a,
                        args.max_tokens,
                        args.temperature,
                        args.stream,
                        args.timeout,
                        benchmark_started,
                    ),
                )
            )
            futures.append(
                (
                    "RED",
                    "BLUE",
                    executor.submit(
                        send_chat_request,
                        f"{i + 1:04d}-B",
                        base_url,
                        args.api_key,
                        model,
                        prompt_b,
                        args.max_tokens,
                        args.temperature,
                        args.stream,
                        args.timeout,
                        benchmark_started,
                    ),
                )
            )

        future_to_labels = {future: (expected, other) for expected, other, future in futures}
        for future in as_completed(future_to_labels):
            expected, other = future_to_labels[future]
            result = future.result()
            labeled_results.append((expected, other, result.request_id, result))

    results = [item[3] for item in labeled_results]
    total_checks = len(results)
    failed = [r for r in results if not r.ok]
    successful = [r for r in results if r.ok]
    mismatches = 0
    for expected, other, _, result in labeled_results:
        if result.ok and isolation_mismatch(result.output_text, expected, other):
            mismatches += 1

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "isolation_test",
        "server": base_url,
        "endpoint": make_api_url(base_url, CHAT_COMPLETIONS_PATH),
        "model": model,
        "stream": args.stream,
        "pairs": pairs,
        "total_checks": total_checks,
        "successful_checks": len(successful),
        "failed_checks": len(failed),
        "mismatches": mismatches,
        "mismatch_rate": (mismatches / total_checks) if total_checks else 0.0,
        "failure_rate": (len(failed) / total_checks) if total_checks else 0.0,
        "duration_s": time.perf_counter() - benchmark_started,
    }
    results.sort(key=lambda r: r.request_id)
    return summary, results


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark an OpenAI-compatible LLM server with chat completions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", required=True, help="Base server URL, for example http://localhost:8000")
    parser.add_argument("--model", help="Model name. If omitted, the first id from /v1/models is used.")
    parser.add_argument("--api-key", default="EMPTY", help="API key for Authorization: Bearer.")
    parser.add_argument("--requests", type=positive_int, default=50, help="Total benchmark requests.")
    parser.add_argument("--concurrency", type=positive_int, default=5, help="Concurrent benchmark requests.")
    parser.add_argument(
        "--concurrency-sweep",
        type=positive_int_list,
        help="Comma-separated concurrency values to run sequentially, for example 1,2,4,8,16,32.",
    )
    parser.add_argument("--prompt", help="Custom user prompt. If set, --prompt-len is ignored.")
    parser.add_argument("--prompt-len", type=positive_int, default=128, help="Approximate synthetic prompt words/tokens.")
    parser.add_argument("--max-tokens", type=positive_int, default=128, help="Maximum output tokens to request.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True, help="Use streaming responses.")
    parser.add_argument("--timeout", type=positive_float, default=120.0, help="Per-request timeout in seconds.")
    parser.add_argument("--warmup", type=nonnegative_int, default=3, help="Warmup requests before benchmarking.")
    parser.add_argument("--plot", action="store_true", help="Save matplotlib plots.")
    parser.add_argument("--out", default="bench_results", help="Output directory.")
    parser.add_argument("--json", dest="save_json", action=argparse.BooleanOptionalAction, default=True, help="Save summary.json.")
    parser.add_argument("--csv", dest="save_csv", action=argparse.BooleanOptionalAction, default=True, help="Save results.csv.")
    parser.add_argument(
        "--isolation-test",
        action="store_true",
        help="Run a basic concurrent BLUE/RED prompt contamination smoke test.",
    )
    args = parser.parse_args(argv)
    if args.isolation_test and args.concurrency_sweep:
        parser.error("--concurrency-sweep cannot be used with --isolation-test")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    try:
        base_url = normalize_base_url(args.url)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        model = args.model or detect_model(base_url, args.api_key, args.timeout)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Server: {base_url}")
    print(f"Model: {model}")

    if args.isolation_test:
        summary, results = run_isolation_test(args, base_url, model)
        print_isolation_summary(summary)
        maybe_save_outputs(args, summary, results)
    elif args.concurrency_sweep:
        summary, runs = run_concurrency_sweep(args, base_url, model)
        print_sweep_summary(summary)
        maybe_save_sweep_outputs(args, summary, runs)
    else:
        summary, results = run_benchmark(args, base_url, model)
        print_benchmark_summary(summary)
        maybe_save_outputs(args, summary, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
