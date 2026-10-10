from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

from agent.core.settings import Settings
from agent.workflows.auto_review_graph import run_review_workflow

# See README.md for single-paper and batch usage examples.
MARKDOWN_SUFFIXES = {".md", ".markdown"}


def find_markdown_papers(path: Path, *, recursive: bool) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in MARKDOWN_SUFFIXES:
            raise ValueError(f"Input file is not markdown: {path}")
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    iterator = path.rglob("*") if recursive else path.glob("*")
    return sorted(
        candidate
        for candidate in iterator
        if candidate.is_file() and candidate.suffix.lower() in MARKDOWN_SUFFIXES
    )


def has_existing_result(settings: Settings, paper_path: Path) -> bool:
    paper_results_dir = settings.results_root / paper_path.stem / "runs"
    if not paper_results_dir.exists():
        return False
    return any(run_dir.joinpath(f"{paper_path.stem}.json").is_file() for run_dir in paper_results_dir.iterdir())


def run_one_paper(paper_path: Path, settings: Settings, *, batch_label: str | None = None) -> str:
    report = run_review_workflow(str(paper_path), settings=settings, batch_label=batch_label)
    return report.output_path or ""


def process_sequential(
    work_items: list[tuple[int, Path]],
    *,
    settings: Settings,
    continue_on_error: bool,
    total: int,
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    succeeded: list[tuple[Path, str]] = []
    failed: list[tuple[Path, str]] = []

    for index, paper_path in work_items:
        print(f"[START] {index}/{total} {paper_path}")
        try:
            output_path = run_one_paper(paper_path, settings, batch_label=f"{index}/{total}")
        except Exception as exc:
            failed.append((paper_path, f"{type(exc).__name__}: {exc}"))
            print(f"[FAIL] {index}/{total} {paper_path} :: {type(exc).__name__}: {exc}")
            if not continue_on_error:
                raise
            continue

        succeeded.append((paper_path, output_path))
        print(f"[OK] {index}/{total} {paper_path} -> {output_path}")

    return succeeded, failed


def process_concurrent(
    work_items: list[tuple[int, Path]],
    *,
    settings: Settings,
    jobs: int,
    continue_on_error: bool,
    total: int,
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    succeeded: list[tuple[Path, str]] = []
    failed: list[tuple[Path, str]] = []
    max_workers = min(max(1, jobs), len(work_items))
    future_to_item: dict[Future[str], tuple[int, Path]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, paper_path in work_items:
            print(f"[START] {index}/{total} {paper_path}")
            future = executor.submit(run_one_paper, paper_path, settings, batch_label=f"{index}/{total}")
            future_to_item[future] = (index, paper_path)

        for future in as_completed(future_to_item):
            index, paper_path = future_to_item[future]
            try:
                output_path = future.result()
            except Exception as exc:
                failed.append((paper_path, f"{type(exc).__name__}: {exc}"))
                print(f"[FAIL] {index}/{total} {paper_path} :: {type(exc).__name__}: {exc}")
                if not continue_on_error:
                    for pending in future_to_item:
                        pending.cancel()
                    raise
                continue

            succeeded.append((paper_path, output_path))
            print(f"[OK] {index}/{total} {paper_path} -> {output_path}")

    order = {paper_path: index for index, paper_path in work_items}
    succeeded.sort(key=lambda item: order[item[0]])
    failed.sort(key=lambda item: order[item[0]])
    return succeeded, failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the PriorReview auto-review workflow on one markdown paper or a directory of papers."
    )
    parser.add_argument("paper_path", help="Path to a markdown paper, or a directory containing markdown papers.")
    parser.add_argument("--limit", type=int, default=0, help="Max papers to process. 0 means all.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N discovered papers before processing.")
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="When paper_path is a directory, only process markdown files directly under it.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep processing later papers if one paper fails.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip papers that already have at least one final review JSON under the results directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list the markdown papers that would be processed.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of papers to process concurrently. Default: 1.",
    )
    parser.add_argument(
        "--output-dir",
        "--results-dir",
        dest="output_dir",
        help=(
            "Directory where workflow results are written. "
            "Overrides PRIORREVIEW_RESULTS_DIR for this run. Default: PRIORREVIEW_RESULTS_DIR or data/results."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.paper_path).expanduser().resolve()
    papers = find_markdown_papers(input_path, recursive=not args.no_recursive)
    if args.offset < 0:
        raise ValueError("--offset must be >= 0")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")

    papers = papers[args.offset :]
    if args.limit:
        papers = papers[: args.limit]

    if not papers:
        print(f"No markdown papers found under: {input_path}")
        return 0

    print(f"Found {len(papers)} markdown paper(s).")
    if args.dry_run:
        for index, paper_path in enumerate(papers, start=1):
            print(f"[DRY-RUN] {index}/{len(papers)} {paper_path}")
        return 0

    settings_overrides = {}
    if args.output_dir:
        settings_overrides["results_dir"] = str(Path(args.output_dir).expanduser().resolve())
    settings = Settings.from_env(**settings_overrides)
    print(f"Results directory: {settings.results_root.resolve()}")
    skipped: list[Path] = []
    work_items: list[tuple[int, Path]] = []
    for index, paper_path in enumerate(papers, start=1):
        if args.skip_existing and has_existing_result(settings, paper_path):
            skipped.append(paper_path)
            print(f"[SKIP] {index}/{len(papers)} {paper_path}")
            continue
        work_items.append((index, paper_path))

    if not work_items:
        succeeded: list[tuple[Path, str]] = []
        failed: list[tuple[Path, str]] = []
    elif args.jobs == 1:
        succeeded, failed = process_sequential(
            work_items,
            settings=settings,
            continue_on_error=args.continue_on_error,
            total=len(papers),
        )
    else:
        print(f"Running with concurrency: jobs={min(args.jobs, len(work_items))}")
        succeeded, failed = process_concurrent(
            work_items,
            settings=settings,
            jobs=args.jobs,
            continue_on_error=args.continue_on_error,
            total=len(papers),
        )

    print(
        "Batch summary: "
        f"succeeded={len(succeeded)}, skipped={len(skipped)}, failed={len(failed)}, total={len(papers)}"
    )
    if failed:
        print("Failed papers:")
        for paper_path, error in failed:
            print(f"  - {paper_path}: {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
