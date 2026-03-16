#!/usr/bin/env python3
"""Interactive launcher for command snippets stored in markdown files."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import shutil
import subprocess
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, unquote, urlparse


EXCLUDED_SOURCE_NAMES = {"license.md", "agents.md"}
EXCLUDED_SOURCE_DIRS = {".git", "__pycache__", "tricks_web"}
LOCAL_SOURCE_SUFFIX = ".local.md"
EXTRA_SOURCE_FILES = {
    "dotfiles/tmux.conf",
    "dotfiles/skhdrc",
    "dotfiles/yabairc",
    "Installing_WebArena/download_webarena_gitlab.sh",
}
UNSPLIT_SOURCE_FILES = {
    "dotfiles/tmux.conf",
    "dotfiles/skhdrc",
    "dotfiles/yabairc",
    "dotfiles/README.md",
    "Installing_WebArena/download_webarena_gitlab.sh",
}
COMMON_COMMAND_PREFIXES = {
    ".",
    "awk",
    "bash",
    "cat",
    "cd",
    "chmod",
    "chown",
    "conda",
    "cp",
    "curl",
    "docker",
    "echo",
    "export",
    "find",
    "git",
    "grep",
    "kubectl",
    "ls",
    "make",
    "mkdir",
    "mogrify",
    "mv",
    "npm",
    "pip",
    "python",
    "python3",
    "rm",
    "run",
    "rsync",
    "scp",
    "sed",
    "salloc",
    "sbatch",
    "scancel",
    "sh",
    "source",
    "squeue",
    "srun",
    "ssh",
    "sudo",
    "set",
    "set-option",
    "tar",
    "tmux",
    "touch",
    "yabai",
    "wandb",
    "wget",
    "xargs",
    "yarn",
}
WEB_ASSETS_DIR = Path(__file__).resolve().parent / "tricks_web"
WEB_ROUTES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/app.js": "app.js",
    "/styles.css": "styles.css",
}
MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}


@dataclass
class Trick:
    id: int
    source: Path
    heading: str
    note: str
    code: str
    line: int


@dataclass
class Section:
    heading: str
    start_line: int
    lines: list[str]


def discover_default_sources(base_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()
    root_readme = (base_dir / "README.md").resolve()
    for path in base_dir.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = {part.lower() for part in path.relative_to(base_dir).parts}
        if rel_parts.intersection(EXCLUDED_SOURCE_DIRS):
            continue
        if path.name.lower().endswith(LOCAL_SOURCE_SUFFIX):
            continue
        try:
            if path.resolve() == root_readme:
                continue
        except OSError:
            pass
        rel_path_posix = path.relative_to(base_dir).as_posix()
        is_markdown = path.suffix.lower() == ".md"
        is_extra_file = rel_path_posix in EXTRA_SOURCE_FILES
        if not is_markdown and not is_extra_file:
            continue
        if is_markdown and path.name.lower() in EXCLUDED_SOURCE_NAMES:
            continue
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        candidates.append(path)
    candidates.sort(key=lambda item: str(item).lower())
    return candidates


def looks_like_command(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    # Keep scheduler/script directives inside a command block.
    if stripped.startswith("#!/") or stripped.startswith("#SBATCH") or stripped.startswith("#PBS"):
        return True
    if " : " in stripped:
        rhs = stripped.split(" : ", maxsplit=1)[1].strip()
        if rhs and looks_like_command(rhs):
            return True
    if stripped.startswith(("#", "-", "*", ">")):
        return False
    if stripped.lower().startswith(("http://", "https://")):
        return True
    if re.match(r"^[A-Z_][A-Z0-9_]*=.*$", stripped):
        return True
    if stripped.startswith(("/", "./", "../", "~")):
        return True
    if any(op in stripped for op in (" | ", " && ", " || ", ";", "$(")):
        return True

    first = stripped.split()[0]
    first_lower = first.lower()
    if first_lower in COMMON_COMMAND_PREFIXES and first == first_lower:
        return True
    if first_lower.startswith("./"):
        return True
    return False


def split_into_sections(markdown_path: Path, lines: list[str]) -> list[Section]:
    heading_meta: list[tuple[int, str, int]] = []
    for line_no, line in enumerate(lines, start=1):
        match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if match:
            heading_meta.append((len(match.group(1)), match.group(2).strip(), line_no))

    if not heading_meta:
        return [Section(heading=markdown_path.stem, start_line=1, lines=lines)]

    top_level = min(level for level, _, _ in heading_meta)
    sections: list[Section] = []
    current = Section(heading=markdown_path.stem, start_line=1, lines=[])

    for line_no, line in enumerate(lines, start=1):
        match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if match and len(match.group(1)) == top_level:
            if current.lines:
                sections.append(current)
            current = Section(
                heading=match.group(2).strip(),
                start_line=line_no + 1,
                lines=[],
            )
            continue
        current.lines.append(line)

    if current.lines:
        sections.append(current)

    return sections


def should_treat_as_single_trick_source(path: Path) -> bool:
    rel = path.as_posix()
    if rel.endswith(tuple(UNSPLIT_SOURCE_FILES)):
        return True
    # Keep markdown notes simple: one file == one trick.
    if path.suffix.lower() == ".md":
        return True
    return False


def parse_whole_file_as_trick(path: Path, lines: list[str]) -> Trick | None:
    raw = "\n".join(lines).strip()
    if not raw:
        return None
    heading = path.stem if path.suffix.lower() == ".md" else path.name
    return Trick(
        id=0,
        source=path,
        heading=heading,
        note="",
        code=raw,
        line=1,
    )


def parse_section_to_trick(markdown_path: Path, section: Section) -> Trick | None:
    in_code = False
    fence = "```"
    code_buffer: list[str] = []
    code_chunks: list[str] = []
    note_chunks: list[str] = []
    first_code_line: int | None = None

    for offset, line in enumerate(section.lines):
        line_no = section.start_line + offset
        stripped = line.strip()

        if not in_code:
            heading_match = re.match(r"^(#{1,6})\s+(.*)$", line)
            if heading_match:
                nested_heading = heading_match.group(2).strip()
                if nested_heading:
                    note_chunks.append(nested_heading)
                continue

            fence_match = re.match(r"^(`{3,})(.*)$", stripped)
            if fence_match:
                fence = fence_match.group(1)
                trailing = fence_match.group(2).strip()
                if trailing.endswith(fence):
                    inline_code = trailing[: -len(fence)].strip()
                    if inline_code:
                        if first_code_line is None:
                            first_code_line = line_no
                        code_chunks.append(inline_code)
                    continue
                in_code = True
                code_buffer = []
                if first_code_line is None:
                    first_code_line = line_no + 1
                continue

            inline_code_match = re.match(r"^(`{1,3})([^`].*?)\1$", stripped)
            if inline_code_match:
                inline_code = inline_code_match.group(2).strip()
                if looks_like_command(inline_code):
                    if first_code_line is None:
                        first_code_line = line_no
                    code_chunks.append(inline_code)
                    continue

            numbered_line_match = re.match(r"^\d+\.\s+(.*)$", stripped)
            if numbered_line_match:
                maybe_command = numbered_line_match.group(1).strip()
                if looks_like_command(maybe_command):
                    if first_code_line is None:
                        first_code_line = line_no
                    code_chunks.append(maybe_command)
                    continue

            if looks_like_command(stripped):
                if first_code_line is None:
                    first_code_line = line_no
                code_chunks.append(stripped)
                continue

            if stripped:
                note_chunks.append(stripped)
            continue

        if stripped.startswith(fence):
            in_code = False
            code = "\n".join(code_buffer).strip()
            if code:
                code_chunks.append(code)
            code_buffer = []
        else:
            code_buffer.append(line)

    # Gracefully include unfinished fenced blocks instead of dropping content.
    if in_code and code_buffer:
        dangling_code = "\n".join(code_buffer).strip()
        if dangling_code:
            code_chunks.append(dangling_code)

    if not code_chunks:
        return None

    code = "\n\n".join(chunk for chunk in code_chunks if chunk).strip()
    note = "\n".join(note_chunks).strip()
    return Trick(
        id=0,
        source=markdown_path,
        heading=section.heading.strip() if section.heading.strip() else markdown_path.stem,
        note=note,
        code=code,
        line=first_code_line if first_code_line is not None else section.start_line,
    )


def parse_tricks(markdown_path: Path) -> list[Trick]:
    if not markdown_path.exists():
        return []

    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    if should_treat_as_single_trick_source(markdown_path):
        one = parse_whole_file_as_trick(markdown_path, lines)
        return [one] if one is not None else []

    sections = split_into_sections(markdown_path, lines)
    tricks: list[Trick] = []
    for section in sections:
        trick = parse_section_to_trick(markdown_path, section)
        if trick is not None:
            tricks.append(trick)
    return tricks


def collect_tricks(paths: Iterable[Path]) -> list[Trick]:
    entries: list[Trick] = []
    for path in paths:
        entries.extend(parse_tricks(path))

    for idx, entry in enumerate(entries, start=1):
        entry.id = idx

    return entries


def normalized_text(entry: Trick) -> str:
    return " ".join((entry.heading, entry.note, entry.code)).lower()


def search_tricks(entries: list[Trick], query: str) -> list[Trick]:
    query = query.strip().lower()
    if not query:
        return entries

    tokens = [token for token in re.split(r"\s+", query) if token]
    scored: list[tuple[int, Trick]] = []

    for entry in entries:
        text = normalized_text(entry)
        score = 0
        for token in tokens:
            if token in text:
                score += 2
        if query in text:
            score += 3
        if entry.heading.lower().startswith(query):
            score += 2
        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda pair: (-pair[0], pair[1].id))
    return [entry for _, entry in scored]


def print_results(results: list[Trick], limit: int = 15) -> None:
    if not results:
        print("No matching tricks found.")
        return

    shown = results[:limit]
    for item in shown:
        first_line = item.code.splitlines()[0][:80]
        print(f"[{item.id:>3}] {item.heading}")
        print(f"      {first_line}")

    if len(results) > len(shown):
        print(f"... {len(results) - len(shown)} more result(s). Refine your search.")


def show_detail(entry: Trick) -> None:
    print("\n" + "=" * 80)
    print(f"Trick #{entry.id}")
    print(f"Topic:   {entry.heading}")
    print(f"Source:  {entry.source}:{entry.line}")
    if entry.note:
        print(f"Note:    {entry.note}")
    print("Command:")
    print("-" * 80)
    print(entry.code)
    print("-" * 80)


def run_command(command: str) -> None:
    danger_patterns = ["rm -rf", "mkfs", "dd if=", "shutdown", "reboot", "poweroff"]
    lower_cmd = command.lower()
    if any(pattern in lower_cmd for pattern in danger_patterns):
        confirm = input("Potentially destructive command. Type RUN to execute: ").strip()
        if confirm != "RUN":
            print("Canceled.")
            return
    else:
        confirm = input("Run this command now? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Canceled.")
            return

    print("\nExecuting...\n")
    subprocess.run(command, shell=True, check=False)


def copy_to_clipboard(command: str) -> bool:
    clipboard_candidates = [
        ["pbcopy"],
        ["xclip", "-selection", "clipboard"],
        ["wl-copy"],
    ]
    for candidate in clipboard_candidates:
        if shutil.which(candidate[0]) is None:
            continue
        try:
            subprocess.run(candidate, input=command.encode("utf-8"), check=True)
            return True
        except subprocess.SubprocessError:
            continue
    return False


def interactive(entries: list[Trick]) -> int:
    print(f"Loaded {len(entries)} tricks.")
    print("Type a search query, or :q to quit.")

    while True:
        query = input("\nsearch> ").strip()
        if query in {":q", "q", "quit", "exit"}:
            return 0
        if query == ":list":
            print_results(entries)
            continue

        results = search_tricks(entries, query)
        print_results(results)
        if not results:
            continue

        while True:
            choice = input("open id (Enter=new search, q=quit): ").strip().lower()
            if choice in {"", "n", "new"}:
                break
            if choice in {"q", "quit", "exit"}:
                return 0
            if not choice.isdigit():
                print("Please enter a valid numeric id.")
                continue

            target_id = int(choice)
            selected = next((item for item in results if item.id == target_id), None)
            if selected is None:
                print("Id not in current result set.")
                continue

            show_detail(selected)
            while True:
                action = input("[c]opy [r]un [b]ack [q]uit: ").strip().lower()
                if action in {"b", "back", ""}:
                    break
                if action in {"q", "quit", "exit"}:
                    return 0
                if action in {"c", "copy"}:
                    copied = copy_to_clipboard(selected.code)
                    if copied:
                        print("Copied command to clipboard.")
                    else:
                        print("Clipboard tool not found (tried pbcopy/xclip/wl-copy).")
                elif action in {"r", "run"}:
                    run_command(selected.code)
                else:
                    print("Use c, r, b, or q.")


def parse_source_args(raw_sources: list[str]) -> list[Path]:
    paths: list[Path] = []
    for source in raw_sources:
        candidate = Path(source).expanduser()
        try:
            paths.append(candidate.resolve())
        except OSError:
            paths.append(candidate)
    return paths


def source_rel_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def source_group(path: Path, base_dir: Path) -> str:
    rel = source_rel_path(path, base_dir)
    if "/" not in rel:
        return "root"
    return rel.split("/", maxsplit=1)[0]


def source_display_name(path: Path) -> str:
    stem = path.stem.replace("_", " ").replace("-", " ").strip()
    if not stem:
        return path.name
    return stem


def trick_to_payload(entry: Trick, base_dir: Path) -> dict[str, object]:
    rel = source_rel_path(entry.source, base_dir)
    return {
        "id": entry.id,
        "source": str(entry.source),
        "source_rel": rel,
        "source_group": source_group(entry.source, base_dir),
        "source_name": entry.source.name,
        "source_display": source_display_name(entry.source),
        "heading": entry.heading,
        "note": entry.note,
        "code": entry.code,
        "line": entry.line,
        "preview": entry.code.splitlines()[0] if entry.code.splitlines() else "",
    }


def build_sources_payload(entries: list[Trick], base_dir: Path) -> list[dict[str, object]]:
    counts: dict[str, dict[str, object]] = {}
    for item in entries:
        rel = source_rel_path(item.source, base_dir)
        if rel not in counts:
            counts[rel] = {
                "key": rel,
                "rel_path": rel,
                "name": item.source.name,
                "display": source_display_name(item.source),
                "group": source_group(item.source, base_dir),
                "count": 0,
            }
        counts[rel]["count"] = int(counts[rel]["count"]) + 1
    return sorted(
        counts.values(),
        key=lambda item: (-int(item["count"]), str(item["rel_path"]).lower()),
    )


def build_tricks_payload(entries: list[Trick], base_dir: Path) -> list[dict[str, object]]:
    return [trick_to_payload(item, base_dir) for item in entries]


def extract_markdown_image_targets(text: str) -> list[str]:
    targets: list[str] = []
    for match in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", text):
        raw = match.group(1).strip()
        if not raw:
            continue
        first = raw.split()[0].strip().strip("<>").strip('"').strip("'")
        if not first:
            continue
        lowered = first.lower()
        if lowered.startswith(("http://", "https://", "data:", "mailto:")):
            continue
        if first.startswith("#"):
            continue
        targets.append(first)
    return targets


def resolve_local_path(base_dir: Path, rel_path: str) -> Path | None:
    target = (base_dir / rel_path).resolve()
    try:
        target.relative_to(base_dir.resolve())
    except ValueError:
        return None
    return target


def collect_export_files(entries: list[Trick], base_dir: Path) -> set[Path]:
    collected: set[Path] = set()
    base_resolved = base_dir.resolve()

    for entry in entries:
        source = entry.source.resolve()
        try:
            source.relative_to(base_resolved)
        except ValueError:
            continue
        if source.is_file():
            collected.add(source)

        if entry.source.suffix.lower() != ".md":
            continue
        for target in extract_markdown_image_targets(entry.code):
            candidate = (entry.source.parent / target).resolve()
            try:
                candidate.relative_to(base_resolved)
            except ValueError:
                continue
            if candidate.is_file():
                collected.add(candidate)

    return collected


def export_static_site(entries: list[Trick], base_dir: Path, output_dir: Path) -> int:
    if not WEB_ASSETS_DIR.exists():
        print(f"Missing web assets directory: {WEB_ASSETS_DIR}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    for asset_name in ("index.html", "app.js", "styles.css"):
        src = WEB_ASSETS_DIR / asset_name
        dst = output_dir / asset_name
        if not src.exists():
            print(f"Missing web asset: {src}")
            return 1
        shutil.copy2(src, dst)

    sources_payload = {
        "total": 0,
        "items": build_sources_payload(entries, base_dir),
    }
    sources_payload["total"] = len(sources_payload["items"])
    tricks_payload = {
        "total": len(entries),
        "items": build_tricks_payload(entries, base_dir),
    }

    (output_dir / "sources.json").write_text(
        json.dumps(sources_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "tricks.json").write_text(
        json.dumps(tricks_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    export_files = collect_export_files(entries, base_dir)
    files_root = output_dir / "files"
    for file_path in export_files:
        relative_path = file_path.relative_to(base_dir.resolve())
        destination = files_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, destination)

    (output_dir / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Static site exported to: {output_dir}")
    print("Publish this folder to GitHub Pages (for example, <site-repo>/tricks).")
    return 0


def serve_web(entries: list[Trick], base_dir: Path, host: str, port: int, open_browser: bool) -> int:
    if not WEB_ASSETS_DIR.exists():
        print(f"Missing web assets directory: {WEB_ASSETS_DIR}")
        return 1

    class TricksWebHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

        def _send_json(self, payload: dict[str, object], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, file_path: Path) -> None:
            if not file_path.exists() or not file_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return
            raw = file_path.read_bytes()
            guessed_type, _ = mimetypes.guess_type(str(file_path))
            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Content-Type",
                MIME_TYPES.get(file_path.suffix, guessed_type or "application/octet-stream"),
            )
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/api/tricks":
                params = parse_qs(parsed.query)
                query = params.get("q", [""])[0]
                source_filters = set(params.get("source", []))
                subset = entries
                if source_filters:
                    subset = [
                        item
                        for item in subset
                        if source_rel_path(item.source, base_dir) in source_filters
                    ]
                subset = search_tricks(subset, query) if query else subset
                self._send_json(
                    {
                        "query": query,
                        "total": len(subset),
                        "items": build_tricks_payload(subset, base_dir),
                    }
                )
                return

            if path == "/api/sources":
                items = build_sources_payload(entries, base_dir)
                self._send_json(
                    {
                        "total": len(items),
                        "items": items,
                    }
                )
                return

            if path == "/sources.json":
                payload = {
                    "total": 0,
                    "items": build_sources_payload(entries, base_dir),
                }
                payload["total"] = len(payload["items"])
                self._send_json(payload)
                return

            if path == "/tricks.json":
                payload = {
                    "total": len(entries),
                    "items": build_tricks_payload(entries, base_dir),
                }
                self._send_json(payload)
                return

            if path.startswith("/files/"):
                rel = unquote(path[len("/files/") :]).lstrip("/")
                target = resolve_local_path(base_dir, rel)
                if target is None:
                    self.send_error(HTTPStatus.BAD_REQUEST, "Invalid file path")
                    return
                self._send_file(target)
                return

            asset_name = WEB_ROUTES.get(path)
            if asset_name is None:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return

            target_file = WEB_ASSETS_DIR / asset_name
            self._send_file(target_file)

    try:
        server = ThreadingHTTPServer((host, port), TricksWebHandler)
    except OSError as exc:
        print(f"Could not start web server on {host}:{port} ({exc}).")
        return 1
    url = f"http://{host}:{port}"
    print(f"Serving tricks web app at {url}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        webbrowser.open(url, new=2)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search and run command snippets from markdown notes."
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Markdown source file to scan (can be repeated).",
    )
    parser.add_argument("--list", action="store_true", help="List all parsed tricks.")
    parser.add_argument("--search", help="Search query for non-interactive output.")
    parser.add_argument("--web", action="store_true", help="Launch a local web UI.")
    parser.add_argument(
        "--export-static",
        metavar="DIR",
        help="Export a static site bundle (index.html/app.js/styles.css + tricks.json/sources.json).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for --web mode.")
    parser.add_argument("--port", type=int, default=8765, help="Port for --web mode.")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open a browser in --web mode.",
    )
    parser.add_argument(
        "--open",
        type=int,
        metavar="ID",
        help="Show one trick by id (combine with --list or --search to discover ids).",
    )

    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parent
    source_inputs = args.source if args.source else [str(path) for path in discover_default_sources(base_dir)]
    sources = parse_source_args(source_inputs)
    existing_sources = [path for path in sources if path.exists()]

    if args.source:
        for path in sources:
            if not path.exists():
                print(f"Warning: source not found: {path}")

    entries = collect_tricks(existing_sources)

    if not entries:
        if not existing_sources:
            print("No markdown sources found. Use --source to add note files.")
        else:
            print("No tricks found in available markdown sources.")
        return 1

    if args.export_static:
        return export_static_site(
            entries=entries,
            base_dir=base_dir,
            output_dir=Path(args.export_static).expanduser(),
        )

    if args.web:
        return serve_web(
            entries=entries,
            base_dir=base_dir,
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
        )

    if args.list:
        print_results(entries, limit=max(15, len(entries)))
        return 0

    if args.search:
        matches = search_tricks(entries, args.search)
        print_results(matches, limit=max(15, len(matches)))
        if args.open is None:
            return 0

    if args.open is not None:
        selected = next((entry for entry in entries if entry.id == args.open), None)
        if selected is None:
            print(f"Trick id {args.open} not found.")
            return 1
        show_detail(selected)
        return 0

    return interactive(entries)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nExiting.")
        raise SystemExit(130)
