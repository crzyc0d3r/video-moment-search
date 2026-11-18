"""Sample a local video, index image vectors, and retrieve timestamped moments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw, ImageStat

MODEL = "google/embeddinggemma-2"


class SearchError(RuntimeError):
    """Invalid input or an unavailable local inference dependency."""


def finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SearchError(f"{name} must be numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise SearchError(f"{name} must be finite") from None
    if not math.isfinite(result):
        raise SearchError(f"{name} must be finite")
    return result


def unit(vector):
    if not isinstance(vector, (list, tuple)) or not 1 <= len(vector) <= 4096:
        raise SearchError("embedding must contain 1-4096 values")
    values = [finite_number(value, "embedding value") for value in vector]
    length = math.hypot(*values)
    if not length or not math.isfinite(length):
        raise SearchError("embedding must have a finite nonzero norm")
    return [value / length for value in values]


class ColorFixtureEncoder:
    """Offline pixel-feature fixture, not a semantic embedding model."""

    backend = "color-fixture"
    label = "Offline color fixture: RGB pixels and red/green/blue query words; no semantic model"

    def images(self, paths):
        vectors = []
        for path in paths:
            with Image.open(path) as image:
                vectors.append(unit(ImageStat.Stat(image.convert("RGB")).mean))
        return vectors

    def query(self, query):
        terms = set(re.findall(r"[a-z]+", query.lower()))
        values = [1.0 if color in terms else 0.0 for color in ("red", "green", "blue")]
        if not any(values):
            raise SearchError("offline fixture queries must name red, green or blue")
        return unit(values)


class GemmaEncoder:
    """Actual text-plus-vision model; loading is explicit and never used by the demo."""

    backend = MODEL
    label = "EmbeddingGemma 2 local text and image vectors"

    def __init__(self, model=None):
        if model is not None:
            self.model = model
            return
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise SearchError("install requirements-model.txt for EmbeddingGemma 2") from None
        self.model = SentenceTransformer(MODEL, device="cpu",
            config_kwargs={"audio_config": None}, model_kwargs={"torch_dtype": torch.float32})

    def images(self, paths):
        # Each frame is embedded individually, preserving a vector per timestamp.
        values = self.model.encode([{"image": str(Path(path).resolve())} for path in paths],
                                   normalize_embeddings=True, show_progress_bar=False)
        rows = values.tolist() if hasattr(values, "tolist") else values
        return [unit(row) for row in rows]

    def query(self, query):
        # Official multimodal examples compare plain text to {"image": ...} inputs.
        values = self.model.encode(query, normalize_embeddings=True, show_progress_bar=False)
        return unit(values.tolist() if hasattr(values, "tolist") else values)


def command(arguments, timeout=60):
    try:
        result = subprocess.run(arguments, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise SearchError("local media command failed or timed out") from None
    if result.returncode:
        raise SearchError("local media command returned an error")
    return result.stdout


def sample_video(video, directory, *, interval=1.0, max_frames=60):
    interval = finite_number(interval, "sampling interval")
    if not 0 < interval <= 3600 or isinstance(max_frames, bool) or not isinstance(max_frames, int) or not 1 <= max_frames <= 300:
        raise SearchError("require interval in (0,3600] and max_frames in [1,300]")
    video, directory = Path(video).resolve(), Path(directory).resolve()
    if not video.is_file() or video.suffix.lower() not in {".mp4", ".webm", ".mov", ".mkv", ".avi"}:
        raise SearchError("provide a local MP4, WebM, MOV, MKV or AVI video")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise SearchError("install ffmpeg and ffprobe for local video sampling")
    metadata = json.loads(command(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video)]))
    try:
        duration = finite_number(float(metadata["format"]["duration"]), "video duration")
    except (KeyError, TypeError, ValueError):
        raise SearchError("video duration is unavailable") from None
    if duration <= 0:
        raise SearchError("video duration must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    frames = []
    for number in range(min(max_frames, math.ceil(duration / interval))):
        seconds = number * interval
        destination = directory / f"frame-{number:04d}.png"
        # Accurate decode seeking. Stored time is the requested sampling position.
        command(["ffmpeg", "-nostdin", "-v", "error", "-i", str(video), "-ss", str(seconds),
                 "-frames:v", "1", "-vf", "scale=640:-2", "-y", str(destination)])
        if not destination.is_file():
            raise SearchError("video decoder did not produce the sampled frame")
        frames.append((seconds, destination))
    return frames


@dataclass
class VideoIndex:
    root: Path
    backend: str
    label: str
    frames: list
    video: Path | None = None

    def search(self, query, encoder, top_k=3):
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise SearchError("query must contain 1-2000 characters")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 50:
            raise SearchError("top_k must be in [1,50]")
        if encoder.backend != self.backend:
            raise SearchError("query encoder does not match the saved image index")
        query_vector = encoder.query(query)
        if len(query_vector) != len(self.frames[0]["vector"]):
            raise SearchError("query and frame dimensions differ")
        scored = [{"id": frame["id"], "seconds": frame["seconds"],
                   "score": math.fsum(a * b for a, b in zip(query_vector, frame["vector"]))}
                  for frame in self.frames]
        return sorted(scored, key=lambda result: -result["score"])[:top_k]

    def save(self, path):
        Path(path).write_text(json.dumps({"version": 1, "backend": self.backend, "label": self.label,
            "video": str(self.video) if self.video else None, "frames": self.frames}, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        with path.open("rb") as handle:
            raw = handle.read(32 * 1024 * 1024 + 1)
        if len(raw) > 32 * 1024 * 1024:
            raise SearchError("index exceeds 32 MiB")
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get("version") != 1 or value.get("backend") not in {MODEL, "color-fixture"}:
            raise SearchError("unsupported index format or backend")
        frames = value.get("frames")
        if not isinstance(frames, list) or not 1 <= len(frames) <= 300:
            raise SearchError("index must contain 1-300 frames")
        ids, dimensions = set(), set()
        for frame in frames:
            if not isinstance(frame, dict) or not isinstance(frame.get("id"), int) or isinstance(frame["id"], bool) or frame["id"] in ids:
                raise SearchError("frame ids must be unique integers")
            ids.add(frame["id"])
            seconds = finite_number(frame.get("seconds"), "frame timestamp")
            if seconds < 0:
                raise SearchError("frame timestamp must be nonnegative")
            if not isinstance(frame.get("image"), str):
                raise SearchError("frame image must be a relative path")
            image = (path.parent / frame["image"]).resolve()
            if not image.is_relative_to(path.parent) or not image.is_file() or Path(frame["image"]).is_absolute():
                raise SearchError("frame image must stay inside the index directory")
            frame["vector"] = unit(frame.get("vector"))
            frame["seconds"] = seconds
            dimensions.add(len(frame["vector"]))
        if len(dimensions) != 1:
            raise SearchError("all frame vectors must have the same dimension")
        video = Path(value["video"]).resolve() if isinstance(value.get("video"), str) else None
        if video and (not video.is_file() or video.suffix.lower() not in {".mp4", ".webm", ".mov", ".mkv", ".avi"}):
            raise SearchError("saved local video is unavailable")
        label = ColorFixtureEncoder.label if value["backend"] == "color-fixture" else GemmaEncoder.label
        return cls(path.parent, value["backend"], label, frames, video)


def build_index(samples, root, encoder, video=None):
    root = Path(root).resolve()
    paths = [Path(path).resolve() for _, path in samples]
    if not paths or len(paths) > 300 or any(not path.is_relative_to(root) for path in paths):
        raise SearchError("sampled images must remain within the output directory")
    vectors = encoder.images(paths)
    if len(vectors) != len(samples):
        raise SearchError("encoder returned the wrong number of image vectors")
    frames = [{"id": number, "seconds": seconds, "image": str(path.relative_to(root)), "vector": unit(vector)}
              for number, ((seconds, _), path, vector) in enumerate(zip(samples, paths, vectors))]
    index = VideoIndex(root, encoder.backend, encoder.label, frames, Path(video).resolve() if video else None)
    index.save(root / "index.json")
    return VideoIndex.load(root / "index.json")


def create_demo(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    samples = []
    for number, color in enumerate(("red", "green", "blue")):
        image = Image.new("RGB", (320, 180), color)
        ImageDraw.Draw(image).text((15, 15), f"{color} frame at {number}s", fill="white")
        path = directory / f"demo-{number:04d}.png"
        image.save(path)
        samples.append((float(number), path))
    video = None
    if shutil.which("ffmpeg"):
        video = directory / "demo.mp4"
        command(["ffmpeg", "-nostdin", "-v", "error", "-framerate", "1", "-i", str(directory / "demo-%04d.png"),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", str(video)])
    return build_index(samples, directory, ColorFixtureEncoder(), video)


def encoder_for(index):
    return ColorFixtureEncoder() if index.backend == "color-fixture" else GemmaEncoder()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    example = commands.add_parser("demo")
    example.add_argument("--out", type=Path, default=Path(".local/demo"))
    build = commands.add_parser("index")
    build.add_argument("video", type=Path)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--interval", type=float, default=1)
    build.add_argument("--max-frames", type=int, default=60)
    build.add_argument("--fixture", action="store_true", help="Use color-only fixture instead of a semantic model")
    search = commands.add_parser("search")
    search.add_argument("index", type=Path)
    search.add_argument("query")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            index = create_demo(args.out)
            result = {"mode": index.label, "quality_evidence": False,
                      "results": index.search("blue frame", ColorFixtureEncoder())}
        elif args.command == "index":
            samples = sample_video(args.video, args.out / "frames", interval=args.interval, max_frames=args.max_frames)
            encoder = ColorFixtureEncoder() if args.fixture else GemmaEncoder()
            index = build_index(samples, args.out, encoder, args.video)
            result = {"mode": index.label, "frames": len(index.frames), "quality_evidence": False}
        else:
            index = VideoIndex.load(args.index)
            result = {"mode": index.label, "quality_evidence": False,
                      "results": index.search(args.query, encoder_for(index))}
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (SearchError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
