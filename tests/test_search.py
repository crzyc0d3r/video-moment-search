import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

from app import handler_for
from video_search import (MODEL, ColorFixtureEncoder, GemmaEncoder, SearchError,
                          VideoIndex, build_index, create_demo, main, sample_video, unit)


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def images(self):
        # Labels and filenames contain no color names; vectors must come from pixels.
        paths = []
        for number, rgb in enumerate(((255, 0, 0), (0, 0, 255), (0, 255, 0))):
            path = self.root / f"sample-{number}.png"
            Image.new("RGB", (16, 16), rgb).save(path)
            paths.append((number * 2.0, path))
        return paths

    def index(self):
        return build_index(self.images(), self.root, ColorFixtureEncoder())

    def test_actual_image_pixels_find_timestamp(self):
        index = self.index()
        result = index.search("blue object", ColorFixtureEncoder(), top_k=1)
        self.assertEqual(result[0]["id"], 1)
        self.assertEqual(result[0]["seconds"], 2.0)
        self.assertEqual(result[0]["score"], 1.0)
        self.assertEqual(index.search("green object", ColorFixtureEncoder(), top_k=1)[0]["seconds"], 4.0)

    def test_image_index_round_trip(self):
        index = self.index()
        loaded = VideoIndex.load(self.root / "index.json")
        self.assertEqual(loaded.frames, index.frames)
        self.assertEqual(loaded.backend, "color-fixture")
        self.assertIn("Offline color fixture", loaded.label)

    def test_cosine_ranking_normalization_and_tie_order(self):
        index = self.index()
        index.frames[1]["vector"] = index.frames[0]["vector"].copy()
        results = index.search("red", ColorFixtureEncoder())
        self.assertEqual([r["id"] for r in results[:2]], [0, 1])
        self.assertAlmostEqual(sum(v * v for v in unit([3, 4])), 1)

    def test_malformed_vectors_and_timestamp_rejected(self):
        self.index()
        path = self.root / "index.json"
        base = json.loads(path.read_text())
        for vector in ([0, 0, 0], [float("nan"), 0, 1], [True, 0, 1], [10 ** 400, 0, 1], [1, 0]):
            changed = json.loads(json.dumps(base))
            changed["frames"][0]["vector"] = vector
            path.write_text(json.dumps(changed))
            with self.subTest(vector=vector), self.assertRaises(SearchError):
                VideoIndex.load(path)
        for timestamp in (-1, float("inf"), True):
            changed = json.loads(json.dumps(base)); changed["frames"][0]["seconds"] = timestamp
            path.write_text(json.dumps(changed))
            with self.subTest(timestamp=timestamp), self.assertRaises(SearchError):
                VideoIndex.load(path)

    def test_frame_path_escape_and_duplicate_ids_rejected(self):
        self.index()
        path = self.root / "index.json"
        base = json.loads(path.read_text())
        for image in ("../outside.png", str((self.root / "sample-0.png").resolve())):
            changed = json.loads(json.dumps(base)); changed["frames"][0]["image"] = image
            path.write_text(json.dumps(changed))
            with self.subTest(image=image), self.assertRaises(SearchError):
                VideoIndex.load(path)
        base["frames"][1]["id"] = base["frames"][0]["id"]
        path.write_text(json.dumps(base))
        with self.assertRaises(SearchError):
            VideoIndex.load(path)

    def test_encoder_backend_and_dimension_must_match(self):
        index = self.index()
        with self.assertRaises(SearchError):
            index.search("blue", Mock(backend=MODEL))
        encoder = Mock(backend="color-fixture", query=Mock(return_value=[1, 0]))
        with self.assertRaises(SearchError):
            index.search("blue", encoder)

    def test_fixture_unknown_query_and_limits(self):
        index = self.index()
        for query in ("a dog", "", "x" * 2001):
            with self.subTest(query=query[:10]), self.assertRaises(SearchError):
                index.search(query, ColorFixtureEncoder())
        for count in (0, 51, True):
            with self.subTest(count=count), self.assertRaises(SearchError):
                index.search("blue", ColorFixtureEncoder(), top_k=count)

    def test_native_multimodal_adapter_sends_images_and_text(self):
        paths = [path for _, path in self.images()]
        model = Mock()
        model.encode.side_effect = [[[1, 0, 0], [0, 0, 1], [0, 1, 0]], [0, 0, 1]]
        encoder = GemmaEncoder(model=model)
        vectors = encoder.images(paths)
        query = encoder.query("blue object")
        self.assertEqual(model.encode.call_args_list[0].args[0], [{"image": str(p.resolve())} for p in paths])
        self.assertEqual(model.encode.call_args_list[1].args[0], "blue object")
        self.assertEqual(vectors[1], query)
        self.assertTrue(model.encode.call_args_list[0].kwargs["normalize_embeddings"])

    def test_invalid_sampler_arguments_make_no_media_calls(self):
        for kwargs in ({"interval": 0}, {"interval": float("nan")}, {"max_frames": 0}, {"max_frames": True}):
            with self.subTest(kwargs=kwargs), patch("video_search.command") as command, self.assertRaises(SearchError):
                sample_video("missing.mp4", self.root, **kwargs)
            command.assert_not_called()

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg and ffprobe required for real video extraction")
    def test_native_video_decode_pixel_index_and_timestamp_search(self):
        demo = create_demo(self.root / "demo")
        self.assertTrue(demo.video.is_file())
        samples = sample_video(demo.video, self.root / "sampled", interval=1.0, max_frames=3)
        self.assertEqual([seconds for seconds, _ in samples], [0, 1, 2])
        index = build_index(samples, self.root, ColorFixtureEncoder(), demo.video)
        result = index.search("blue frame", ColorFixtureEncoder(), top_k=1)
        self.assertEqual(result[0]["seconds"], 2)
        self.assertGreater(result[0]["score"], .99)

    def test_cli_fixture_search_json(self):
        self.index()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["search", str(self.root / "index.json"), "blue"]), 0)
        result = json.loads(stdout.getvalue())
        self.assertFalse(result["quality_evidence"])
        self.assertEqual(result["results"][0]["seconds"], 2)


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        # The server tests also work without ffmpeg; a byte fixture tests range semantics.
        with patch("video_search.shutil.which", return_value=None):
            cls.index = create_demo(cls.root)
        cls.index.video = cls.root / "range.mp4"
        cls.index.video.write_bytes(b"0123456789")
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(cls.index, ColorFixtureEncoder()))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join()
        cls.temporary.cleanup()

    def test_page_search_and_image_routes(self):
        with urlopen(self.base) as response:
            self.assertEqual(response.status, 200)
            self.assertIn(b"video.currentTime=result.seconds", response.read())
        with urlopen(self.base + "/api/search?q=blue") as response:
            result = json.load(response)
            self.assertEqual(result["results"][0]["seconds"], 2)
            self.assertIn("fixture", result["mode"])
        with urlopen(self.base + "/frames/2") as response:
            self.assertEqual(response.headers["Content-Type"], "image/png")
            self.assertTrue(response.read().startswith(b"\x89PNG"))

    def test_video_range_supports_timestamp_seeking(self):
        request = Request(self.base + "/video", headers={"Range": "bytes=2-5"})
        with urlopen(request) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.headers["Content-Range"], "bytes 2-5/10")
            self.assertEqual(response.read(), b"2345")

    def test_invalid_query_range_and_path_return_errors(self):
        for route, expected in (("/api/search?q=dog", 400), ("/frames/999", 404), ("/frames/../README.md", 404)):
            with self.subTest(route=route), self.assertRaises(HTTPError) as error:
                urlopen(self.base + route)
            self.assertEqual(error.exception.code, expected)
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(self.base + "/video", headers={"Range": "bytes=100-200"}))
        self.assertEqual(error.exception.code, 416)


if __name__ == "__main__":
    unittest.main()
