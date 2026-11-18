"""Loopback-only video search preview with timestamp seeking."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from video_search import SearchError, VideoIndex, create_demo, encoder_for

PAGE = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Video Moment Search</title><style>body{max-width:960px;margin:40px auto;padding:0 20px;font:16px system-ui;color:#172334;background:#f3f6fa}h1{margin-bottom:8px}input{padding:12px;width:65%;font:inherit}button{padding:12px;font:inherit;cursor:pointer}video{display:block;width:100%;max-height:420px;background:#132031;margin:20px 0}.results{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:16px}.card{background:white;border:1px solid #ccd6e1;border-radius:10px;padding:12px}.card img{width:100%}.mode{background:#e4edf6;padding:14px;border-radius:8px}#status{min-height:24px}</style>
<h1>Video Moment Search</h1><p>Find a frame, then jump to its moment.</p><p id="mode" class="mode"></p>
<form id="search"><input id="query" value="blue frame" aria-label="Search query"><button>Search</button></form>
<p id="status" role="status"></p><video id="video" controls preload="metadata" src="/video"></video>
<div id="results" class="results"></div><script>
const status=document.getElementById('status'),video=document.getElementById('video');
async function search(e){if(e)e.preventDefault();status.textContent='Searching…';try{
const response=await fetch('/api/search?q='+encodeURIComponent(document.getElementById('query').value));const data=await response.json();
if(!response.ok)throw Error(data.error||'Search failed');document.getElementById('mode').textContent=data.mode;
video.hidden=!data.has_video;document.getElementById('results').replaceChildren();
for(const result of data.results){const card=document.createElement('div');card.className='card';
const image=document.createElement('img');image.src='/frames/'+result.id;image.alt='Sampled frame at '+result.seconds+' seconds';
const label=document.createElement('p');label.textContent=result.seconds.toFixed(2)+'s · similarity '+result.score.toFixed(3);
const button=document.createElement('button');button.textContent=data.has_video?'Jump to moment':'View timestamp';button.onclick=()=>{video.currentTime=result.seconds;status.textContent='Selected '+result.seconds.toFixed(2)+' seconds';};
card.append(image,label,button);document.getElementById('results').append(card);}status.textContent=data.results.length+' moments found';
}catch(error){status.textContent=error.message;}}document.getElementById('search').onsubmit=search;search();
</script></html>'''


def handler_for(index, encoder):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == "/":
                return self.send(200, PAGE.encode(), "text/html; charset=utf-8")
            if url.path == "/api/search":
                try:
                    query = parse_qs(url.query).get("q", [""])[0]
                    results = index.search(query, encoder)
                    value = {"mode": index.label, "quality_evidence": False,
                             "has_video": index.video is not None, "results": results}
                    return self.send(200, json.dumps(value, allow_nan=False).encode(), "application/json")
                except (SearchError, ValueError) as exc:
                    return self.send(400, json.dumps({"error": str(exc)}).encode(), "application/json")
            if url.path.startswith("/frames/"):
                try:
                    number = int(url.path.removeprefix("/frames/"))
                    frame = next(frame for frame in index.frames if frame["id"] == number)
                    path = (index.root / frame["image"]).resolve()
                    if not path.is_relative_to(index.root):
                        raise ValueError()
                    return self.send(200, path.read_bytes(), "image/png")
                except (ValueError, StopIteration, OSError):
                    return self.send(404, b"Frame not found", "text/plain")
            if url.path == "/video" and index.video:
                # Byte ranges let the browser seek without fetching the whole video.
                size = index.video.stat().st_size
                start, end, status = 0, size - 1, 200
                header = self.headers.get("Range")
                if header:
                    import re
                    match = re.fullmatch(r"bytes=(\d+)-(\d*)", header)
                    if not match:
                        return self.send(416, b"Unsupported range", "text/plain")
                    start = int(match[1]); end = min(int(match[2]), end) if match[2] else end
                    if start > end or start >= size:
                        return self.send(416, b"Unsatisfiable range", "text/plain")
                    status = 206
                self.send_response(status)
                self.send_header("Content-Type", mimetypes.guess_type(index.video.name)[0] or "application/octet-stream")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(end - start + 1))
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                with index.video.open("rb") as source:
                    source.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        chunk = source.read(min(64 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk); remaining -= len(chunk)
                return
            self.send(404, b"Not found", "text/plain")
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--port", type=int, default=8843)
    args = parser.parse_args()
    index = VideoIndex.load(args.index) if args.index else create_demo(Path(".local/demo"))
    encoder = encoder_for(index)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(index, encoder))
    print(f"Video Moment Search: http://127.0.0.1:{server.server_port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
