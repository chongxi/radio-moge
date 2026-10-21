"""Serve only a generated viewer directory, including precompressed depth arrays."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


class ViewerHandler(SimpleHTTPRequestHandler):
    def list_directory(self, path):
        self.send_error(404)
        return None

    def end_headers(self):
        path = urlsplit(self.path).path
        if path.endswith(".depth.gz"):
            # Fetch transparently decodes the precompressed, lossless float32 planes.
            self.send_header("Content-Encoding", "gzip")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-cache" if path.endswith((".json", ".html", "/")) else "max-age=3600")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob: data:; "
                         "style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; "
                         "object-src 'none'; base-uri 'none'")
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not (directory / "index.html").is_file() or not (directory / "manifest.json").is_file():
        parser.error("directory must contain the exported index.html and manifest.json")
    server = ThreadingHTTPServer((args.host, args.port), partial(ViewerHandler, directory=str(directory)))
    print(f"Serving held-out viewer at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
