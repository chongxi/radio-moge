"""Serve one standalone visualization without exposing sibling experiment artifacts."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("page", type=Path)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    page = args.page.resolve(strict=True)

    class Handler(BaseHTTPRequestHandler):
        def respond(self, body):
            if urlsplit(self.path).path not in ("/", "/index.html"):
                self.send_error(404)
                return
            data = page.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if body:
                self.wfile.write(data)

        def do_GET(self):
            self.respond(True)

        def do_HEAD(self):
            self.respond(False)

    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
