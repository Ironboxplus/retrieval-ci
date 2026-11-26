"""PDF and JATS from the PMC OA bucket first, Europe PMC's render URL second.

A real child process hosts a stand-in bucket (ListObjectsV2 XML, versioned
folders), a Cloudflare-style 403 render endpoint and a forwarding proxy; httpx,
the proxy and the filesystem are all real, and the proxy log proves which URLs
were fetched.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path

import pytest

from pipeline.medical_corpus.models import MedicalPaperRecord
from pipeline.medical_corpus.pdf_fetch import download_pdf
from pipeline.medical_corpus.pmc_oa import resolve_pmc_oa
from pipeline.medical_corpus.proxy import MedicalProxyConfig

_SERVER = r'''
import http.server, json, pathlib, socketserver, sys, threading, urllib.parse, urllib.request

root = pathlib.Path(sys.argv[1])
log = root / "proxy-requests.jsonl"
lock = threading.Lock()

def pdf(marker):
    header = b"%PDF-1.7\n"
    objs = [b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 72 72] /Contents 4 0 R >>\nendobj\n",
            b"4 0 obj\n<< /Length " + str(len(marker) + 2048).encode() + b" >>\nstream\n" + marker + b"y" * 2048 + b"\nendstream\nendobj\n"]
    offsets, cur = [], len(header)
    for o in objs:
        offsets.append(cur); cur += len(o)
    xref = b"xref\n0 5\n0000000000 65535 f \n" + b"".join(f"{x:010d} 00000 n \n".encode() for x in offsets)
    return header + b"".join(objs) + xref + b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n" + str(cur).encode() + b"\n%%EOF\n"

JATS = b'<?xml version="1.0"?>\n<article><body><p><mml:math>x</mml:math></p></body></article>\n'
# PMC1: versions 1 and 2, both with PDF and XML. PMC2: PDF only. PMC3: not in the bucket.
FILES = {
    "PMC1.1/PMC1.1.pdf": pdf(b"v1"), "PMC1.1/PMC1.1.xml": JATS.replace(b"x", b"old"),
    "PMC1.2/PMC1.2.pdf": pdf(b"v2"), "PMC1.2/PMC1.2.xml": JATS, "PMC1.2/fig1.jpg": b"jpg",
    "PMC10.1/PMC10.1.pdf": pdf(b"other"),
    "PMC2.1/PMC2.1.pdf": pdf(b"pdf-only"),
}

class Origin(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        url = urllib.parse.urlsplit(self.path)
        q = urllib.parse.parse_qs(url.query)
        if url.path == "/bucket/" and q.get("list-type") == ["2"]:
            prefix = q.get("prefix", [""])[0]
            keys = "".join(f"<Contents><Key>{k}</Key></Contents>" for k in sorted(FILES) if k.startswith(prefix))
            body = f'<?xml version="1.0"?><ListBucketResult><Prefix>{prefix}</Prefix>{keys}</ListBucketResult>'.encode()
            return self.reply(200, "application/xml", body)
        if url.path.startswith("/bucket/") and url.path[len("/bucket/"):] in FILES:
            key = url.path[len("/bucket/"):]
            return self.reply(200, "application/pdf" if key.endswith(".pdf") else "text/xml", FILES[key])
        if url.path == "/render":
            return self.reply(403, "text/html", b"<html><title>Just a moment...</title>cloudflare cf-chl</html>",
                              {"Server": "cloudflare", "cf-mitigated": "challenge"})
        if url.path == "/paper.pdf":
            return self.reply(200, "application/pdf", pdf(b"fallback"))
        self.send_error(404)
    def reply(self, status, ctype, body, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a): pass

class Proxy(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        with lock, log.open("a", encoding="utf-8") as out:
            out.write(json.dumps({"target": self.path}) + "\n")
        try:
            upstream = urllib.request.urlopen(self.path, timeout=30)
        except urllib.error.HTTPError as err:
            upstream = err
        body = upstream.read()
        self.send_response(upstream.status if hasattr(upstream, "status") else upstream.code)
        for k, v in upstream.headers.items():
            if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a): pass

origin = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Origin)
proxy = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Proxy)
threading.Thread(target=origin.serve_forever, daemon=True).start()
threading.Thread(target=proxy.serve_forever, daemon=True).start()
print(json.dumps({"origin": origin.server_address[1], "proxy": proxy.server_address[1]}), flush=True)
threading.Event().wait()
'''


@contextmanager
def _bucket(tmp_path: Path):
    proc = subprocess.Popen([sys.executable, "-u", "-c", textwrap.dedent(_SERVER), str(tmp_path)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        line = proc.stdout.readline().strip()
        assert line, proc.stderr.read()
        ports = json.loads(line)
        origin = f"http://127.0.0.1:{ports['origin']}"
        yield {"origin": origin, "bucket": f"{origin}/bucket",
               "proxy": MedicalProxyConfig(http_proxy=f"http://127.0.0.1:{ports['proxy']}",
                                           https_proxy=f"http://127.0.0.1:{ports['proxy']}"),
               "log": tmp_path / "proxy-requests.jsonl"}
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture()
def bucket(tmp_path: Path):
    with _bucket(tmp_path) as server:
        yield server


def _fetched(server) -> list[str]:
    return [json.loads(ln)["target"] for ln in server["log"].read_text(encoding="utf-8").splitlines()]


def _record(pmcid: str | None, pdf_url: str) -> MedicalPaperRecord:
    return MedicalPaperRecord(paper_id=f"med-{pmcid or 'none'}", title="t", year=2024, journal="Test Journal",
                              pmcid=pmcid, pdf_url=pdf_url, has_pdf=True, is_open_access=True)


def test_resolve_picks_the_newest_version_with_a_pdf_and_ignores_prefix_neighbours(bucket) -> None:
    article = resolve_pmc_oa("pmc1", proxy=bucket["proxy"], base_url=bucket["bucket"])
    assert article is not None and article.version == 2
    assert article.pdf_url == f"{bucket['bucket']}/PMC1.2/PMC1.2.pdf"
    assert article.xml_url == f"{bucket['bucket']}/PMC1.2/PMC1.2.xml"
    assert resolve_pmc_oa("PMC3", proxy=bucket["proxy"], base_url=bucket["bucket"]) is None
    assert all(t.startswith(bucket["origin"]) for t in _fetched(bucket))


def test_bucket_pdf_and_jats_are_used_and_the_cloudflare_render_url_is_never_requested(
    bucket, tmp_path: Path,
) -> None:
    out = download_pdf(_record("PMC1", f"{bucket['origin']}/render"), pdf_root=tmp_path / "pdf",
                       proxy=bucket["proxy"], execute=True, allow_cf_bypass=False,
                       pmc_oa_base_url=bucket["bucket"])

    assert out.download_status == "downloaded", out.download_error
    pdf = Path(out.pdf_path)
    assert b"v2" in pdf.read_bytes()
    assert out.pdf_sha256 == hashlib.sha256(pdf.read_bytes()).hexdigest()
    jats = Path(out.jats_path)
    assert jats == pdf.with_suffix(".xml") and b"<mml:math>x</mml:math>" in jats.read_bytes()
    assert out.jats_sha256 == hashlib.sha256(jats.read_bytes()).hexdigest()
    assert out.pdf_url == f"{bucket['origin']}/render"  # the manifest keeps the catalogue URL
    assert any("PMC1.2/PMC1.2.pdf" in n for n in out.notes)
    assert f"{bucket['origin']}/render" not in _fetched(bucket)
    assert not list(pdf.parent.glob("*.write_in_progress*"))


def test_an_article_without_bucket_xml_still_downloads_its_pdf_and_says_so(bucket, tmp_path: Path) -> None:
    out = download_pdf(_record("PMC2", f"{bucket['origin']}/render"), pdf_root=tmp_path / "pdf",
                       proxy=bucket["proxy"], execute=True, allow_cf_bypass=False,
                       pmc_oa_base_url=bucket["bucket"])
    assert out.download_status == "downloaded"
    assert out.jats_path is None and out.jats_sha256 is None
    assert any("no JATS" in n for n in out.notes)


def test_an_article_missing_from_the_bucket_falls_back_to_the_catalogue_url(bucket, tmp_path: Path) -> None:
    out = download_pdf(_record("PMC3", f"{bucket['origin']}/paper.pdf"), pdf_root=tmp_path / "pdf",
                       proxy=bucket["proxy"], execute=True, allow_cf_bypass=False,
                       pmc_oa_base_url=bucket["bucket"])
    assert out.download_status == "downloaded"
    assert b"fallback" in Path(out.pdf_path).read_bytes()
    assert out.jats_path is None
    assert f"{bucket['origin']}/paper.pdf" in _fetched(bucket)


def test_the_fallback_still_reports_the_render_url_block(bucket, tmp_path: Path) -> None:
    out = download_pdf(_record("PMC3", f"{bucket['origin']}/render"), pdf_root=tmp_path / "pdf",
                       proxy=bucket["proxy"], execute=True, allow_cf_bypass=False,
                       pmc_oa_base_url=bucket["bucket"])
    assert out.download_status == "blocked"
    assert out.pdf_path is None


def test_jats_fields_survive_the_manifest_round_trip() -> None:
    rec = _record("PMC1", "u")
    rec.jats_path, rec.jats_sha256 = "/x/a.xml", "ab" * 32
    again = MedicalPaperRecord.from_dict(json.loads(json.dumps(rec.to_dict())))
    assert (again.jats_path, again.jats_sha256) == ("/x/a.xml", "ab" * 32)
