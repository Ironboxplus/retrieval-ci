"""VolumeReader against a real OpenList process (Local storage standing in for a Cfs volume).

Listing, the streamed WebDAV read, the size checks and the read-only user's refusal
are all the node's own behaviour, so nothing here is faked: the node is the cluster's
binary, the volume a directory, the files real bytes. The one stand-in is a plain
socket server that promises more bytes than it sends, because a real node does not
cut a transfer on demand.
"""
from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import threading
import urllib.error
import urllib.request

import pytest

from pipeline.openlist_reader import VolumeError, VolumeNotFound, VolumeReader
from tests.openlist_node import ADMIN_PASSWORD, READER_PASSWORD, READER_USER, start_node

VOLUME = "/cfs-tier1"


@pytest.fixture(scope="module")
def node(tmp_path_factory):
    node = start_node(tmp_path_factory)
    yield node
    node.stop()


@pytest.fixture(scope="module")
def volume(node, tmp_path_factory):
    root = node.mount_local(VOLUME, tmp_path_factory.mktemp("volume"))
    (root / "vldb" / "2022").mkdir(parents=True)
    (root / "vldb" / "2022" / "a.pdf").write_bytes(b"%PDF-1.4 a")
    (root / "vldb" / "2022" / "a.provenance.json").write_text('{"paper_id": "a"}', encoding="utf-8")
    (root / "vldb" / "2022" / "b.pdf").write_bytes(os.urandom(3 * 1024 * 1024 + 17))
    (root / "vldb" / "2023").mkdir()
    (root / "vldb" / "2023" / "c.pdf").write_bytes(b"%PDF-1.4 c")
    (root / ".hidden").mkdir()
    (root / ".hidden" / "log").write_bytes(b"x")
    return root


@pytest.fixture(scope="module")
def reader(node, volume):
    return VolumeReader(node.url, "retrival-reader", password_file=node.add_reader())


def test_list_files_is_recursive_relative_and_sized(reader, volume):
    listed = reader.list_files(f"{VOLUME}/vldb")
    assert listed == {
        "2022/a.pdf": 10,
        "2022/a.provenance.json": len('{"paper_id": "a"}'),
        "2022/b.pdf": (volume / "vldb" / "2022" / "b.pdf").stat().st_size,
        "2023/c.pdf": 10,
    }
    assert reader.list_files(f"{VOLUME}/vldb/2023/") == {"c.pdf": 10}


def test_listing_a_missing_directory_says_not_found(reader):
    with pytest.raises(VolumeNotFound, match="vldb/2099"):
        reader.list_files(f"{VOLUME}/vldb/2099")


def test_fetch_streams_a_large_file_and_leaves_no_part(reader, volume, tmp_path):
    src = volume / "vldb" / "2022" / "b.pdf"
    dest = tmp_path / "out" / "2022" / "b.pdf"
    written = reader.fetch(f"{VOLUME}/vldb/2022/b.pdf", dest)
    assert written == src.stat().st_size
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == hashlib.sha256(src.read_bytes()).hexdigest()
    assert [p.name for p in dest.parent.iterdir()] == ["b.pdf"]


def test_a_size_that_differs_from_the_listing_removes_the_part_and_raises(reader, tmp_path):
    dest = tmp_path / "a.pdf"
    dest.write_bytes(b"older copy")
    with pytest.raises(VolumeError, match="10 bytes.*expected 11"):
        reader.fetch(f"{VOLUME}/vldb/2022/a.pdf", dest, size=11)
    assert dest.read_bytes() == b"older copy"  # the earlier file is not replaced by a bad one
    assert [p.name for p in tmp_path.iterdir()] == ["a.pdf"]  # and no .part is left


def test_fetching_a_missing_file_says_not_found(reader, tmp_path):
    with pytest.raises(VolumeNotFound):
        reader.fetch(f"{VOLUME}/vldb/2022/nope.pdf", tmp_path / "nope.pdf")
    assert list(tmp_path.iterdir()) == []


def test_a_transfer_cut_short_removes_the_part_and_raises(tmp_path):
    class Truncating(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "1000")
            self.end_headers()
            self.wfile.write(b"x" * 400)  # then the connection closes: 600 bytes short

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Truncating)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        cut = VolumeReader(f"http://127.0.0.1:{server.server_port}", "u", "p")
        with pytest.raises(VolumeError, match="cut short|incomplete"):
            cut.fetch("/cfs-tier1/x.pdf", tmp_path / "x.pdf")
    finally:
        server.shutdown()
    assert list(tmp_path.iterdir()) == []


def test_read_bytes_returns_a_small_file_and_refuses_a_big_one(reader):
    assert reader.read_bytes(f"{VOLUME}/vldb/2022/a.provenance.json") == b'{"paper_id": "a"}'
    with pytest.raises(VolumeError, match="larger than"):
        reader.read_bytes(f"{VOLUME}/vldb/2022/b.pdf", limit=1024)


def _refusal(request):
    """What the node answered a write with: its 403, or None when it hung up first.

    A node that refuses without reading the body may close the socket while the client is
    still sending it, so the refusal then reaches the client as a reset, not a status.
    """
    try:
        with urllib.request.urlopen(request, timeout=20) as resp:
            body = resp.read()
    except urllib.error.HTTPError as err:
        return err.code
    except ConnectionError:
        return None
    return json.loads(body)["code"]


def test_the_reader_user_cannot_write(node, volume, reader):
    """The node refuses PUT for this user whichever door it comes through; the volume is unchanged."""
    before = sorted(p.relative_to(volume).as_posix() for p in volume.rglob("*"))
    basic = base64.b64encode(f"{READER_USER}:{READER_PASSWORD}".encode()).decode()
    dav_put = urllib.request.Request(
        f"{node.url}/dav{VOLUME}/vldb/2022/new.pdf", data=b"zzz", method="PUT",
        headers={"Authorization": "Basic " + basic, "Content-Type": "application/octet-stream"})
    assert _refusal(dav_put) in (403, None)

    token = node.api("POST", "/api/auth/login", {"username": READER_USER, "password": READER_PASSWORD})["token"]
    api_put = urllib.request.Request(
        node.url + "/api/fs/put", data=b"zzz", method="PUT",
        headers={"Authorization": token, "File-Path": f"{VOLUME}/vldb/2022/new2.pdf",
                 "Content-Type": "application/octet-stream", "Content-Length": "3"})
    assert _refusal(api_put) in (403, None)
    assert sorted(p.relative_to(volume).as_posix() for p in volume.rglob("*")) == before
    assert not hasattr(reader, "put") and not hasattr(reader, "delete")


def test_errors_never_carry_the_password(node, volume, tmp_path):
    secret = "s3cret-reader-password"
    wrong = VolumeReader(node.url, READER_USER, secret)
    with pytest.raises(VolumeError) as bad_login:
        wrong.list_files(f"{VOLUME}/vldb")
    assert secret not in str(bad_login.value)
    with pytest.raises(VolumeError) as bad_dav:
        wrong.fetch(f"{VOLUME}/vldb/2022/a.pdf", tmp_path / "a.pdf")
    assert secret not in str(bad_dav.value)
    assert base64.b64encode(f"{READER_USER}:{secret}".encode()).decode() not in str(bad_dav.value)
    assert ADMIN_PASSWORD not in str(bad_dav.value)


def test_from_env_reads_the_user_and_the_password_file(node, reader, tmp_path):
    password_file = tmp_path / "pw"
    password_file.write_text(READER_PASSWORD + "\n", encoding="utf-8")
    env = {"OPENLIST_URL": node.url, "OPENLIST_READER_USER": READER_USER,
           "OPENLIST_READER_PASS_FILE": str(password_file)}
    assert VolumeReader.from_env(env).list_files(f"{VOLUME}/vldb/2023") == {"c.pdf": 10}
    with pytest.raises(VolumeError, match="OPENLIST_READER_USER"):
        VolumeReader.from_env({"OPENLIST_URL": node.url})
    with pytest.raises(VolumeError, match="OPENLIST_READER_PASS_FILE"):
        VolumeReader.from_env({"OPENLIST_READER_USER": READER_USER})
    empty = tmp_path / "empty"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(VolumeError, match="empty"):
        VolumeReader(node.url, READER_USER, password_file=empty)
    assert VolumeReader.from_env({"OPENLIST_READER_USER": "u", "OPENLIST_READER_PASS_FILE": str(password_file)}).url \
        == "http://127.0.0.1:5244"
