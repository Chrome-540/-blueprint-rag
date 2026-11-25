"""The app itself, without Gemini: the router, indexing and answers are replaced by stand-ins.

python tests/7_app/run.py
"""
import asyncio
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]
sys.stdout.reconfigure(encoding="utf-8")


def alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return str(pid) in out


def test_workers_end_with_the_app() -> None:
    code = ("import sys, time; sys.path.insert(0, r'%s'); import workers; from concurrent.futures import ProcessPoolExecutor as P; "
            "p = P(2, initializer=workers.exit_with_parent); f = [p.submit(time.sleep, 1) for _ in range(2)]; "
            "[x.result() for x in f]; print(*p._processes, flush=True); time.sleep(60)" % ROOT)
    parent = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, cwd=ROOT)
    pids = [int(x) for x in parent.stdout.readline().split()]
    assert pids and all(alive(p) for p in pids), "workers did not start"
    # the venv's python.exe starts the real one as a child, so stop the whole chain except the workers
    real = subprocess.run(["powershell", "-NoProfile", "-Command",
                           f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pids[0]}').ParentProcessId"],
                          capture_output=True, text=True).stdout.split()
    for pid in {parent.pid, *(int(x) for x in real)} - set(pids):
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
    for _ in range(20):
        if not any(alive(p) for p in pids):
            break
        time.sleep(.5)
    left = [p for p in pids if alive(p)]
    for p in left:
        subprocess.run(["taskkill", "/F", "/PID", str(p)], capture_output=True)
    assert not left, f"workers still running after the app ended: {left}"
    print("ok  2. workers end with the app")


def test_backend() -> None:
    tmp = Path(tempfile.mkdtemp())
    os.chdir(tmp)
    os.environ.update(IV_DATA_DIR=str(tmp / "projects"), GEMINI_API_KEY="unused")
    pdf = b"%PDF-1.4 test"
    (tmp / "projects/Old").mkdir(parents=True)
    (tmp / "projects/Old/Set.pdf").write_bytes(pdf)  # an unfinished project from before the start
    import pipeline
    import main
    import router
    from fastapi.testclient import TestClient
    indexed = []

    async def fake_run(pdfs, status, project):
        indexed.append((project, sorted(p.name for p in pdfs)))
        for p in pdfs:
            status[pipeline.slug(p)] = {"file": p.name, "stage": "done", "done": 1, "total": 1}
            marker = pipeline.BASE_DIR / project / pipeline.slug(p) / ".indexed"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
        await asyncio.sleep(0)

    pipeline.run = fake_run
    corpus = []
    main._rag = lambda slug: MagicMock(corpus=corpus, ask=lambda q, history=None: ("42", [{"metadata": {}}] if corpus else []))
    main._sheet_card = lambda slug, hit: {"src": "x"}
    c = TestClient(main.app)

    def wait() -> None:
        for _ in range(50):
            time.sleep(.1)
            if all(e["state"] != "indexing" for e in c.get("/index_status").json().values()):
                return

    def up(project, name, **data):
        r = c.post("/upload", data={"domain": project, **data}, files=[("files", (name, pdf))])
        wait()
        return r

    def chat(tool, project="P", **args):
        router.detect_intent = lambda *a: {"name": tool, "args": args}
        return c.post("/chat", json={"domain": project, "message": "x"}).json()

    time.sleep(1.5)
    assert indexed == [] and c.get("/index_status").json() == {}, "something was indexed at startup"
    assert c.get("/unfinished").json() == ["Old"]
    c.post("/continue", data={"domain": "Old"})
    wait()
    assert indexed == [("old", ["Set.pdf"])] and c.get("/unfinished").json() == []
    c.post("/continue", data={"domain": "Old"})
    wait()
    assert len(indexed) == 1, "a finished project must not be indexed again"
    print("ok  3. nothing runs at startup, Continue finishes a project")

    assert up("P", "Plan.pdf").status_code == 200 and indexed[-1] == ("p", ["Plan.pdf"])
    n = len(indexed)
    assert up("P", "Contract Agreement.pdf", kind="other").status_code == 200 and len(indexed) == n
    assert (tmp / "projects/P/_documents/Contract Agreement.pdf").exists()
    assert up("P", "Plan 2.pdf", kind="blueprint").status_code == 200
    assert indexed[-1] == ("p", ["Plan 2.pdf", "Plan.pdf"]), "a document must never be indexed"
    assert up("P", "x.pdf", kind="nonsense").status_code == 400
    a = chat("show_file", file_hint="contract agreement")
    assert len(a["pdfs"]) == 1 and c.get(a["pdfs"][0]).content == pdf
    assert "Contract Agreement.pdf" in chat("list_contents")["answer"]
    up("Docs only", "Quote.pdf", kind="other")
    assert c.get("/domains").json() == ["Docs only", "Old", "P"] and "Docs only" not in c.get("/unfinished").json()
    src = tmp / "src/documents"
    src.mkdir(parents=True)
    (src / "Set.pdf").write_bytes(pdf)
    for i in range(10):  # the KeyError crash depended on timing
        assert c.post("/add_project", data={"name": f"R{i}", "path": str(tmp / "src")}).status_code == 200
        wait()
        assert indexed[-1] == (f"r{i}", ["Set.pdf"]), c.get("/index_status").json()[f"r{i}"]
    n = len(indexed)
    assert c.post("/add_project", data={"name": "Q", "path": str(tmp / "src"), "kind": "other"}).status_code == 200
    wait()
    assert len(indexed) == n and len(main._documents("q")) == 1
    assert c.delete("/domains/P").status_code == 200 and not (tmp / "projects/P").exists()
    print("ok  5. Blueprint / Other document, projects created and deleted")

    corpus.append({"metadata": {"sheet_id": "A1"}})
    assert main.AI_NOTE in chat("answer_from_context", "Old", query="x")["answer"]
    assert main.AI_NOTE not in chat("small_talk", "Old", reply="Hi")["answer"]
    print("ok  1. note under AI answers only")


def test_frontend() -> None:
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    files = {"v1": {"file": "Set v1.pdf", "stage": "done", "done": 154, "total": 154, "secs": 200.0, "t0": 1},
             "v2": {"file": "Set v2.pdf", "stage": "enriching", "done": 128, "total": 152, "secs": None, "t0": 1e9}}
    status, unfinished, posts = {"aaa": {"state": "indexing", "files": files}}, ["Old"], []

    def fake(method, url, **k):
        if method == "post":
            posts.append((url.rsplit("/", 1)[-1], k.get("data")))
        data = ["Old", "aaa"] if url.endswith("/domains") else unfinished if url.endswith("/unfinished") else status
        return SimpleNamespace(json=lambda: data, raise_for_status=lambda: None)

    with patch("requests.request", fake):
        def open_app(domain):
            st.cache_data.clear()
            at = AppTest.from_file(str(ROOT / "frontend/app.py"), default_timeout=30)
            at.session_state["domain"] = domain
            at.run()
            assert not at.exception, at.exception
            return at

        at = open_app(None)
        bars = [p.proto.text for p in at.get("progress")]
        assert any("Set v2.pdf" in b and "128/152" in b for b in bars), bars
        next(b for b in at.button if b.label == "Continue indexing").click().run()
        assert posts == [("continue", {"domain": "Old"})], posts
        assert open_app("aaa").get("progress") and not open_app("Old").get("progress")
        at = open_app(None)
        status["aaa"]["state"] = "done"
        at.run()
        assert any("ready" in s.value for s in at.success)
        next(b for b in at.button if b.label == "Upload documents").click().run()
        radios = [r for r in at.radio if r.label == "What are these files?"]
        assert len(radios) == 2 and all(r.value == "Blueprint" for r in radios)
    print("ok  4. progress and Continue on the start screen, file type in both upload tabs")


if __name__ == "__main__":
    test_workers_end_with_the_app()
    test_backend()
    os.chdir(ROOT)
    test_frontend()
    print("\nALL OK")
