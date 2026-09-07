import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import queue
import time

import pytest


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    executable = os.environ.get("KAPSL_PYTHON_TEST_SERVER")
    if not executable:
        pytest.fail("Set KAPSL_PYTHON_TEST_SERVER to the built python_test_server example")
    root = tmp_path_factory.mktemp("server")
    stderr = (root / "stderr").open("w")
    process = subprocess.Popen(
        [executable, str(root)], stdout=subprocess.PIPE, stderr=stderr, text=True,
    )
    lines = queue.Queue()
    threading.Thread(
        target=lambda: lines.put(process.stdout.readline()), daemon=True,
    ).start()
    try:
        try:
            line = lines.get(timeout=15)
        except queue.Empty:
            pytest.fail("Test server did not become ready")
        if not line:
            stderr.flush()
            pytest.fail((root / "stderr").read_text())
        yield json.loads(line)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT) if os.name != "nt" else process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        process.stdout.close()
        stderr.close()


@pytest.fixture
def released(server):
    def wait(request_id):
        end = time.monotonic() + 3
        while time.monotonic() < end:
            if f"released:{request_id}" in Path(server["events"]).read_text().splitlines():
                return
            time.sleep(0.01)
        pytest.fail(f"Backend request was not released: {request_id}")
    return wait
