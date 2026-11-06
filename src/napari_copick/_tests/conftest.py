import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
from copick.models import PickableObject
from copick.ops.open import new_config

SHAPE = (8, 12, 16)


@pytest.fixture
def run(tmp_path):
    root = new_config(
        str(tmp_path / "config.json"),
        str(tmp_path / "overlay"),
        pickable_objects=[
            PickableObject(name="background", is_particle=False, label=1, color=[0, 0, 255, 255]),
            PickableObject(name="membrane", is_particle=False, label=2, color=[255, 0, 0, 255]),
        ],
    )
    run = root.new_run("TS-1")
    tomogram = run.new_voxel_spacing(10.0).new_tomogram("denoised")
    tomogram.from_numpy(np.random.default_rng(0).random(SHAPE, dtype=np.float32), levels=1)
    return run


def service_digest(array):
    array = np.ascontiguousarray(array)
    header = f"{array.dtype.str}:{','.join(map(str, array.shape))}:".encode()
    return "sha256:" + hashlib.sha256(header + array.tobytes()).hexdigest()


class FakeCellCanvas:
    """The CellCanvas service routes this client uses; "prediction" fills unknown voxels with the last class."""

    def __init__(self, shape, scale):
        self.shape, self.scale = list(shape), list(scale)
        self.painting = None
        self.revision = None
        self.version = 0
        self.prediction = None
        self.corrupt = False
        self.uploads = []


def handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, body, status=200):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self):
            return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/datasets/demo":
                return self._json({"service_id": "s1", "painting": {"shape": state.shape, "revision": state.revision}})
            if url.path.startswith("/datasets/demo/results/"):
                return self._json(
                    {
                        "result": {
                            "digest": service_digest(state.prediction),
                            "painting_revision": state.revision,
                            "features": {"model": "fake"},
                        },
                    },
                )
            if url.path == "/datasets/demo/prediction":
                if int(parse_qs(url.query)["version"][0]) != state.version:
                    return self._json({"error": "not current", "code": "not_ready"}, 409)
                data = state.prediction.copy()
                if state.corrupt:
                    data.flat[0] ^= 1
                body = data.tobytes()
                self.send_response(200)
                self.send_header("X-Shape", ",".join(map(str, data.shape)))
                self.send_header("X-Dtype", str(data.dtype))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return None
            return self._json({"error": "not found", "code": "not_found"}, 404)

        def do_PUT(self):
            body = self._body()
            state.uploads.append(body)
            if body["shape"] != state.shape or body["scale"] != state.scale:
                return self._json({"error": "grid differs", "code": "geometry_mismatch"}, 409)
            painting = np.zeros(int(np.prod(state.shape)), np.uint8)
            painting[body["indices"]] = body["values"]
            state.painting, state.revision = painting.reshape(state.shape), body["revision"]
            return self._json({"painting_revision": body["revision"], "voxels": len(body["indices"])})

        def do_POST(self):
            self._body()
            if self.path.endswith("/train"):
                return self._json({"id": "t", "state": "succeeded", "result": {"inferred": []}}, 202)
            state.version += 1
            fill = int(state.painting.max())
            state.prediction = np.where(state.painting == 0, fill, state.painting).astype(np.uint8)
            return self._json({"id": "p", "state": "succeeded", "result": {"prediction_version": state.version}}, 202)

    return Handler


@pytest.fixture
def fake_cellcanvas():
    state = FakeCellCanvas(SHAPE, [10.0, 10.0, 10.0])
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield state
    server.shutdown()
