"""Train a CellCanvas service on a copick painting and publish its prediction as a proposal.

The CellCanvas service (``python -m cellcanvas_service`` from cellcanvas/cellcanvas) serves datasets prepared from
one copick reconstruction with saved features. This client talks to it over HTTP with the standard library, so a
napari/copick environment (Zarr 2) needs nothing from the service environment (Zarr 3)::

    service = CellCanvasService("http://127.0.0.1:8001", "organelles")
    painting = Painting.open(run, 40.048, "denoised", "painting", "kyle", "1")
    service.train(painting)                       # uploads the painting at its revision, on its grid
    prediction = service.predict()                # completed, digest-checked, tied to that revision
    proposal = service.publish(prediction, run, 40.048, "cellcanvas-prediction", "cellcanvas", "1")
    painting.accept(painting.revision, proposal, proposal_label=2, label=2, digest=proposal.record["digest"])

The service refuses a painting on another voxel grid or with undeclared classes, and only serves a prediction
once it is complete; :meth:`CellCanvasService.predict` additionally checks the bytes against the service's
result record before anything is written to copick.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass

import numpy as np

from napari_copick.painting import Painting, PaintingError, Proposal


class CellCanvasError(RuntimeError):
    """A refused or failed service operation; ``code`` is the service's error code when it gave one."""

    def __init__(self, message: str, code: str = "service_error"):
        super().__init__(message)
        self.code = code


def _service_digest(array: np.ndarray) -> str:
    """The service's label digest: sha256 of dtype, shape and C-order bytes."""
    array = np.ascontiguousarray(array)
    header = f"{array.dtype.str}:{','.join(map(str, array.shape))}:".encode()
    return "sha256:" + hashlib.sha256(header + array.tobytes()).hexdigest()


@dataclass
class Prediction:
    """A completed prediction read from the service and checked against its result record."""

    labels: np.ndarray
    version: int
    painting_revision: str
    record: dict


class CellCanvasService:
    """One dataset of a running CellCanvas service."""

    def __init__(self, url: str, dataset: str, timeout: float = 30.0):
        self.url = url.rstrip("/")
        self.dataset = dataset
        self.timeout = timeout
        self._painting_revision = None

    # transport ------------------------------------------------------------------------------------------------

    def _send(self, method: str, path: str = "", body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            f"{self.url}/datasets/{self.dataset}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read(), response.headers
        except urllib.error.HTTPError as error:
            try:
                detail = json.loads(error.read() or b"{}")
            except ValueError:
                detail = {}
            raise CellCanvasError(
                f"{method} {path or '/'}: {detail.get('error') or error.reason}",
                detail.get("code", "http_error"),
            ) from None
        except urllib.error.URLError as error:
            raise CellCanvasError(
                f"CellCanvas service unreachable at {self.url}: {error.reason}",
                "unreachable",
            ) from None

    def _call(self, method: str, path: str = "", body=None) -> dict:
        return json.loads(self._send(method, path, body)[0])

    def _run(self, operation: str, service_id: str, options: dict, timeout: float) -> dict:
        receipt = self._call(
            "POST",
            f"/{operation}",
            {"operation_id": uuid.uuid4().hex, "service_id": service_id, **options},
        )
        deadline = time.monotonic() + timeout
        while receipt["state"] == "running":
            if time.monotonic() > deadline:
                raise CellCanvasError(f"{operation} still running after {timeout:.0f} s.", "timeout")
            time.sleep(0.5)
            receipt = self._call("GET", f"/operations/{receipt['id']}")
        if receipt["state"] != "succeeded":
            raise CellCanvasError(f"{operation} {receipt['state']}: {receipt.get('error')}", receipt["state"])
        return receipt["result"]

    # operations -----------------------------------------------------------------------------------------------

    def describe(self) -> dict:
        """The dataset's grid, declared classes, training and prediction state."""
        return self._call("GET")

    def train(self, painting: Painting, label_policy: dict | None = None, timeout: float = 3600) -> dict:
        """Upload ``painting`` at its current revision and train on it."""
        described = self.describe()
        service_id = described["service_id"]
        labels = painting.labels()
        grid = painting.grid
        revision = painting.revision or "content-" + _service_digest(labels)[7:39]
        flat = labels.reshape(-1)
        indices = np.flatnonzero(flat)
        if not indices.size:
            raise CellCanvasError("Paint at least one class before training.", "no_painting")
        self._call(
            "PUT",
            "/painting",
            {
                "service_id": service_id,
                "revision": revision,
                "shape": grid["shape"],
                "scale": grid["scale_angstrom"],
                "translation": grid["translation_angstrom"],
                "indices": indices.tolist(),
                "values": flat[indices].astype(int).tolist(),
            },
        )
        options = {} if label_policy is None else {"label_policy": label_policy}
        result = self._run("train", service_id, options, timeout)
        self._painting_revision = revision
        return {**result, "painting_revision": revision, "painted_voxels": int(indices.size)}

    def predict(self, timeout: float = 3600) -> Prediction:
        """Predict with the trained model and read the completed result, checked against its record."""
        service_id = self.describe()["service_id"]
        result = self._run("predict", service_id, {}, timeout)
        version = int(result["prediction_version"])
        body, headers = self._send("GET", f"/prediction?version={version}")
        shape = [int(v) for v in headers["X-Shape"].split(",")]
        labels = np.frombuffer(body, dtype=headers["X-Dtype"]).reshape(shape)
        record = self._call("GET", f"/results/{version}")["result"]
        if _service_digest(labels) != record.get("digest"):
            raise CellCanvasError("Prediction read does not match its result record.", "not_complete")
        return Prediction(
            labels=labels,
            version=version,
            painting_revision=record.get("painting_revision"),
            record=record,
        )

    def publish(
        self,
        prediction: Prediction,
        run,
        voxel_size: float,
        name: str,
        user_id: str,
        session_id: str,
    ) -> Proposal:
        """Write a completed prediction into its own copick segmentation as a CellCanvas proposal."""
        try:
            return Proposal.publish(
                run,
                voxel_size,
                name,
                user_id,
                session_id,
                prediction.labels,
                "cellcanvas",
                prediction_version=prediction.version,
                painting_revision=prediction.painting_revision,
                service_digest=prediction.record.get("digest"),
                features=(prediction.record.get("features") or {}).get("model"),
            )
        except PaintingError as error:
            raise CellCanvasError(str(error), error.code) from None
