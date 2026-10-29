"""HTTP contract tests against the FastAPI app (in-memory backend)."""

from __future__ import annotations

import base64
import io

import pytest
from app.main import create_app
from fastapi.testclient import TestClient

from tests.conftest import make_settings

API = "/api/v1"


@pytest.fixture(scope="module")
def client():
    app = create_app(make_settings())
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_readiness(client: TestClient):
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["checks"]["repository"] is True


def test_info_reports_free_provider_chain(client: TestClient):
    response = client.get(f"{API}/info")
    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "memory"
    assert payload["chatProviders"] == ["extractive"]
    assert payload["embeddingDimensions"] >= 64


def test_seeded_documents_are_listed(client: TestClient):
    response = client.get(f"{API}/documents?limit=50")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 3
    assert len(payload["documents"]) == 3
    first = payload["documents"][0]
    assert first["status"] == "ready"
    assert first["chunkCount"] > 0


def test_chat_returns_citations_and_grounding(client: TestClient):
    response = client.post(
        f"{API}/chat",
        json={
            "query": (
                "What was the consolidated net revenue and adjusted EBITDA margin "
                "for Q3 2026?"
            ),
            "topK": 4,
            "similarityThreshold": 0.2,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["abstained"] is False
    assert payload["citations"]
    assert 0 <= payload["groundingScore"] <= 100
    assert payload["latencyMs"] >= 0
    assert payload["citations"][0]["documentTitle"]


def test_chat_abstains_for_out_of_domain_question(client: TestClient):
    response = client.post(
        f"{API}/chat",
        json={"query": "What were the lunar space flight revenues in Q3 2026?"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["abstained"] is True
    assert payload["citations"] == []


def test_chat_rejects_blank_query(client: TestClient):
    response = client.post(f"{API}/chat", json={"query": "   "})
    assert response.status_code == 422


def test_upload_text_document_roundtrip(client: TestClient):
    response = client.post(
        f"{API}/documents/upload",
        json={
            "title": "Orion Logistics Addendum",
            "filename": "orion.txt",
            "textContent": (
                "The Orion programme freight budget is $12.5 million. "
                "Deliveries begin in April 2027.\fThe Orion audit closes in June 2027."
            ),
        },
    )
    assert response.status_code == 201
    document = response.json()["document"]
    assert document["pageCount"] == 2
    assert document["status"] == "ready"

    detail = client.get(f"{API}/documents/{document['id']}")
    assert detail.status_code == 200
    assert len(detail.json()["document"]["pages"]) == 2

    deleted = client.delete(f"{API}/documents/{document['id']}")
    assert deleted.status_code == 204
    assert client.get(f"{API}/documents/{document['id']}").status_code == 404


def _build_pdf(text: str) -> bytes:
    """Assemble a minimal, xref-correct single-page PDF with extractable text."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode()
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    ]

    output = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_offset = len(output)
    output += f"xref\n0 {len(objects) + 1}\n".encode()
    output += b"0000000000 65535 f \n"
    for offset in offsets:
        output += f"{offset:010d} 00000 n \n".encode()
    output += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()
    return bytes(output)


def test_upload_pdf_file(client: TestClient):
    content = _build_pdf("Helios budget is 4.2 million USD.")
    response = client.post(
        f"{API}/documents/upload-file",
        files={"file": ("helios.pdf", io.BytesIO(content), "application/pdf")},
        data={"title": "Helios Brief"},
    )
    assert response.status_code == 201, response.text
    document = response.json()["document"]
    assert document["mimeType"] == "application/pdf"
    assert "Helios budget" in document["pages"][0]["text"]
    client.delete(f"{API}/documents/{document['id']}")


def test_upload_rejects_invalid_base64(client: TestClient):
    response = client.post(
        f"{API}/documents/upload",
        json={"title": "Broken", "filename": "broken.pdf", "fileBase64": "not-base64!!"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_BASE64"


def test_upload_accepts_data_uri(client: TestClient):
    payload = base64.b64encode(b"Vega quarterly revenue was 3.1 million.").decode()
    response = client.post(
        f"{API}/documents/upload",
        json={
            "title": "Vega Note",
            "filename": "vega.txt",
            "fileBase64": f"data:text/plain;base64,{payload}",
        },
    )
    assert response.status_code == 201
    client.delete(f"{API}/documents/{response.json()['document']['id']}")


def test_document_not_found_is_structured(client: TestClient):
    response = client.get(f"{API}/documents/{'0' * 8}-0000-0000-0000-{'0' * 12}")
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "DOCUMENT_NOT_FOUND"
    assert body["requestId"]


def test_invalid_uuid_returns_422(client: TestClient):
    response = client.get(f"{API}/documents/not-a-uuid")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_DOCUMENT_ID"


def test_retrieval_debug_endpoint(client: TestClient):
    response = client.post(
        f"{API}/retrieval",
        json={"query": "AES-256 key rotation", "topK": 4},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["citations"]
    assert payload["totalCandidates"] > 0


def test_eval_dataset_and_run(client: TestClient):
    dataset = client.get(f"{API}/eval/dataset")
    assert dataset.status_code == 200
    assert len(dataset.json()["testCases"]) == 8

    run = client.post(
        f"{API}/eval/run",
        json={
            "customTests": [
                {
                    "id": "api-1",
                    "documentTitle": "Apex Holdings Q3 2026 Financial Report",
                    "question": "What was the consolidated net revenue for Q3 2026?",
                    "expectedPassage": "consolidated net revenue of $482.6 million",
                    "expectedPage": 1,
                    "isNegativeTest": False,
                    "description": "API revenue check",
                }
            ]
        },
    )
    assert run.status_code == 200
    report = run.json()["report"]
    assert report["totalTests"] == 1
    assert report["results"][0]["testCaseId"] == "api-1"

    latest = client.get(f"{API}/eval/latest")
    assert latest.json()["report"]["runId"] == report["runId"]
    history = client.get(f"{API}/eval/history")
    assert history.json()["reports"]


def test_export_files_and_zip(client: TestClient):
    files = client.get(f"{API}/export/files")
    assert files.status_code == 200
    paths = {item["path"] for item in files.json()["files"]}
    assert "backend/app/rag/workflow.py" in paths
    assert "backend/app/main.py" in paths

    archive = client.get(f"{API}/export/download")
    assert archive.status_code == 200
    assert archive.headers["content-type"] == "application/zip"
    assert archive.content[:2] == b"PK"


def test_security_and_request_id_headers(client: TestClient):
    response = client.get("/healthz", headers={"x-request-id": "test-trace-1"})
    assert response.headers["x-request-id"] == "test-trace-1"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"


def test_reset_defaults_restores_corpus(client: TestClient):
    response = client.post(f"{API}/documents/reset-defaults")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 3
    assert payload["seeded"] >= 0


def test_eval_run_with_empty_body_uses_default_dataset(client: TestClient):
    run = client.post(f"{API}/eval/run", json={})
    assert run.status_code == 200
    report = run.json()["report"]
    assert report["totalTests"] >= 1


def test_rate_limiting_headers_and_rejection():
    app = create_app(
        make_settings(rate_limit_enabled=True, rate_limit_chat="2/minute")
    )
    with TestClient(app) as rate_client:
        payload = {"query": "What was the consolidated net revenue for Q3 2026?"}
        first = rate_client.post(f"{API}/chat", json=payload)
        assert first.status_code == 200
        assert "x-ratelimit-limit" in {key.lower() for key in first.headers}

        assert rate_client.post(f"{API}/chat", json=payload).status_code == 200
        blocked = rate_client.post(f"{API}/chat", json=payload)
        assert blocked.status_code == 429
        assert blocked.json()["error"]["code"] == "RATE_LIMITED"
        assert blocked.headers["retry-after"]
