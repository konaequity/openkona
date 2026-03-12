from __future__ import annotations

from pathlib import Path


def test_repo_documents_karls_six_capabilities_in_verification_matrix():
    matrix = Path("/Users/joeyroth/Desktop/openkona/docs/verification-matrix.md")
    text = matrix.read_text()
    for phrase in [
        "Architecture",
        "Data synthesis",
        "Training",
        "Inference",
        "Evaluation",
    ]:
        assert phrase in text, f"verification-matrix.md must describe {phrase} coverage"


def test_repo_contains_spec_first_tests_before_implementation():
    specs_dir = Path("/Users/joeyroth/Desktop/openkona/tests/specs")
    test_files = sorted(specs_dir.glob("test_*.py"))
    assert len(test_files) >= 5, "Expected a substantial spec-first suite before implementation begins"
