from __future__ import annotations

import unittest
from unittest.mock import patch

from src.module_c.semantic_consistency import (
    SemanticConfig,
    build_verifier,
    verify_multiview_semantic_consistency,
)


class JointVerifier:
    def __init__(self):
        self.calls = []

    def verify(self, video_path, text):
        raise AssertionError("Single-view verification must not be used")

    def verify_many(self, video_paths, text):
        self.calls.append((video_paths, text))
        return {
            "label": "consistent",
            "confidence": 0.9,
            "error_types": [],
            "suggested_text": text,
            "verifier": "joint-test",
        }


class SemanticVerifierConfigTest(unittest.TestCase):
    def test_qwen3_vl_flash_verifier_uses_generic_dashscope_config(self):
        with patch.dict(
            "os.environ",
            {
                "DASHSCOPE_BASE_URL": "https://example.test/compatible-mode/v1",
                "VLM_MODEL": "qwen3-vl-flash",
            },
        ):
            verifier = build_verifier(
                SemanticConfig(
                    verifier="qwen3-vl-flash",
                    qwen_vl_api_key_env="TEST_QWEN_API_KEY",
                )
            )

        self.assertEqual(verifier.provider, "qwen3-vl-flash")
        self.assertEqual(verifier.base_url, "https://example.test/compatible-mode/v1")
        self.assertEqual(verifier.api_key_env, "TEST_QWEN_API_KEY")
        self.assertEqual(verifier.model, "qwen3-vl-flash")

    def test_qwen3_vl_plus_remains_supported(self):
        verifier = build_verifier(SemanticConfig(verifier="qwen3-vl-plus"))

        self.assertEqual(verifier.provider, "qwen3-vl-plus")
        self.assertEqual(verifier.model, "qwen3-vl-plus")

    def test_multiview_uses_one_joint_request(self):
        verifier = JointVerifier()

        result, reasons = verify_multiview_semantic_consistency(
            {
                "fixed": "fixed.mp4",
                "ego": "ego.mp4",
            },
            "place the bowl on the plate",
            verifier,
        )

        self.assertEqual(len(verifier.calls), 1)
        self.assertEqual(
            verifier.calls[0][0],
            {"ego": "ego.mp4", "fixed": "fixed.mp4"},
        )
        self.assertEqual(result["label"], "consistent")
        self.assertEqual(result["fusion_strategy"], "joint_multiview")
        self.assertNotIn("per_view", result)
        self.assertEqual(reasons, [])


if __name__ == "__main__":
    unittest.main()
