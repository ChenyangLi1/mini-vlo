from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib import error, request

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


@dataclass
class SemanticConfig:
    verifier: str = "qwen3-vl-flash"
    prompt_file: str = "src/module_c/semantic_consistency_v1.txt"
    request_timeout_s: float = 300.0
    qwen_vl_base_url: str = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    qwen_vl_api_key_env: str = "DASHSCOPE_API_KEY"
    qwen_vl_model: str = "qwen3-vl-flash"
    # Kept for backward compatibility with existing qwen3-vl-plus configs.
    qwen3vl_plus_base_url: str = (
        "https://ws-2rplzca29h2l058p.cn-beijing.maas.aliyuncs.com/"
        "compatible-mode/v1"
    )
    qwen3vl_plus_api_key_env: str = "QWEN3VL_PLUS_API_KEY"
    qwen3vl_plus_model: str = "qwen3-vl-plus"
    local_video_max_upload_mb: float = 18.0
    local_video_frame_count: int = 8
    local_video_frame_jpeg_quality: int = 85


class SemanticVerifier(Protocol):
    def verify(self, video_path: str, text: str) -> dict[str, object]:
        ...

    def verify_many(
        self,
        video_paths: dict[str, str],
        text: str,
    ) -> dict[str, object]:
        ...


def _heuristic_result(text: str) -> dict[str, object]:
    text_len = len(text.strip())
    keywords = [
        "grasp",
        "place",
        "move",
        "lift",
        "turn",
        "approach",
        "contact",
        "open",
        "close",
        "pick",
        "drawer",
        "handle",
    ]
    lowered = text.lower()
    keyword_count = sum(1 for word in keywords if word in lowered)
    confidence = min(1.0, 0.15 + 0.02 * text_len + 0.08 * keyword_count)
    if confidence > 0.75:
        label = "consistent"
        errors: list[str] = []
    elif confidence > 0.45:
        label = "uncertain"
        errors = ["low_detail"]
    else:
        label = "inconsistent"
        errors = ["possible_hallucination"]
    return {
        "label": label,
        "confidence": confidence,
        "error_types": errors,
        "suggested_text": text.strip(),
    }


def _failure_result(provider: str, reason: str) -> dict[str, object]:
    """Production failures are uncertain and can never be promoted to keep."""
    return {
        "label": "uncertain",
        "confidence": 0.0,
        "error_types": [reason],
        "suggested_text": "",
        "verifier": f"{provider}_failed",
        "request_failed": True,
    }


def _normalize_result(raw: dict[str, Any], provider: str) -> dict[str, object]:
    if "result" in raw and isinstance(raw["result"], dict):
        raw = raw["result"]
    label = str(raw.get("label", "uncertain"))
    if label not in {"consistent", "uncertain", "inconsistent"}:
        label = "uncertain"
    try:
        confidence = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    error_types = raw.get("error_types", [])
    if not isinstance(error_types, list):
        error_types = [str(error_types)]
    error_types = [str(item) for item in error_types]
    suggested_text = str(raw.get("suggested_text", "")).strip()
    return {
        "label": label,
        "confidence": confidence,
        "error_types": error_types,
        "suggested_text": suggested_text,
        "verifier": provider,
    }


class MockVLMVerifier:
    """Offline verifier for debugging the refinement flow."""

    def verify(self, video_path: str, text: str) -> dict[str, object]:
        result = _heuristic_result(text)
        result["verifier"] = "mock"
        return result

    def verify_many(
        self,
        video_paths: dict[str, str],
        text: str,
    ) -> dict[str, object]:
        result = self.verify(next(iter(video_paths.values()), ""), text)
        result["fusion_strategy"] = "joint_multiview_mock"
        result["views"] = sorted(video_paths)
        return result


class _HTTPJSONVerifier:
    def __init__(
        self,
        provider: str,
        endpoint: str,
        api_key_env: str,
        timeout_s: float,
        prompt_text: str,
    ):
        self.provider = provider
        self.endpoint = endpoint
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self.prompt_text = prompt_text

    def verify(self, video_path: str, text: str) -> dict[str, object]:
        if not self.endpoint:
            return _failure_result(
                self.provider,
                f"{self.provider}_endpoint_missing",
            )

        payload = {
            "video_path": video_path,
            "text": text,
            "task": "semantic_consistency",
            "instruction": self.prompt_text,
        }
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv(self.api_key_env, "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            return _normalize_result(raw, self.provider)
        except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError):
            return _failure_result(
                self.provider,
                f"{self.provider}_request_failed",
            )

    def verify_many(
        self,
        video_paths: dict[str, str],
        text: str,
    ) -> dict[str, object]:
        if not self.endpoint:
            return _failure_result(
                self.provider,
                f"{self.provider}_endpoint_missing",
            )

        payload = {
            "video_paths": video_paths,
            "text": text,
            "task": "joint_multiview_semantic_consistency",
            "instruction": self.prompt_text,
        }
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv(self.api_key_env, "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            result = _normalize_result(raw, self.provider)
            result["fusion_strategy"] = "joint_multiview"
            result["views"] = sorted(video_paths)
            return result
        except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError):
            return _failure_result(
                self.provider,
                f"{self.provider}_request_failed",
            )


class _DashScopeCompatVerifier:
    def __init__(
        self,
        provider: str,
        base_url: str,
        api_key_env: str,
        timeout_s: float,
        prompt_text: str,
        model: str,
        local_video_max_upload_mb: float,
        local_video_frame_count: int,
        local_video_frame_jpeg_quality: int,
    ):
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self.prompt_text = prompt_text
        self.model = model
        self.local_video_max_upload_bytes = max(
            1,
            int(local_video_max_upload_mb * 1024 * 1024),
        )
        self.local_video_frame_count = max(1, int(local_video_frame_count))
        self.local_video_frame_jpeg_quality = max(
            1,
            min(100, int(local_video_frame_jpeg_quality)),
        )

    def _fallback(self, text: str, reason: str) -> dict[str, object]:
        return _failure_result(self.provider, reason)

    @staticmethod
    def _local_file_to_data_url(path: Path) -> str:
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _image_bytes_to_data_url(image_bytes: bytes) -> str:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _sample_video_frames(self, path: Path) -> list[str]:
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("opencv_python_missing") from exc

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError("local_video_open_failed")

        try:
            frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_total <= 0:
                raise RuntimeError("local_video_frame_count_missing")

            if self.local_video_frame_count == 1:
                frame_indexes = [frame_total // 2]
            else:
                frame_indexes = [
                    round(i * (frame_total - 1) / (self.local_video_frame_count - 1))
                    for i in range(self.local_video_frame_count)
                ]

            frames: list[str] = []
            encode_params = [
                int(cv2.IMWRITE_JPEG_QUALITY),
                self.local_video_frame_jpeg_quality,
            ]
            for frame_index in frame_indexes:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                ok, encoded = cv2.imencode(".jpg", frame, encode_params)
                if ok:
                    frames.append(self._image_bytes_to_data_url(encoded.tobytes()))

            if not frames:
                raise RuntimeError("local_video_frame_extract_failed")
            return frames
        finally:
            cap.release()

    def _build_local_video_content(
        self,
        video_path: str,
    ) -> tuple[list[dict[str, Any]], str | None]:
        path = Path(video_path)
        if not path.exists() or not path.is_file():
            return [
                {
                    "type": "text",
                    "text": (
                        "Video path is local but the file was not found: "
                        f"{video_path}. Judge conservatively."
                    ),
                }
            ], "local_video_not_found"

        if path.stat().st_size <= self.local_video_max_upload_bytes:
            return [
                {
                    "type": "video_url",
                    "video_url": {"url": self._local_file_to_data_url(path)},
                }
            ], None

        try:
            frames = self._sample_video_frames(path)
        except RuntimeError as exc:
            return [
                {
                    "type": "text",
                    "text": (
                        "Video path is local but it exceeded the inline upload limit "
                        f"and frame extraction failed ({exc}). Judge conservatively. "
                        f"Path: {video_path}"
                    ),
                }
            ], str(exc)

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "The local video exceeded the inline upload limit, so the following "
                    "uniformly sampled frames are provided in temporal order."
                ),
            }
        ]
        content.extend(
            {"type": "image_url", "image_url": {"url": frame}} for frame in frames
        )
        return content, None

    @staticmethod
    def _extract_message_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if "text" in item:
                        parts.append(str(item["text"]))
                    elif item.get("type") == "output_text" and "text" in item:
                        parts.append(str(item["text"]))
            return "\n".join(parts)
        return str(content)

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any] | None:
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None

    def _media_content(
        self,
        video_path: str,
    ) -> tuple[list[dict[str, Any]], str | None]:
        if video_path.startswith(("http://", "https://")):
            return [
                {"type": "video_url", "video_url": {"url": video_path}}
            ], None
        return self._build_local_video_content(video_path)

    def _request(
        self,
        user_content: list[dict[str, Any]],
        text: str,
    ) -> dict[str, object]:
        if not self.base_url:
            return self._fallback(text, f"{self.provider}_base_url_missing")

        api_key = os.getenv(self.api_key_env, "")
        if not api_key:
            return self._fallback(text, f"{self.provider}_api_key_missing")

        endpoint = self.base_url
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a strict semantic-video consistency verifier.",
                },
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        req = request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            choices = raw.get("choices", [])
            if not isinstance(choices, list) or not choices:
                return self._fallback(text, f"{self.provider}_invalid_response")
            msg = choices[0].get("message", {})
            msg_content = self._extract_message_text(msg.get("content", ""))
            parsed = self._parse_json_object(msg_content)
            if parsed is None:
                return self._fallback(text, f"{self.provider}_response_parse_failed")
            return _normalize_result(parsed, self.provider)
        except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError):
            return self._fallback(text, f"{self.provider}_request_failed")

    def verify(self, video_path: str, text: str) -> dict[str, object]:
        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"{self.prompt_text}\n\n"
                    "Please evaluate the semantic consistency between the video and "
                    "text.\n"
                    "Return ONLY a JSON object with keys: label "
                    "(consistent|uncertain|inconsistent), confidence (0-1), "
                    "error_types (list), suggested_text (string).\n"
                    f"Text: {text}"
                ),
            }
        ]
        media_content, media_reason = self._media_content(video_path)
        if media_reason:
            return self._fallback(text, media_reason)
        user_content.extend(media_content)
        return self._request(user_content, text)

    def verify_many(
        self,
        video_paths: dict[str, str],
        text: str,
    ) -> dict[str, object]:
        ordered_views = sorted(video_paths.items())
        if not ordered_views:
            return self._fallback(text, "multiview_paths_missing")

        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"{self.prompt_text}\n\n"
                    "Jointly evaluate the candidate text against ALL synchronized "
                    "camera views below. Treat the views as complementary evidence "
                    "of the same event, not as independent samples. A claim may be "
                    "supported by one view even when it is occluded or outside the "
                    "frame in another view. Return one fused judgment only.\n"
                    "Return ONLY a JSON object with keys: label "
                    "(consistent|uncertain|inconsistent), confidence (0-1), "
                    "error_types (list), suggested_text (string).\n"
                    f"Candidate text: {text}\n"
                    f"Available views: {', '.join(view_id for view_id, _ in ordered_views)}"
                ),
            }
        ]
        for view_id, video_path in ordered_views:
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        f"VIEW {view_id}: synchronized camera stream for the same "
                        "manipulation event."
                    ),
                }
            )
            media_content, media_reason = self._media_content(video_path)
            if media_reason:
                return self._fallback(text, f"{view_id}:{media_reason}")
            user_content.extend(media_content)

        result = self._request(user_content, text)
        result["fusion_strategy"] = "joint_multiview"
        result["views"] = [view_id for view_id, _ in ordered_views]
        return result


def _resolve_prompt_path(prompt_file: str) -> Path:
    path = Path(prompt_file)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def _load_prompt_text(prompt_file: str) -> str:
    path = _resolve_prompt_path(prompt_file)
    if not path.exists():
        return (
            "You are a strict semantic-video consistency verifier for manipulation "
            "clips. Judge whether the text accurately describes the visible "
            "manipulation steps, object interactions, and temporal order in the "
            "video. Return JSON with keys: label "
            "(consistent|uncertain|inconsistent), confidence (0-1), error_types "
            "(list), suggested_text (string)."
        )
    return path.read_text(encoding="utf-8").strip()


def build_verifier(cfg: SemanticConfig) -> SemanticVerifier:
    name = cfg.verifier.strip().lower()
    prompt_text = _load_prompt_text(cfg.prompt_file)
    if name == "mock":
        return MockVLMVerifier()
    if name in {"qwen3-vl-flash", "qwen3_vl_flash", "qwen3vlflash"}:
        return _DashScopeCompatVerifier(
            provider="qwen3-vl-flash",
            base_url=os.getenv("DASHSCOPE_BASE_URL", cfg.qwen_vl_base_url),
            api_key_env=cfg.qwen_vl_api_key_env,
            timeout_s=cfg.request_timeout_s,
            prompt_text=prompt_text,
            model=os.getenv("SEMANTIC_JUDGE_MODEL", cfg.qwen_vl_model),
            local_video_max_upload_mb=cfg.local_video_max_upload_mb,
            local_video_frame_count=cfg.local_video_frame_count,
            local_video_frame_jpeg_quality=cfg.local_video_frame_jpeg_quality,
        )
    if name in {"qwen3-vl-plus", "qwen3_vl_plus", "qwen3vlplus"}:
        return _DashScopeCompatVerifier(
            provider="qwen3-vl-plus",
            base_url=cfg.qwen3vl_plus_base_url,
            api_key_env=cfg.qwen3vl_plus_api_key_env,
            timeout_s=cfg.request_timeout_s,
            prompt_text=prompt_text,
            model=cfg.qwen3vl_plus_model,
            local_video_max_upload_mb=cfg.local_video_max_upload_mb,
            local_video_frame_count=cfg.local_video_frame_count,
            local_video_frame_jpeg_quality=cfg.local_video_frame_jpeg_quality,
        )
    raise ValueError(
        f"Unknown semantic verifier: {cfg.verifier}. "
        "Supported: mock, qwen3-vl-flash, qwen3-vl-plus."
    )


def verify_semantic_consistency(
    video_path: str,
    text: str,
    verifier: SemanticVerifier,
) -> tuple[dict[str, object], list[str]]:
    result = verifier.verify(video_path=video_path, text=text)
    reasons: list[str] = list(result.get("error_types", []))
    verifier_name = str(result.get("verifier", ""))
    if result.get("request_failed") or verifier_name.endswith(("_failed", "_fallback")):
        result["label"] = "uncertain"
        result["confidence"] = 0.0
        reasons.append("semantic_verifier_failed")
    label = result.get("label")
    if label == "uncertain":
        reasons.append("semantic_uncertain")
    if label == "inconsistent":
        reasons.append("semantic_mismatch")
    return result, sorted(set(reasons))


def verify_multiview_semantic_consistency(
    video_paths: dict[str, str],
    text: str,
    verifier: SemanticVerifier,
) -> tuple[dict[str, object], list[str]]:
    """Judge all synchronized views jointly in one multimodal request."""
    normalized_paths = {
        str(view_id): str(video_path)
        for view_id, video_path in sorted(video_paths.items())
        if str(video_path).strip()
    }
    verify_many = getattr(verifier, "verify_many", None)
    if not normalized_paths:
        result = _failure_result("joint_multiview", "multiview_paths_missing")
    elif not callable(verify_many):
        result = _failure_result(
            "joint_multiview",
            "joint_multiview_verifier_unsupported",
        )
    else:
        result = verify_many(video_paths=normalized_paths, text=text)

    result["fusion_strategy"] = "joint_multiview"
    result["views"] = sorted(normalized_paths)
    reasons: list[str] = list(result.get("error_types", []))
    verifier_name = str(result.get("verifier", ""))
    if result.get("request_failed") or verifier_name.endswith(
        ("_failed", "_fallback")
    ):
        result["label"] = "uncertain"
        result["confidence"] = 0.0
        reasons.append("semantic_verifier_failed")
    label = result.get("label")
    if label == "uncertain":
        reasons.append("semantic_uncertain")
    if label == "inconsistent":
        reasons.append("semantic_mismatch")
    return result, sorted(set(reasons))

