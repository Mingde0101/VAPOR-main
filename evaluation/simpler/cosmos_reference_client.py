"""Local Cosmos Serve client for rolling SIMPLER reference generation."""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.error import HTTPError
from urllib.parse import quote, urlparse
from urllib.request import Request, build_opener, ProxyHandler

import numpy as np

from evaluation.simpler.eval_logger import resample_rgb_frames_for_video, write_compatible_mp4


COSMOS3_VIDEO_RES_SIZE_INFO: dict[str, dict[str, tuple[int, int]]] = {
    "256": {
        "1,1": (256, 256), "4,3": (320, 256), "3,4": (256, 320),
        "16,9": (320, 192), "9,16": (192, 320),
    },
    "480": {
        "1,1": (640, 640), "4,3": (736, 544), "3,4": (544, 736),
        "16,9": (832, 480), "9,16": (480, 832),
    },
    "704": {
        "1,1": (960, 960), "4,3": (1088, 832), "3,4": (832, 1088),
        "16,9": (1280, 704), "9,16": (704, 1280),
    },
    "720": {
        "1,1": (960, 960), "4,3": (1104, 832), "3,4": (832, 1104),
        "16,9": (1280, 720), "9,16": (720, 1280),
    },
}


@dataclass
class CosmosReferenceResult:
    sample: Dict[str, Any]
    sample_dir: Path
    npz_path: Path
    mp4_path: Path | None
    compat_mp4_path: Path | None
    remote_outputs: Dict[str, Any]
    elapsed_sec: float


def safe_stem(name: object) -> str:
    text = str(name or "cosmos_reference")
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    return cleaned.strip("_") or "cosmos_reference"


def write_rgb_png(path: str | Path, rgb: np.ndarray) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"RGB image must be [H,W,3], got {arr.shape}.")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    import imageio.v2 as imageio

    imageio.imwrite(out, np.ascontiguousarray(arr))
    return out


def write_rgb_video(
    path: str | Path,
    frames: Iterable[np.ndarray],
    *,
    fps: int,
    source_fps: float | None = None,
) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    frame_list = [np.asarray(frame, dtype=np.uint8) for frame in frames]
    if not frame_list:
        raise ValueError("Cannot write an empty conditioning video.")
    if source_fps is not None:
        frame_list = resample_rgb_frames_for_video(
            frame_list,
            source_fps=float(source_fps),
            target_fps=float(fps),
        )
    write_compatible_mp4(out, frame_list, fps=int(fps))
    return out


def read_video_frames(mp4_path: str | Path) -> np.ndarray:
    path = Path(mp4_path).expanduser().resolve()
    try:
        import imageio.v3 as iio

        frames = np.asarray(iio.imread(path))
    except Exception:
        import imageio.v2 as imageio

        reader = imageio.get_reader(path)
        try:
            frames = np.stack([np.asarray(frame) for frame in reader], axis=0)
        finally:
            reader.close()
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Decoded Cosmos video must be [T,H,W,3], got {frames.shape} from {path}.")
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frames)


def save_compatible_mp4(mp4_path: Path, compat_path: Path) -> Path:
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(mp4_path),
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(compat_path),
    ]
    subprocess.run(cmd, check=True)
    return compat_path


def save_reference_npz(sample: Dict[str, Any], mp4_path: Path, npz_path: Path) -> Path:
    frames = read_video_frames(mp4_path)
    extra = sample.get("extra") if isinstance(sample.get("extra"), dict) else {}
    seed = sample.get("seed", extra.get("seed"))
    np.savez_compressed(
        npz_path,
        frames=frames,
        fps=np.array(float(sample.get("fps", -1.0)), dtype=np.float32),
        prompt=np.array(str(sample.get("prompt", ""))),
        instruction=np.array(str(extra.get("instruction", sample.get("prompt", "")))),
        task_name=np.array(str(extra.get("task_name", ""))),
        seed=np.array(-1 if seed is None else int(seed), dtype=np.int64),
        segment_index=np.array(int(extra.get("segment_index", -1)), dtype=np.int64),
        rollout_step=np.array(int(extra.get("rollout_step", -1)), dtype=np.int64),
        model_mode=np.array(str(sample.get("model_mode", ""))),
        vision_path=np.array(str(sample.get("vision_path", ""))),
    )
    return npz_path


def _request_json(url: str, *, method: str = "GET", payload: Dict[str, Any] | None = None, timeout: int = 1800) -> Any:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with _url_opener(url).open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Cosmos request failed: {exc.code} {exc.reason}: {detail}") from exc


def _request_multipart(
    url: str,
    *,
    fields: Dict[str, Any],
    files: Dict[str, tuple[Path, str]],
    timeout: int = 1800,
) -> tuple[bytes, Dict[str, str]]:
    """POST multipart form data using only the Python standard library."""
    boundary = f"----slot-attention-video-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        if value is None:
            continue
        chunks.extend([
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
            str(value).encode("utf-8"),
            b"\r\n",
        ])
    for name, (path, content_type) in files.items():
        local_path = Path(path).expanduser().resolve()
        chunks.extend([
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{name}"; filename="{local_path.name}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8"),
            local_path.read_bytes(),
            b"\r\n",
        ])
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    data = b"".join(chunks)
    request = Request(
        url,
        data=data,
        headers={
            "Accept": "video/mp4",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(data)),
        },
        method="POST",
    )
    try:
        with _url_opener(url).open(request, timeout=timeout) as response:
            headers = {key.lower(): value for key, value in response.headers.items()}
            return response.read(), headers
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Cosmos multipart request failed: {exc.code} {exc.reason}: {detail}") from exc


def _url_opener(url: str):
    """Bypass configured proxies for local Cosmos services."""
    host = urlparse(url).hostname or ""
    if host in {"127.0.0.1", "localhost", "::1"}:
        no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
        merged = [part.strip() for part in no_proxy.split(",") if part.strip()]
        for local_host in ("127.0.0.1", "localhost", "::1"):
            if local_host not in merged:
                merged.append(local_host)
        os.environ["NO_PROXY"] = ",".join(merged)
        os.environ["no_proxy"] = os.environ["NO_PROXY"]
        return build_opener(ProxyHandler({}))
    return build_opener()


def _download(base_url: str, remote_file: str, local_path: Path, *, timeout: int) -> Path:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{base_url}/outputs/{remote_file}"
    request = Request(url, method="GET")
    try:
        with _url_opener(url).open(request, timeout=timeout) as response:
            local_path.write_bytes(response.read())
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Failed to download Cosmos output {remote_file!r}: {detail}") from exc
    return local_path


def _iter_output_files(sample_outputs: Dict[str, Any]) -> List[str]:
    files: List[str] = []
    for output in sample_outputs.get("outputs", []):
        for item in output.get("files", []):
            files.append(str(item))
    return files


class CosmosReferenceGenerator:
    """Synchronous client for the local Cosmos Ray Serve /generate endpoint.

    ``video2video`` uses the same ``vision_path`` field as ``image2video``; the
    only difference is that the path points to an MP4 conditioning clip and the
    sample's ``model_mode`` is set to ``video2video``.
    """

    def __init__(
        self,
        *,
        output_root: str | Path,
        host: str = "127.0.0.1",
        port: int = 8000,
        timeout: int = 1800,
        model: str = "Cosmos3-Nano",
        fps: int = 10,
        duration_sec: float = 2.0,
        resolution: str = "480",
        aspect_ratio: str = "4,3",
        num_steps: int = 35,
        guidance: float = 6.0,
        shift: float = 10.0,
        sigma_max: float = 80.0,
        normalize_cfg: bool = False,
        prompt_upsampling: bool = False,
        backend: str = "cosmos_framework",
        vllm_size: str = "",
        negative_prompt: str = "",
    ) -> None:
        self.output_root = Path(output_root).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.base_url = f"http://{host}:{int(port)}"
        self.timeout = int(timeout)
        if backend not in {"cosmos_framework", "vllm_omni"}:
            raise ValueError(f"Unknown Cosmos backend {backend!r}.")
        self.backend = backend
        self.model = model
        self.fps = int(fps)
        self.duration_sec = float(duration_sec)
        self.resolution = str(resolution)
        self.aspect_ratio = str(aspect_ratio)
        self.num_steps = int(num_steps)
        self.guidance = float(guidance)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.normalize_cfg = bool(normalize_cfg)
        self.prompt_upsampling = bool(prompt_upsampling)
        self.vllm_size = str(vllm_size or "")
        self.negative_prompt = str(negative_prompt or "")
        self._models: List[str] | None = None

    def check_server(self) -> List[str]:
        if self.backend == "vllm_omni":
            info = _request_json(f"{self.base_url}/v1/models", timeout=self.timeout)
            raw_models = info.get("data", []) if isinstance(info, dict) else []
            models = [item.get("id", "") for item in raw_models if isinstance(item, dict)]
        else:
            info = _request_json(f"{self.base_url}/models", timeout=self.timeout)
            models = info if isinstance(info, list) else info.get("models", [])
        available = [str(model) for model in models]
        matches = [name for name in available if name == self.model or Path(name).name == self.model]
        if len(matches) != 1:
            raise RuntimeError(f"Expected Cosmos model {self.model!r} at {self.base_url}. Served: {available}")
        self.model = matches[0]
        self._models = available
        return self._models

    def build_sample(
        self,
        *,
        name: str,
        model_mode: str,
        prompt: str,
        vision_path: str | Path,
        seed: int,
        extra: Dict[str, Any],
    ) -> Dict[str, Any]:
        if model_mode not in {"image2video", "video2video"}:
            raise ValueError(f"Rolling reference generation only supports image2video/video2video, got {model_mode}.")
        vision = Path(vision_path).expanduser().resolve()
        if not vision.is_file():
            raise FileNotFoundError(f"Missing Cosmos vision_path: {vision}")
        return {
            "name": name,
            "model": self.model,
            "model_mode": model_mode,
            "prompt": prompt,
            "vision_path": str(vision),
            "resolution": self.resolution,
            "aspect_ratio": self.aspect_ratio,
            "fps": self.fps,
            # Include both endpoints: 2.0s at 10Hz requests frames at
            # timestamps 0.0, 0.1, ..., 2.0.
            "num_frames": max(1, int(round(self.fps * self.duration_sec)) + 1),
            "num_steps": self.num_steps,
            "guidance": self.guidance,
            "shift": self.shift,
            "sigma_max": self.sigma_max,
            "normalize_cfg": self.normalize_cfg,
            "seed": int(seed),
            "num_outputs": 1,
            "enable_sound": False,
            "prompt_upsampling": self.prompt_upsampling,
            "extra": extra,
        }

    def _sample_for_server(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if self._models is None:
            self.check_server()
        sample = dict(sample, model=self.model)
        if self.backend == "vllm_omni":
            return self._sample_for_vllm_omni(sample)
        return sample

    def _vllm_size(self) -> str:
        if self.vllm_size:
            return self.vllm_size
        bucket = COSMOS3_VIDEO_RES_SIZE_INFO.get(str(self.resolution), {}).get(str(self.aspect_ratio))
        if bucket is not None:
            width, height = bucket
            return f"{width}x{height}"

        resolution = int(self.resolution)
        try:
            aspect_w, aspect_h = [int(part.strip()) for part in self.aspect_ratio.split(",", maxsplit=1)]
        except Exception as exc:
            raise ValueError(f"Invalid aspect_ratio {self.aspect_ratio!r}; expected '<width>,<height>'.") from exc
        if aspect_w <= 0 or aspect_h <= 0:
            raise ValueError(f"Invalid aspect_ratio {self.aspect_ratio!r}; both dimensions must be positive.")
        if aspect_w >= aspect_h:
            height = resolution
            width = int(round(float(resolution) * float(aspect_w) / float(aspect_h)))
        else:
            width = resolution
            height = int(round(float(resolution) * float(aspect_h) / float(aspect_w)))
        width -= width % 2
        height -= height % 2
        return f"{width}x{height}"

    def _sample_for_vllm_omni(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        request_sample = dict(sample)
        extra_params = {
            "use_resolution_template": False,
            "use_duration_template": False,
            "guardrails": False,
            "model_mode": request_sample.get("model_mode", ""),
            "normalize_cfg": bool(request_sample.get("normalize_cfg", False)),
            "prompt_upsampling": bool(request_sample.get("prompt_upsampling", False)),
            "source": "rolling Cosmos reference",
        }
        return {
            "model": request_sample["model"],
            "prompt": request_sample.get("prompt", ""),
            "negative_prompt": self.negative_prompt,
            "size": self._vllm_size(),
            "num_frames": int(request_sample.get("num_frames", 1)),
            "fps": int(request_sample.get("fps", self.fps)),
            "num_inference_steps": int(request_sample.get("num_steps", self.num_steps)),
            "guidance_scale": float(request_sample.get("guidance", self.guidance)),
            "flow_shift": float(request_sample.get("shift", self.shift)),
            "seed": int(request_sample.get("seed", 0)),
            "extra_params": json.dumps(extra_params, ensure_ascii=False),
            "vision_path": request_sample.get("vision_path", ""),
            "model_mode": request_sample.get("model_mode", ""),
        }

    def _submit_vllm_omni(
        self,
        sample: Dict[str, Any],
        *,
        sample_dir: Path,
        stem: str,
    ) -> CosmosReferenceResult:
        request_sample = self._sample_for_server(sample)
        (sample_dir / "sample_args.submitted.json").write_text(
            json.dumps(request_sample, indent=2, ensure_ascii=False) + "\n"
        )
        vision_path = Path(str(request_sample.pop("vision_path"))).expanduser().resolve()
        if not vision_path.is_file():
            raise FileNotFoundError(f"Missing vLLM-Omni input_reference: {vision_path}")
        content_type = "video/mp4" if vision_path.suffix.lower() in {".mp4", ".mov", ".mkv"} else "image/png"
        fields = {key: value for key, value in request_sample.items() if key != "model_mode"}
        started = time.time()
        body, headers = _request_multipart(
            f"{self.base_url}/v1/videos/sync",
            fields=fields,
            files={"input_reference": (vision_path, content_type)},
            timeout=self.timeout,
        )
        elapsed = time.time() - started
        content_type_header = headers.get("content-type", "")
        if "json" in content_type_header.lower():
            detail = body.decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM-Omni returned JSON instead of mp4: {detail}")

        mp4_path = sample_dir / f"{stem}.mp4"
        mp4_path.write_bytes(body)
        remote_outputs = {
            "backend": self.backend,
            "endpoint": "/v1/videos/sync",
            "content_type": content_type_header,
            "bytes": len(body),
        }
        (sample_dir / "sample_outputs.remote.json").write_text(
            json.dumps(remote_outputs, indent=2, ensure_ascii=False) + "\n"
        )
        compat_path = save_compatible_mp4(mp4_path, sample_dir / f"{stem}_compat.mp4")
        npz_path = save_reference_npz(sample, mp4_path, sample_dir / f"{stem}.npz")
        result = {
            "sample": sample.get("name"),
            "backend": self.backend,
            "elapsed_sec": elapsed,
            "downloaded_files": [str(mp4_path)],
            "mp4_path": str(mp4_path),
            "compat_mp4_path": str(compat_path),
            "npz_path": str(npz_path),
        }
        (sample_dir / "sample_outputs.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )
        return CosmosReferenceResult(
            sample=sample,
            sample_dir=sample_dir,
            npz_path=npz_path,
            mp4_path=mp4_path,
            compat_mp4_path=compat_path,
            remote_outputs=remote_outputs,
            elapsed_sec=elapsed,
        )

    def submit(self, sample: Dict[str, Any]) -> CosmosReferenceResult:
        stem = safe_stem(sample.get("name"))
        sample_dir = self.output_root / stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "sample_args.json").write_text(json.dumps(sample, indent=2, ensure_ascii=False) + "\n")
        if self.backend == "vllm_omni":
            return self._submit_vllm_omni(sample, sample_dir=sample_dir, stem=stem)
        request_sample = self._sample_for_server(sample)
        (sample_dir / "sample_args.submitted.json").write_text(
            json.dumps(request_sample, indent=2, ensure_ascii=False) + "\n"
        )
        started = time.time()
        sample_outputs = _request_json(
            f"{self.base_url}/generate",
            method="POST",
            payload=request_sample,
            timeout=self.timeout,
        )
        elapsed = time.time() - started
        (sample_dir / "sample_outputs.remote.json").write_text(
            json.dumps(sample_outputs, indent=2, ensure_ascii=False) + "\n"
        )

        downloaded: List[Path] = []
        mp4_index = 0
        for remote_file in _iter_output_files(sample_outputs):
            rel = Path(remote_file)
            if rel.suffix.lower() == ".mp4":
                suffix = "" if mp4_index == 0 else f"_{mp4_index}"
                local_path = sample_dir / f"{stem}{suffix}.mp4"
                mp4_index += 1
            else:
                local_name = rel.name if rel.name else quote(remote_file, safe="")
                local_path = sample_dir / local_name
            downloaded.append(_download(self.base_url, remote_file, local_path, timeout=self.timeout))

        mp4_candidates = [path for path in downloaded if path.suffix.lower() == ".mp4"]
        if not mp4_candidates:
            raise RuntimeError(f"Cosmos generation produced no mp4 output: {sample_outputs}")
        mp4_path = mp4_candidates[0]
        compat_path = save_compatible_mp4(mp4_path, sample_dir / f"{stem}_compat.mp4")
        npz_path = save_reference_npz(sample, mp4_path, sample_dir / f"{stem}.npz")
        result = {
            "sample": sample.get("name"),
            "elapsed_sec": elapsed,
            "downloaded_files": [str(path) for path in downloaded],
            "mp4_path": str(mp4_path),
            "compat_mp4_path": str(compat_path),
            "npz_path": str(npz_path),
        }
        (sample_dir / "sample_outputs.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        return CosmosReferenceResult(
            sample=sample,
            sample_dir=sample_dir,
            npz_path=npz_path,
            mp4_path=mp4_path,
            compat_mp4_path=compat_path,
            remote_outputs=sample_outputs,
            elapsed_sec=elapsed,
        )
