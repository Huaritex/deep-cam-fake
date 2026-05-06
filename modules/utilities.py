import glob
import json
import mimetypes
import os
import platform
import shutil
import ssl
import subprocess
import urllib
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple
from tqdm import tqdm

import numpy as np

import modules.globals

TEMP_FILE = "temp.mp4"
TEMP_DIRECTORY = "temp"


def run_ffmpeg(args: List[str]) -> bool:
    """Run ffmpeg with hardware acceleration and optimized settings."""
    commands = [
        "ffmpeg",
        "-hide_banner",
        "-hwaccel", "auto",  # Auto-detect hardware acceleration
        "-hwaccel_output_format", "auto",  # Use hardware format when possible
        "-threads", str(modules.globals.execution_threads or 0),  # 0 = auto-detect optimal thread count
        "-loglevel", modules.globals.log_level,
    ]
    commands.extend(args)
    try:
        subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True
    except Exception:
        pass
    return False


def detect_fps(target_path: str) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        target_path,
    ]
    output = subprocess.check_output(command).decode().strip().split("/")
    try:
        numerator, denominator = map(int, output)
        return numerator / denominator
    except Exception:
        pass
    return 30.0


def extract_frames(target_path: str) -> None:
    """Extract frames with hardware acceleration and optimized settings."""
    temp_directory_path = get_temp_directory_path(target_path)
    
    # Use hardware-accelerated decoding and optimized pixel format
    run_ffmpeg(
        [
            "-i", target_path,
            "-vf", "format=rgb24",  # Use video filter for format conversion (faster)
            "-vsync", "0",  # Prevent frame duplication
            "-frame_pts", "1",  # Preserve frame timing
            os.path.join(temp_directory_path, "%04d.png"),
        ]
    )


def _build_encoder_args(
    encoder: str, quality: int, providers: List[str]
) -> Tuple[str, List[str]]:
    """Return (encoder_name, extra_ffmpeg_args) for the given provider/encoder."""
    if "CUDAExecutionProvider" in providers:
        if encoder == "libx264":
            return "h264_nvenc", [
                "-preset", "p7", "-tune", "hq", "-rc", "vbr",
                "-cq", str(quality), "-b:v", "0", "-multipass", "fullres",
            ]
        if encoder == "libx265":
            return "hevc_nvenc", [
                "-preset", "p7", "-tune", "hq", "-rc", "vbr",
                "-cq", str(quality), "-b:v", "0",
            ]
    if "DmlExecutionProvider" in providers:
        if encoder == "libx264":
            return "h264_amf", [
                "-quality", "quality", "-rc", "vbr_latency",
                "-qp_i", str(quality), "-qp_p", str(quality),
            ]
        if encoder == "libx265":
            return "hevc_amf", [
                "-quality", "quality", "-rc", "vbr_latency",
                "-qp_i", str(quality), "-qp_p", str(quality),
            ]
    # CPU
    if encoder == "libx265":
        return encoder, ["-preset", "medium", "-crf", str(quality), "-x265-params", "log-level=error"]
    if encoder == "libvpx-vp9":
        return encoder, ["-crf", str(quality), "-b:v", "0", "-cpu-used", "2"]
    return encoder, ["-preset", "medium", "-crf", str(quality), "-tune", "film"]


_COMMON_OUTPUT_ARGS = ["-pix_fmt", "yuv420p", "-movflags", "+faststart",
                       "-vf", "colorspace=bt709:iall=bt601-6-625:fast=1"]


def create_video(target_path: str, fps: float = 30.0) -> None:
    """Create video from temp PNG frames with hardware-accelerated encoding."""
    temp_output_path = get_temp_output_path(target_path)
    temp_directory_path = get_temp_directory_path(target_path)

    enc_name, enc_opts = _build_encoder_args(
        modules.globals.video_encoder,
        modules.globals.video_quality,
        modules.globals.execution_providers,
    )

    ffmpeg_args = [
        "-r", str(fps),
        "-i", os.path.join(temp_directory_path, "%04d.png"),
        "-c:v", enc_name,
        *enc_opts,
        *_COMMON_OUTPUT_ARGS,
        "-y", temp_output_path,
    ]

    success = run_ffmpeg(ffmpeg_args)

    if not success and enc_name in ("h264_nvenc", "hevc_nvenc", "h264_amf", "hevc_amf"):
        print(f"Hardware encoding with {enc_name} failed, falling back to software...")
        fallback = "libx264" if "h264" in enc_name else "libx265"
        fb_name, fb_opts = _build_encoder_args(fallback, modules.globals.video_quality, [])
        run_ffmpeg([
            "-r", str(fps),
            "-i", os.path.join(temp_directory_path, "%04d.png"),
            "-c:v", fb_name,
            *fb_opts,
            *_COMMON_OUTPUT_ARGS,
            "-y", temp_output_path,
        ])


# ---------------------------------------------------------------------------
# Streaming pipeline — zero temp-file I/O for frame data
# ---------------------------------------------------------------------------

def get_video_info(video_path: str) -> dict:
    """Return first video-stream metadata from ffprobe (empty dict on failure)."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "v:0",
        video_path,
    ]
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        streams = json.loads(raw).get("streams", [])
        return streams[0] if streams else {}
    except Exception:
        return {}


def stream_frames_from_video(video_path: str) -> Generator[np.ndarray, None, None]:
    """Yield BGR uint8 numpy frames from *video_path* via an ffmpeg stdout pipe.

    Bypasses all temp-file I/O.  Falls back gracefully on probe failure.
    """
    info = get_video_info(video_path)
    width = int(info.get("width", 0))
    height = int(info.get("height", 0))
    if width <= 0 or height <= 0:
        return

    cmd = [
        "ffmpeg", "-hide_banner",
        "-hwaccel", "auto",
        "-i", video_path,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-vsync", "0",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = width * height * 3
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
    finally:
        proc.stdout.close()
        proc.wait()


class VideoFrameEncoder:
    """Context manager: encode BGR numpy frames to a video file via ffmpeg stdin pipe.

    Usage::
        with VideoFrameEncoder(path, fps, w, h) as enc:
            for frame in frames:
                enc.write(frame)
    """

    def __init__(self, output_path: str, fps: float, width: int, height: int) -> None:
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self._proc: Optional[subprocess.Popen] = None

    def __enter__(self) -> "VideoFrameEncoder":
        enc_name, enc_opts = _build_encoder_args(
            modules.globals.video_encoder,
            modules.globals.video_quality,
            modules.globals.execution_providers,
        )
        cmd = [
            "ffmpeg", "-hide_banner", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-r", str(self.fps),
            "-s", f"{self.width}x{self.height}",
            "-i", "pipe:0",
            "-c:v", enc_name,
            *enc_opts,
            *_COMMON_OUTPUT_ARGS,
            self.output_path,
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return self

    def write(self, frame: np.ndarray) -> None:
        if self._proc and self._proc.stdin:
            self._proc.stdin.write(frame.tobytes())

    def __exit__(self, *_: object) -> None:
        if self._proc:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait()


def restore_audio(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    done = run_ffmpeg(
        [
            "-i",
            temp_output_path,
            "-i",
            target_path,
            "-c:v",
            "copy",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-y",
            output_path,
        ]
    )
    if not done:
        move_temp(target_path, output_path)


def get_temp_frame_paths(target_path: str) -> List[str]:
    temp_directory_path = get_temp_directory_path(target_path)
    return glob.glob((os.path.join(glob.escape(temp_directory_path), "*.png")))


def get_temp_directory_path(target_path: str) -> str:
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    target_directory_path = os.path.dirname(target_path)
    return os.path.join(target_directory_path, TEMP_DIRECTORY, target_name)


def get_temp_output_path(target_path: str) -> str:
    temp_directory_path = get_temp_directory_path(target_path)
    return os.path.join(temp_directory_path, TEMP_FILE)


def normalize_output_path(source_path: str, target_path: str, output_path: str) -> Any:
    if source_path and target_path:
        source_name, _ = os.path.splitext(os.path.basename(source_path))
        target_name, target_extension = os.path.splitext(os.path.basename(target_path))
        if os.path.isdir(output_path):
            return os.path.join(
                output_path, source_name + "-" + target_name + target_extension
            )
    return output_path


def create_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)


def move_temp(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    if os.path.isfile(temp_output_path):
        if os.path.isfile(output_path):
            os.remove(output_path)
        shutil.move(temp_output_path, output_path)


def clean_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    parent_directory_path = os.path.dirname(temp_directory_path)
    if not modules.globals.keep_frames and os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path)
    if os.path.exists(parent_directory_path) and not os.listdir(parent_directory_path):
        os.rmdir(parent_directory_path)


def has_image_extension(image_path: str) -> bool:
    return image_path.lower().endswith(("png", "jpg", "jpeg"))


def is_image(image_path: str) -> bool:
    if image_path and os.path.isfile(image_path):
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith("image/"))
    return False


def is_video(video_path: str) -> bool:
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith("video/"))
    return False


def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)
    for url in urls:
        download_file_path = os.path.join(
            download_directory_path, os.path.basename(url)
        )
        if not os.path.exists(download_file_path):
            request = urllib.request.Request(url)
            
            # Create a specific SSL context for macOS to avoid globally disabling verification
            ctx = None
            if platform.system().lower() == "darwin":
                ctx = ssl._create_unverified_context()
                
            response = urllib.request.urlopen(request, context=ctx)
            total = int(response.headers.get("Content-Length", 0))
            with tqdm(
                total=total,
                desc="Downloading",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress:
                with open(download_file_path, "wb") as f:
                    while True:
                        buffer = response.read(8192)
                        if not buffer:
                            break
                        f.write(buffer)
                        progress.update(len(buffer))


def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))
