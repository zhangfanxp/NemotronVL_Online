"""Local Gradio UI for image and sampled-video analysis with NVIDIA Nemotron VL."""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Iterator

import cv2
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
MAX_IMAGE_BYTES = 30 * 1024 * 1024
MAX_VIDEO_BYTES = 500 * 1024 * 1024
MAX_FRAME_SIDE = 1280


def get_client() -> OpenAI:
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key or api_key == "请在这里填入你的 NVIDIA API Key":
        raise ValueError("请先在项目根目录的 .env 文件中填写 NVIDIA_API_KEY。")

    return OpenAI(
        base_url=os.getenv("NVIDIA_BASE_URL", DEFAULT_BASE_URL).strip(),
        api_key=api_key,
        timeout=300.0,
        max_retries=1,
    )


def image_as_data_url(image_path: str) -> str:
    path = Path(image_path)
    if not path.is_file():
        raise ValueError("没有找到图片文件，请重新上传。")
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError("图片超过 30 MB，请压缩后再试。")

    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def extract_video_frames(video_path: str, requested_frames: int) -> list[tuple[float, str]]:
    path = Path(video_path)
    if not path.is_file():
        raise ValueError("没有找到视频文件，请重新上传。")
    if path.stat().st_size > MAX_VIDEO_BYTES:
        raise ValueError("视频超过 500 MB。")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("无法读取该视频。请尝试使用 MP4 或 MOV 文件。")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_total <= 0:
            raise ValueError("无法获取视频时长或帧数，请转换为常见的 MP4/MOV 格式后重试。")

        sample_count = min(max(1, int(requested_frames)), frame_total)
        if sample_count == 1:
            frame_indexes = [0]
        else:
            frame_indexes = [
                round(index * (frame_total - 1) / (sample_count - 1))
                for index in range(sample_count)
            ]

        frames: list[tuple[float, str]] = []
        for frame_index in frame_indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue

            height, width = frame.shape[:2]
            longest_side = max(width, height)
            if longest_side > MAX_FRAME_SIDE:
                scale = MAX_FRAME_SIDE / longest_side
                frame = cv2.resize(
                    frame,
                    (round(width * scale), round(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            ok, jpeg = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88]
            )
            if not ok:
                continue

            timestamp = frame_index / fps if fps > 0 else 0.0
            encoded = base64.b64encode(jpeg.tobytes()).decode("ascii")
            frames.append((timestamp, f"data:image/jpeg;base64,{encoded}"))

        if not frames:
            raise ValueError("视频中没有读取到可用画面。")
        return frames
    finally:
        capture.release()


def stream_completion(
    prompt: str,
    content: list[dict],
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
) -> Iterator[str]:
    if not prompt.strip():
        yield "请先输入你想了解的问题。"
        return

    content.insert(0, {"type": "text", "text": prompt.strip()})
    try:
        client = get_client()
        response = client.chat.completions.create(
            model=os.getenv("NVIDIA_MODEL", DEFAULT_MODEL).strip(),
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            top_p=0.95,
            max_tokens=int(max_tokens),
            extra_body={
                "reasoning_effort": "high" if enable_thinking else "none"
            },
            stream=True,
        )

        answer = ""
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                answer += delta.content
                yield answer

        if not answer:
            yield "模型没有返回文本。请稍后重试或调整问题。"
    except Exception as exc:  # Show actionable API/configuration errors in the UI.
        message = str(exc)
        secret = os.getenv("NVIDIA_API_KEY", "").strip()
        if secret:
            message = message.replace(secret, "***")
        yield f"调用失败：{message[:2000]}"


def stream_with_button_lock(
    answers: Iterator[str],
) -> Iterator[tuple[str, dict]]:
    latest_answer = "正在等待模型响应…"
    for answer in answers:
        latest_answer = answer
        yield latest_answer, gr.update(interactive=False)
    yield latest_answer, gr.update(interactive=True)


def analyze_image(
    image_path: str | None,
    prompt: str,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
) -> Iterator[tuple[str, dict]]:
    if not image_path:
        yield "请先上传一张图片。", gr.update(interactive=True)
        return

    yield "正在处理图片并请求模型…", gr.update(interactive=False)
    try:
        data_url = image_as_data_url(image_path)
        content = [
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    except Exception as exc:
        yield f"图片处理失败：{exc}", gr.update(interactive=True)
        return

    yield from stream_with_button_lock(
        stream_completion(prompt, content, temperature, max_tokens, enable_thinking)
    )


def analyze_video(
    video_path: str | None,
    prompt: str,
    requested_frames: int,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
) -> Iterator[tuple[str, dict]]:
    if not video_path:
        yield "请先上传一个视频。", gr.update(interactive=True)
        return

    yield "正在读取视频并抽取画面…", gr.update(interactive=False)
    try:
        frames = extract_video_frames(video_path, requested_frames)
        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    f"以下是按时间均匀抽取的视频画面，共 {len(frames)} 帧；"
                    "方括号中包含每帧在原视频中的时间。请结合这些画面回答问题。"
                ),
            }
        ]
        for index, (timestamp, data_url) in enumerate(frames, start=1):
            content.append(
                {
                    "type": "text",
                    "text": f"[视频帧 {index}/{len(frames)}，时间 {timestamp:.2f} 秒]",
                }
            )
            content.append({"type": "image_url", "image_url": {"url": data_url}})
    except Exception as exc:
        yield f"视频处理失败：{exc}", gr.update(interactive=True)
        return

    yield from stream_with_button_lock(
        stream_completion(prompt, content, temperature, max_tokens, enable_thinking)
    )


with gr.Blocks(title="Nemotron VL 图片与视频分析") as demo:
    gr.Markdown(
        "# Nemotron VL 图片与视频分析\n"
        "使用 NVIDIA Nemotron 3.5 Super VL 在线接口，在本机浏览器中分析图片和视频。\n\n"
        "**隐私提示：** 图片以及视频抽取出的画面会发送到 NVIDIA API；视频文件本身不会作为完整文件上传。"
    )

    with gr.Row():
        temperature = gr.Slider(
            minimum=0,
            maximum=1.2,
            value=0.6,
            step=0.1,
            label="温度（视觉分析建议 0.6）",
        )
        max_tokens = gr.Slider(
            minimum=512,
            maximum=8192,
            value=4096,
            step=512,
            label="最大生成长度",
        )
        enable_thinking = gr.Checkbox(value=True, label="启用模型思考模式")

    with gr.Tab("图片分析"):
        with gr.Row():
            image_input = gr.Image(
                label="上传图片",
                type="filepath",
                sources=["upload"],
            )
            image_prompt = gr.Textbox(
                label="你想了解什么？",
                value="请详细描述这张图片中的内容。",
                lines=4,
            )
        image_button = gr.Button("分析图片", variant="primary")
        image_output = gr.Markdown(label="分析结果")
        image_button.click(
            fn=analyze_image,
            inputs=[image_input, image_prompt, temperature, max_tokens, enable_thinking],
            outputs=[image_output, image_button],
            trigger_mode="once",
        )

    with gr.Tab("视频分析"):
        video_input = gr.Video(
            label="上传视频（MP4 / MOV 等）",
            sources=["upload"],
        )
        video_prompt = gr.Textbox(
            label="你想了解什么？",
            value="请概括视频内容，并按时间顺序说明关键事件。",
            lines=4,
        )
        requested_frames = gr.Slider(
            minimum=2,
            maximum=24,
            value=8,
            step=1,
            label="均匀抽取的画面数（越多细节越多，也会增加请求大小）",
        )
        video_button = gr.Button("分析视频", variant="primary")
        video_output = gr.Markdown(label="分析结果")
        video_button.click(
            fn=analyze_video,
            inputs=[
                video_input,
                video_prompt,
                requested_frames,
                temperature,
                max_tokens,
                enable_thinking,
            ],
            outputs=[video_output, video_button],
            trigger_mode="once",
        )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("APP_PORT", "7860")),
        inbrowser=True,
        max_file_size="500mb",
    )
