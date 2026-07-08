"""Gradio web app for LuxTTS.

Run with:
    python -m zipvoice.web_app
"""

from __future__ import annotations

import argparse
import tempfile
from typing import Any

DEFAULT_MODEL_ID = "YatharthS/LuxTTS"
DEFAULT_SAMPLE_RATE = 48_000

_lux_tts: Any | None = None
_loaded_config: tuple[str, str, int] | None = None


def _require_gradio():
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "The LuxTTS web app requires Gradio. Install it with "
            "`pip install gradio` or `pip install -r requirements.txt`."
        ) from exc

    return gr


def _ui_error(message: str) -> Exception:
    gr = _require_gradio()
    return gr.Error(message)


def _normalize_model_path(model_path: str | None) -> str:
    model_path = (model_path or "").strip()
    return model_path or DEFAULT_MODEL_ID


def get_tts(model_path: str, device: str, threads: int) -> Any:
    """Load and cache a LuxTTS instance for the selected runtime options."""
    global _lux_tts, _loaded_config

    from zipvoice.luxvoice import LuxTTS

    normalized_model_path = _normalize_model_path(model_path)
    normalized_device = (device or "cuda").strip().lower()
    normalized_threads = max(1, int(threads))
    config = (normalized_model_path, normalized_device, normalized_threads)

    if _lux_tts is None or _loaded_config != config:
        _lux_tts = LuxTTS(
            model_path=normalized_model_path,
            device=normalized_device,
            threads=normalized_threads,
        )
        _loaded_config = config

    return _lux_tts


def _audio_path_from_gradio(prompt_audio: Any) -> str:
    """Return a filesystem path from Gradio's filepath audio input."""
    if prompt_audio is None:
        raise _ui_error("Please upload or record a reference audio clip first.")

    if isinstance(prompt_audio, str):
        return prompt_audio

    if isinstance(prompt_audio, dict) and prompt_audio.get("path"):
        return str(prompt_audio["path"])

    if isinstance(prompt_audio, (tuple, list)) and len(prompt_audio) == 2:
        sample_rate, data = prompt_audio
        try:
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover - only used for non-file audio modes
            raise _ui_error("Install soundfile to use raw audio input tuples.") from exc

        suffix = ".wav"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.close()
        sf.write(tmp.name, data, int(sample_rate))
        return tmp.name

    raise _ui_error("Unsupported audio input format. Please upload a WAV, MP3, FLAC, or M4A file.")


def _tensor_to_gradio_audio(wav: Any) -> tuple[int, Any]:
    """Convert the model waveform tensor to a Gradio audio output tuple."""
    import numpy as np

    audio = wav.detach().cpu().float().numpy()
    audio = np.squeeze(audio)

    if audio.ndim == 0:
        raise _ui_error("The model returned an empty audio tensor.")

    if audio.ndim > 1:
        # Gradio expects channels last for numpy audio. LuxTTS normally returns mono.
        if audio.shape[0] in (1, 2):
            audio = np.moveaxis(audio, 0, -1)
        else:
            audio = audio.reshape(-1)

    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    return DEFAULT_SAMPLE_RATE, audio


def synthesize(
    text: str,
    prompt_audio: Any,
    model_path: str,
    device: str,
    threads: int,
    reference_duration: float,
    rms: float,
    num_steps: int,
    guidance_scale: float,
    t_shift: float,
    speed: float,
    return_smooth: bool,
) -> tuple[int, Any]:
    """Generate speech from the web UI inputs."""
    text = (text or "").strip()
    if not text:
        raise _ui_error("Please enter text to synthesize.")

    prompt_path = _audio_path_from_gradio(prompt_audio)
    tts = get_tts(model_path, device, threads)
    encoded_prompt = tts.encode_prompt(prompt_path, duration=reference_duration, rms=rms)
    wav = tts.generate_speech(
        text,
        encoded_prompt,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        t_shift=t_shift,
        speed=speed,
        return_smooth=return_smooth,
    )
    return _tensor_to_gradio_audio(wav)


def build_demo():
    """Build the LuxTTS Gradio interface."""
    gr = _require_gradio()

    css = """
    .lux-header {text-align: center; margin-bottom: 1rem;}
    .lux-header h1 {font-size: 2.4rem; margin-bottom: 0.25rem;}
    .lux-header p {font-size: 1.05rem; color: var(--body-text-color-subdued);}
    """

    with gr.Blocks(title="LuxTTS Studio", theme=gr.themes.Soft(), css=css) as demo:
        gr.HTML(
            """
            <div class="lux-header">
              <h1>LuxTTS Studio</h1>
              <p>Clean local voice-cloning text-to-speech interface for LuxTTS.</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=5):
                text = gr.Textbox(
                    label="Text to synthesize",
                    placeholder="Type the sentence you want LuxTTS to speak...",
                    lines=5,
                )
                prompt_audio = gr.Audio(
                    label="Reference voice audio",
                    sources=["upload", "microphone"],
                    type="filepath",
                )
                generate_button = gr.Button("Generate speech", variant="primary")
            with gr.Column(scale=4):
                output_audio = gr.Audio(label="Generated speech", type="numpy", autoplay=False)
                gr.Markdown(
                    """
                    **Tips**
                    - Use at least 3 seconds of clean reference speech.
                    - Increase steps for quality; decrease steps for speed.
                    - Try **Smooth output** if the result sounds metallic.
                    """
                )

        with gr.Accordion("Model and sampling settings", open=False):
            with gr.Row():
                model_path = gr.Textbox(
                    label="Model path or Hugging Face ID",
                    value=DEFAULT_MODEL_ID,
                )
                device = gr.Dropdown(
                    label="Device",
                    choices=["cuda", "mps", "cpu"],
                    value="cuda",
                )
                threads = gr.Slider(label="CPU threads", minimum=1, maximum=16, value=4, step=1)
            with gr.Row():
                reference_duration = gr.Slider(
                    label="Reference duration (seconds)",
                    minimum=1,
                    maximum=30,
                    value=5,
                    step=0.5,
                )
                rms = gr.Slider(label="Reference RMS", minimum=0.001, maximum=0.05, value=0.01, step=0.001)
                num_steps = gr.Slider(label="Sampling steps", minimum=1, maximum=16, value=4, step=1)
            with gr.Row():
                guidance_scale = gr.Slider(label="Guidance scale", minimum=0.1, maximum=8.0, value=3.0, step=0.1)
                t_shift = gr.Slider(label="T-shift", minimum=0.1, maximum=1.5, value=0.9, step=0.05)
                speed = gr.Slider(label="Speed", minimum=0.5, maximum=1.5, value=1.0, step=0.05)
                return_smooth = gr.Checkbox(label="Smooth output", value=False)

        generate_button.click(
            fn=synthesize,
            inputs=[
                text,
                prompt_audio,
                model_path,
                device,
                threads,
                reference_duration,
                rms,
                num_steps,
                guidance_scale,
                t_shift,
                speed,
                return_smooth,
            ],
            outputs=output_audio,
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LuxTTS Gradio web app.")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind.")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind.")
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio share URL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_demo()
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
