"""Creates a file "whisperx_transcribed_{language_code}.srt in each of the data directories"""

import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal

import whisperx

from subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.utils import WHISPER_LANGUAGE_CODES, get_torch_device
from subtitled_videos_to_asr_dataset.log_config import setup_logger

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Set log colors
setup_logger(logger)


def transcribe_audio(
    directory: Path,
    out_directory: Path | None = None,
    batch_size: int = 1,
    device: str = "auto",
    compute_type: str = "float16",
    model_name: str = "large-v3",
    infer_language: bool = False,
) -> None:
    """Transcribe the audio file in the directory using whisperX."""
    device = get_torch_device(device)

    if out_directory is None:
        out_directory = directory

    audio_file_path = next(directory.glob("*.wav"))
    audio = whisperx.load_audio(audio_file_path)

    subtitle_file_paths = directory.glob("*.srt")
    model = whisperx.load_model(model_name, device, compute_type=compute_type)

    for subtitle_file_path in subtitle_file_paths:
        out_name = f"whisperx_transcribed_{model_name.split('/')[-1]}"

        if not infer_language:
            language_code = WHISPER_LANGUAGE_CODES[subtitle_file_path.stem.split("_")[-1]]
            out_name += f"_{language_code}"

        transcribed_subtitles_file_path = out_directory / f"{out_name}.json"
        if transcribed_subtitles_file_path.exists():
            logger.info(
                "Subtitles transcribed with WhisperX model %s already exist for %s, skipping",
                model_name,
                audio_file_path,
            )
            continue

        transcribed_subtitles = model.transcribe(audio, batch_size=batch_size, language=language_code)
        with open(transcribed_subtitles_file_path, "w") as f:
            json.dump(transcribed_subtitles, f)

        if infer_language:
            break


def transcribe_all_audio_in_directories(
    input_directory: Path, output_directory: Path | None, model_name: str = "large-v3", infer_language: bool = False
) -> None:
    base_output_directory = output_directory
    for directory in input_directory.glob("*/"):
        logger.info("Transcribing audio in %s", directory)

        if base_output_directory:
            output_directory = base_output_directory / directory.name
        else:
            output_directory = None
        transcribe_audio(
            directory, out_directory=output_directory, model_name=model_name, infer_language=infer_language
        )


def transcribe_using_all_models(
    input_directory: Path,
    output_directory: Path | None,
):
    for model_name in ["NbAiLab/nb-whisper-large", "NbAiLab/nb-whisper-medium", "NbAiLab/nb-whisper-small"]:
        logger.info(
            "Transcribing audio using the %s model (transcribing to the same language as the subtitle file)",
            model_name,
        )
        transcribe_all_audio_in_directories(
            input_directory, output_directory, model_name=model_name, infer_language=False
        )


def main():
    """Transcribe the audio file in all subdirectories of the given directory using WhisperX.

    Assumes the following directory structure
    ```
    input_directory
    ├── directory1
    │   ├── some_audio_name.wav
    │   ├── some_subtitle_name_nn.srt  (optional)
    │   └── some_subtitle_name_no.srt  (optional)
    └── directory2
        ├── some_audio_name.wav
        ├── some_subtitle_name_nn.srt  (optional)
        └── some_subtitle_name_no.srt  (optional)
    ```

    We assume exactly one audio file (stored as .wav) in each directory. There can be one or more subtitle files in
    each directory, and they need to have filename ending with the language code - ``nn`` for Nynorsk and ``no`` or
    ``nb`` for Bokmål (``no`` since that is used to signify Bokmål by Whisper). After running the script, the
    transcribed subtitles will be saved in the same directory as the audio file with the filename
    ``whisperx_transcribed_{language_code}.json``. If there are multiple srt-files in the directory, then multiple
    transcribed files will be saved, one for each language.

    This script assumes that there is only one subtitle file for each language in each directory. If there are multiple
    subtitle files for the same language, then only one automatic transcription will be saved.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "input_directory", type=Path, help="The directory containing subdirectorys with the audio and subtitle files."
    )
    parser.add_argument(
        "--output_directory", type=Path, help="The directory where the aligned subtitles will be saved."
    )
    parser.add_argument("--log_level", type=str, default="INFO", help="The logging level.")

    args = parser.parse_args()
    logger.setLevel(getattr(logging, args.log_level.upper()))
    transcribe_using_all_models(args.input_directory, args.output_directory)
