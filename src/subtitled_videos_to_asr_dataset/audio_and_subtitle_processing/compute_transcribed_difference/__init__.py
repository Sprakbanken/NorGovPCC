import json
import logging
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Callable

import Levenshtein
import srt

from subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.utils import WHISPER_LANGUAGE_CODES
from subtitled_videos_to_asr_dataset.log_config import setup_logger

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Set log colors
setup_logger(logger)


def calculate_subtitle_difference(subtitle1: str, subtitle2: str) -> int:
    """Calculate the Levenshtein distance between two subtitles."""
    return Levenshtein.distance(subtitle1, subtitle2)


def concatenate_srt_subtitle(subtitles: list[srt.Subtitle]) -> str:
    """Concatenate the text of the subtitles."""
    return " ".join(subtitle.content.replace("\n", " ") for subtitle in subtitles)


def concatenate_whisper_subtitle(subtitles: list[dict[str, str]]) -> str:
    """Concatenate the text of the subtitles."""
    return " ".join(subtitle["text"].replace("\n", " ") for subtitle in subtitles)


def get_transcription_file(
    subtitle_file_path: Path, out_directory: Path, model: str = "nb-whisper-large", include_language=True
) -> Path:
    """Get the path to the transcribed file."""
    if not include_language:
        return out_directory / f"whisperx_transcribed_{model}.json"
    language_code = WHISPER_LANGUAGE_CODES[subtitle_file_path.stem.split("_")[-1]]
    return out_directory / f"whisperx_transcribed_{model}_{language_code}.json"


def compute_transcribed_difference(
    directory: Path,
    get_auto_transcription: Callable[[Path], Path],
    out_directory: Path | None = None,
) -> None:
    """Compute the edit distance between the two transcripts and store in a file "autotranscript_error_nb-whisper-large_{language_code}.json" """
    if out_directory is None:
        out_directory = directory
    logger.info("Computing the difference in %s", directory)

    subtitle_file_paths = directory.glob("*.srt")
    for subtitle_file_path in subtitle_file_paths:
        logger.info(f"Processing %s", subtitle_file_path)

        language_code = WHISPER_LANGUAGE_CODES[subtitle_file_path.stem.split("_")[-1]]
        auto_transcribed_file_path = get_auto_transcription(subtitle_file_path, out_directory)

        # Parse and concatenate the subtitles
        with open(auto_transcribed_file_path, "r") as f:
            auto_transcribed_subtitles = json.load(f)["segments"]

        subtitles = srt.parse(subtitle_file_path.read_text())
        concatenated_subtitles = concatenate_srt_subtitle(subtitles)
        concatenated_auto_transcribed_subtitles = concatenate_whisper_subtitle(auto_transcribed_subtitles)

        # Calculate the difference between the two transcripts
        distance = calculate_subtitle_difference(concatenated_subtitles, concatenated_auto_transcribed_subtitles)

        # Store the error in a file
        error_file_path = out_directory / f"autotranscript_error_{language_code}.json"
        logger.info(f"Saving error file to %s, error is %d", error_file_path, distance)
        if error_file_path.exists():
            errors = json.loads(error_file_path.read_text())
        else:
            errors = []

        errors.append(
            {
                "autotranscript_file": auto_transcribed_file_path.name,
                "levenshtein_distance": distance,
                "subtitle_length": len(concatenated_subtitles),
                "autotranscribed_length": len(concatenated_auto_transcribed_subtitles),
                "relative_error_rate": 2
                * distance
                / (len(concatenated_subtitles) + len(concatenated_auto_transcribed_subtitles)),
            }
        )

        with open(error_file_path, "w") as f:
            json.dump(errors, f)


def compute_all_transcribed_differences_in_directories(
    input_directory: Path, get_auto_transcription: Callable[[Path], Path], output_directory: Path | None
) -> None:
    base_output_dir = output_directory
    for directory in input_directory.glob("*/"):
        if base_output_dir:
            output_directory = base_output_dir / directory.name
        else:
            output_directory = None

        compute_transcribed_difference(
            directory, get_auto_transcription=get_auto_transcription, out_directory=output_directory
        )


def main():
    """Compute the Levenshtein distance between the auto-transcribed subtitles and the original subtitles.

    Assumes the following directory structure
    ```
    input_directory
    ├── directory1
    │   ├── some_audio_name.wav
    │   ├── some_subtitle_name_nn.srt  (optional)
    │   ├── some_subtitle_name_no.srt  (optional)
    |   ├── whisperx_transcribed_nn.json  (optional)
    |   └── whisperx_transcribed_no.json  (optional)
    └── directory2
        ├── some_audio_name.wav
        ├── some_subtitle_name_nn.srt  (optional)
        ├── some_subtitle_name_no.srt  (optional)
        ├── whisperx_transcribed_nn.json  (optional)
        └── whisperx_transcribed_no.json  (optional)

    We assume that there is only one srt file for each language code in each directory, and that for each srt-file,
    there is an accompanying ``whisperx_transcribed_{language_code}.json`` file. There can be one or more subtitle files
    in each directory.

    The output will be a file ``autotranscript_error_{language_code}.json`` for each language code in each directory.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "input_directory", type=Path, help="The directory containing subdirectorys with the audio and subtitle files."
    )
    parser.add_argument(
        "--output_directory", type=Path, help="The directory where the aligned subtitles will be saved."
    )
    parser.add_argument(
        "--include_language_in_filenames",
        action="store_true",
        help="Include the language code in the subtitle filenames.",
    )
    parser.add_argument("--log_level", type=str, default="INFO", help="The log level for the logger.")

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    compute_all_transcribed_differences_in_directories(
        args.input_directory,
        partial(get_transcription_file, model="nb-whisper-small", include_language=args.include_language_in_filenames),
        args.output_directory,
    )
    compute_all_transcribed_differences_in_directories(
        args.input_directory,
        partial(get_transcription_file, model="nb-whisper-medium", include_language=args.include_language_in_filenames),
        args.output_directory,
    )
    compute_all_transcribed_differences_in_directories(
        args.input_directory,
        partial(get_transcription_file, model="nb-whisper-large", include_language=args.include_language_in_filenames),
        args.output_directory,
    )
