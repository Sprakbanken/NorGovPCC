import logging
from argparse import ArgumentParser
from pathlib import Path

from subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.language_identification.huggingface_identification import (
    DEFAULT_HUGGINGFACE_MODEL,
)
from subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.language_identification.huggingface_identification import (
    detect_languages_all_audio_in_directories as huggingface_detect_languages,
)
from subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.language_identification.whisper_identification import (
    DEFAULT_WHISPER_MODEL,
)
from subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.language_identification.whisper_identification import (
    detect_languages_all_audio_in_directories as whisper_detect_languages,
)
from subtitled_videos_to_asr_dataset.log_config import setup_logger

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Set log colors
setup_logger(logger)


def main():
    """Use the start and end indices from the aligned subtitles to detect the language of the speaker in the source audio for each subtitle.

    Assumes the following directory structure
    ```
    input_directory
    ├── directory1
    │   ├── some_audio_name.mp3
    │   └── some_subtitle_name_aligned.json
    └── directory2
        ├── some_audio_name.mp3
        └── some_subtitle_name_aligned.json
    ```

    We assume exactly one audio file (stored as .mp3) and at least one aligned subtitle .json file in each directory.
    The detected languages are written to the subtitle file(s).
    """
    parser = ArgumentParser()
    parser.add_argument(
        "input_directory", type=Path, help="The directory containing subdirectories with the audio and subtitle files."
    )
    parser.add_argument("--log_level", type=str, default="INFO", help="The logging level.")

    args = parser.parse_args()
    logger.setLevel(getattr(logging, args.log_level.upper()))

    input_dir = Path(args.input_directory)
    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        exit()

    huggingface_detect_languages(DEFAULT_HUGGINGFACE_MODEL, input_dir)
    whisper_detect_languages(DEFAULT_WHISPER_MODEL, input_dir)
