import logging
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from subprocess import PIPE, Popen

from subtitled_videos_to_asr_dataset.log_config import setup_logger

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Set log colors
setup_logger(logger)


def convert_wav_to_mp3(input_file: str, output_file: str, overwrite: bool) -> None:
    """
    Converts a WAV file to an MP3 file with a sample rate of 16000Hz using FFMPEG.
    """
    if output_file.exists() and not overwrite:
        logger.info(f"File {output_file} already exists. Skipping conversion.")
        return
    # -i: input file
    # -ar: audio sample rate
    # -b:a: audio bitrate
    ffmpeg_command = f"ffmpeg -y -i {input_file} -ar 16000 -b:a 128k {output_file}"

    logger.info("Converting %s to %s", input_file, output_file)
    logger.debug("Running command: %s", ffmpeg_command)
    subprocess.run(ffmpeg_command, shell=True)


def compress_audio_files(directory: Path, overwrite: bool) -> None:
    """Compress all wav files in the directory and save them as MP3 files."""
    logger.info("Compressing audio files in %s", directory)

    for audio_file in directory.glob("*.wav"):
        output_file = directory / audio_file.with_suffix(".mp3").name
        convert_wav_to_mp3(audio_file, output_file, overwrite=overwrite)
        logger.info(f"Compressed {audio_file} to {output_file}")


def compress_audio_files_in_directories(input_directory: Path, overwrite: bool) -> None:
    """Recursively compress all wav files in the input directory and save them as MP3 files."""
    logger.info("Compressing audio files in %s", input_directory)

    for directory in input_directory.iterdir():
        if not directory.is_dir():
            continue

        compress_audio_files(directory, overwrite)


def main():
    """Create a compressed mp3 file for each audio file.

    Assumes the following directory structure
    ```
    input_directory
    ├── directory1
    │   ├── some_audio_name.wav
    └── directory2
        ├── some_audio_name.wav


    The output will be a file ``{some_audio_name}_16kHz.mp3`` in each directory.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "input_directory", type=Path, help="The directory containing subdirectorys with the audio files."
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.", default=False)
    parser.add_argument("--log_level", type=str, default="INFO", help="The log level for the logger.")

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    compress_audio_files_in_directories(args.input_directory, overwrite=args.overwrite)
