import json
import logging
import shutil
import subprocess
from argparse import ArgumentParser
from pathlib import Path

from vtt_to_srt.vtt_to_srt import VttToStr as BaseVttToSrt

from subtitled_videos_to_asr_dataset.log_config import setup_logger

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Set log colors
setup_logger(logger)


class VttToStr(BaseVttToSrt):
    def convert_header(self, contents: str) -> str:
        """Converts the header of a VTT file to SRT format."""
        return contents.partition("\n\n")[2]


def mp4_to_wav(source_directory: Path, target_directory: Path, overwrite: bool) -> None:
    """Creates .wav-files from the audio of .mp4-files in source_directory."""
    for e in source_directory.iterdir():
        if e.suffix == ".mp4":
            audio_path = target_directory / f"{e.name[:-4]}.wav"
            if audio_path.exists() and not overwrite:
                logger.debug(f"Audio file {audio_path} already exists, skipping")
                continue
            shell_str = f'ffmpeg -y -i "{e}" -ac 2 -f wav "{audio_path}"'
            logger.debug(f"Converting {e.name} to .wav")
            subprocess.call(shell_str, shell=True)


def mp4_to_wav_nested(source_directory: Path, target_directory: Path, relevant_ids: list[str], overwrite: bool) -> None:
    """Creates a .wav-file for every .mp4-file in the relevant subdirectories of source_directory.

    Subdirectory structure of source_directory is copied onto target_directory.
    Will overwrite existing .wav-files if overwrite is True.
    """
    for subdir in source_directory.iterdir():
        if subdir.name in relevant_ids:
            audio_subdir = target_directory / subdir.name
            if audio_subdir.exists() and overwrite:
                logger.debug(f"Overwriting {audio_subdir}")
                shutil.rmtree(audio_subdir)

            audio_subdir.mkdir(exist_ok=True, parents=True)
            mp4_to_wav(subdir, audio_subdir, overwrite)


def copy_subtitle_files(target_directory: Path, subtitle_files: dict[str, list[Path]], overwrite: bool) -> None:
    """Copies the specified subtitle files from video directory to audio directory.

    Assumes that source_directory and target_directory contain subdirectories with names corresponing to keys of ids_subs.
    Will skip existing subtitle files in target_directory if overwrite is False.
    """
    for id, subs in subtitle_files.items():
        target_p = target_directory / id
        copied_subs: dict[str, Path] = {}  # Mapping the content of a subtitle file to the file itself

        for subtitle_file in subs:
            subtitle_content = subtitle_file.read_text()
            if subtitle_content in copied_subs:
                logger.debug(f"Skipping {subtitle_file} as it is a duplicate of {copied_subs[subtitle_content]}")
                continue
            copied_subs[subtitle_content] = subtitle_file

            subtitle_file_target = target_p / subtitle_file.name
            if subtitle_file_target.exists() and not overwrite:
                logger.debug(f"Subtitle file {subtitle_file_target} already exists, skipping")
                continue
            logger.debug(f"Copying {subtitle_file} to {subtitle_file_target}")
            shutil.copy(subtitle_file, subtitle_file_target)

            # Convert VTT to SRT
            VttToStr().process(str(subtitle_file_target), "utf-8")


def get_relevant_subtitle_files(source_directory: Path, languages: list[Path]) -> dict[str, list[Path]]:
    """Get the subdirectory ids and subtitle filenames if subs are available in specified languages

    Returns a dictionary where the keys are the relevant subdirectory names and values are lists of pairs of (language code, subtitle filename)
    """
    ids_subs = {
        subdir.name: [subdir / f"{language}.vtt" for language in languages if (subdir / f"{language}.vtt").exists()]
        for subdir in source_directory.iterdir()
    }

    return {key: value for key, value in ids_subs.items() if value}


def main():
    parser = ArgumentParser()
    parser.add_argument("--source_dir", "-sd", type=str, help="Path to directory with scraped raw data", required=True)
    parser.add_argument(
        "--output_dir",
        "-od",
        type=str,
        help="Path to directory where sound and subtitles will be saved (is created if it doesn't already exists)",
        required=True,
    )

    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        dest="overwrite",
        default=True,
        help="Only fill in missing sound and subtitle data to output_dir (if True, overwrite any existing data)",
    )

    parser.add_argument(
        "--languages",
        "-l",
        nargs="+",
        default=["nn", "nb", "no"],
        help="List of languages (two-character ISO-639-codes) to keep. Only videos with subtitles in these languages will be converted to sound",
    )
    args = parser.parse_args()

    languages = args.languages
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    overwrite = args.overwrite

    if not source_dir.exists():
        logger.error(f"Source directory {source_dir} does not exist")
        exit()

    output_dir.mkdir(exist_ok=True, parents=True)

    relevant_subtitle_files = get_relevant_subtitle_files(source_directory=source_dir, languages=languages)
    mp4_to_wav_nested(
        source_directory=source_dir,
        target_directory=output_dir,
        relevant_ids=relevant_subtitle_files.keys(),
        overwrite=overwrite,
    )
    copy_subtitle_files(target_directory=output_dir, subtitle_files=relevant_subtitle_files, overwrite=overwrite)

    logger.info("Finished converting video to audio and copying subtitle files")
