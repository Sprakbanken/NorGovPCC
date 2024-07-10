import datetime
import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Generator, Iterable, Literal

import numpy as np
import srt
import whisperx

from subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.utils import WHISPER_LANGUAGE_CODES, get_torch_device
from subtitled_videos_to_asr_dataset.log_config import setup_logger
from subtitled_videos_to_asr_dataset.types import Subtitle

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Set log colors
setup_logger(logger)


LanguageCode = Literal["no", "nn"]


def remove_dash(text: str) -> str:
    """Remove dash if present from the start and end of the text."""
    if text.endswith("-"):
        text = text[:-1]
    if text.startswith("-"):
        text = text[1:]
    return text


def process_text(text: str) -> str:
    r"""
    Process the text to remove dash and new line characters.

    >>> process_text("Hello-\nWorld")
    'Hello World'
    >>> process_text("Python-\n-Programming")
    'Python Programming'
    >>> process_text("Hello\nWorld")
    'Hello World'
    """
    text = " ".join(remove_dash(line) for line in text.split("\n"))
    return text


def iter_whisper_x_format(subtitles: Iterable[srt.Subtitle]) -> Generator[Subtitle, None, None]:
    """Iterate over the subtitles in WhisperX format."""
    for subtitle in subtitles:
        yield {
            "start": subtitle.start.total_seconds(),
            "end": subtitle.end.total_seconds(),
            "text": process_text(subtitle.content),
        }


def join_split_subtitles(
    subtitles: Iterable[Subtitle], punctuation_marks: str = ".!?", buffer: float = 0
) -> Generator[Subtitle, None, None]:
    """Combine subtitles until they end with a punctuation mark.

    >>> subtitles = [
    ...     {"start": 0, "end": 1, "text": "Hello"},
    ...     {"start": 1, "end": 2, "text": "world."},
    ...     {"start": 2, "end": 3, "text": "How"},
    ...     {"start": 3, "end": 4, "text": "are you"},
    ...     {"start": 4, "end": 5, "text": "today?"}
    ... ]
    >>> list(join_split_subtitles(subtitles))
    [{'start': 0, 'end': 2, 'text': 'Hello world.', 'num_joined': 2}, {'start': 2, 'end': 5, 'text': 'How are you today?', 'num_joined': 3}]
    """
    subtitle_stack = []
    for subtitle in subtitles:
        subtitle_stack.append(subtitle)

        if subtitle["text"].strip().endswith(tuple(punctuation_marks)):
            yield {
                "start": subtitle_stack[0]["start"] - buffer,
                "end": subtitle_stack[-1]["end"] + buffer,
                "text": " ".join(s["text"].strip() for s in subtitle_stack),
                "num_joined": len(subtitle_stack),
            }
            subtitle_stack = []


def join_subtitles_by_time(
    subtitles: Iterable[Subtitle], target_time: float, max_time: float = 120, skip_too_long_subtitles: bool = True
) -> Generator[Subtitle, None, None]:
    """
    Combine subtitles until they reach the target time without exceeding the maximum time.


    >>> subtitles = [
    ...     {"start": 0.0, "end": 4.6, "text": "Her er en setning i en undertekst."},
    ...     {"start": 5.0, "end": 16.46, "text": "Her kommer en setning til. Og en til."},
    ...     {"start": 16.49, "end": 23.32, "text": "En siste setning kommer her."},
    ... ]
    >>> list(join_subtitles_by_time(subtitles, target_time=10, max_time=20))
    [{'start': 0.0, 'end': 16.46, 'text': 'Her er en setning i en undertekst. Her kommer en setning til. Og en til.', 'num_joined': 2}, {'start': 16.49, 'end': 23.32, 'text': 'En siste setning kommer her.', 'num_joined': 1}]
    >>> list(join_subtitles_by_time(subtitles, target_time=5, max_time=12))
    [{'start': 0.0, 'end': 4.6, 'text': 'Her er en setning i en undertekst.', 'num_joined': 1}, {'start': 5.0, 'end': 16.46, 'text': 'Her kommer en setning til. Og en til.', 'num_joined': 1}, {'start': 16.49, 'end': 23.32, 'text': 'En siste setning kommer her.', 'num_joined': 1}]
    """
    subtitle_stack = []
    max_single_subtitle_time = max_time if skip_too_long_subtitles else float("inf")
    for subtitle in subtitles:
        subtitle_stack.append(subtitle)

        dt = subtitle_stack[-1]["end"] - subtitle_stack[0]["start"]
        if dt > target_time:
            # The current subtitle stack lasts too long: Remove the top entry to be sure that we are below target_time
            if dt > max_time:
                subtitle_stack.pop()

            # If there is something on the stack, then we must yield that first and reset the stack
            if subtitle_stack:
                num_joined = sum(s.get("num_joined", 1) for s in subtitle_stack)
                yield {
                    "start": subtitle_stack[0]["start"],
                    "end": subtitle_stack[-1]["end"],
                    "text": " ".join(s["text"].strip().replace("\n", " ") for s in subtitle_stack),
                    "num_joined": num_joined,
                }
                subtitle_stack = []

            # If the current subtitle is long enough on its own, but not too long then we yield that
            if target_time <= subtitle["end"] - subtitle["start"] < max_single_subtitle_time and dt > max_time:
                yield {**subtitle, "num_joined": subtitle.get("num_joined", 1)}
            # If it’s not long enough on its own, we add it to the stack and start the next iteration
            elif subtitle["end"] - subtitle["start"] < target_time and dt > max_time:
                subtitle_stack.append(subtitle)
            # Else (if the subtitle is too long), we skip it

    # If there are any not yielded subtitles left, we yield them now
    if subtitle_stack:
        num_joined = sum(s.get("num_joined", 1) for s in subtitle_stack)
        yield {
            "start": subtitle_stack[0]["start"],
            "end": subtitle_stack[-1]["end"],
            "text": " ".join(s["text"].strip().replace("\n", " ") for s in subtitle_stack),
            "num_joined": num_joined,
        }


def align_subtitles_with_whisper_x(
    audio: np.ndarray,
    subtitles: Iterable[Subtitle],
    language_code: LanguageCode,
    device: str = "auto",
) -> list[Subtitle]:
    """Align the subtitles with the audio using WhisperX."""
    device = get_torch_device(device)
    model_names = {"no": "NbAiLab/nb-wav2vec2-1b-bokmaal", "nn": "NbAiLab/nb-wav2vec2-1b-nynorsk"}
    model_name = model_names.get(language_code)
    if model_name:
        model, metadata = whisperx.load_align_model(language_code=None, model_name=model_name, device=device)
    else:
        model, metadata = whisperx.load_align_model(language_code=language_code, device=device)

    aligned_subtitles = whisperx.align(list(subtitles), model, metadata, audio, device, return_char_alignments=False)
    return aligned_subtitles


def whisper_x_output_to_srt(subtitles: list[Subtitle]) -> str:
    """Convert the WhisperX output to srt format.

    >>> subtitles = [
    ...     {"start": 0.0, "end": 1.0, "text": "This is a mock subtitle."},
    ...     {"start": 2.0, "end": 3.0, "text": "This is another mock subtitle."},
    ... ]
    >>> print(whisper_x_output_to_srt(subtitles))
    1
    00:00:00,000 --> 00:00:01,000
    This is a mock subtitle.
    <BLANKLINE>
    2
    00:00:02,000 --> 00:00:03,000
    This is another mock subtitle.
    <BLANKLINE>
    <BLANKLINE>"""
    if not all(isinstance(s, dict) and "start" in s and "end" in s and "text" in s for s in subtitles):
        raise ValueError(f"Each subtitle must be a dictionary with keys 'start', 'end', and 'text'. not {subtitles}")

    return srt.compose(
        [
            srt.Subtitle(
                i + 1,
                start=datetime.timedelta(seconds=s["start"]),
                end=datetime.timedelta(seconds=s["end"]),
                content=s["text"],
            )
            for i, s in enumerate(subtitles)
        ]
    )


def whisper_x_output_to_vtt(subtitles: list[Subtitle]) -> str:
    """Convert the WhisperX output to vtt format.

    >>> subtitles = [
    ...     {"start": 0.0, "end": 1.0, "text": "This is a mock subtitle."},
    ...     {"start": 2.0, "end": 3.0, "text": "This is another mock subtitle."},
    ... ]
    >>> print(whisper_x_output_to_vtt(subtitles))
    WEBVTT
    <BLANKLINE>
    1
    00:00:00.000 --> 00:00:01.000
    This is a mock subtitle.
    <BLANKLINE>
    2
    00:00:02.000 --> 00:00:03.000
    This is another mock subtitle.
    <BLANKLINE>
    <BLANKLINE>
    """
    if not all(isinstance(s, dict) and "start" in s and "end" in s and "text" in s for s in subtitles):
        raise ValueError(f"Each subtitle must be a dictionary with keys 'start', 'end', and 'text'. not {subtitles}")

    def format_timestamp(seconds: float) -> str:
        timestamp = datetime.time(0, 0, second=int(seconds), microsecond=int((seconds % 1) * 1_000_000))
        return timestamp.strftime("%H:%M:%S.%f")[:-3]

    start = "WEBVTT\n\n"
    return (
        start
        + "\n\n".join(
            f"{i}\n{format_timestamp(seconds=s['start'])} --> {format_timestamp(seconds=s['end'])}\n{s['text']}"
            for i, s in enumerate(subtitles, start=1)
        )
        + "\n\n"
    )


def align_subtitles(
    directory: Path,
    out_directory: Path | None = None,
    target_time: float = 30,
    max_time: float = 120,
    save_srt: bool = False,
    save_vtt: bool = False,
) -> None:
    """Align the subtitles in the directory with the audio file in the directory."""
    if out_directory is None:
        out_directory = directory

    logger.info("Aligning subtitles in directory %s", directory)
    logger.info("Saving aligned subtitles in directory %s", directory)

    audio_file_path = next(directory.glob("*.wav"))
    audio = whisperx.load_audio(audio_file_path)
    logging.info("Loaded audio file %s", audio_file_path)

    subtitle_file_paths = directory.glob("*.srt")
    for subtitle_file_path in subtitle_file_paths:
        logging.info("Aligning subtitles in file %s", subtitle_file_path)
        language_code = WHISPER_LANGUAGE_CODES.get(subtitle_file_path.stem.split("_")[-1], None)
        if language_code is None:
            logger.warning("Invalid language code for file %s, skipping", subtitle_file_path)
            continue

        aligned_subtitle_file_path = out_directory / f"{subtitle_file_path.stem}_aligned.json"
        if aligned_subtitle_file_path.exists():
            logging.warning("Aligned subtitle file %s already exists. Skipping alignment.", aligned_subtitle_file_path)

            aligned_srt_subtitle_file_path = out_directory / f"{subtitle_file_path.stem}_aligned.srt"
            if save_srt and not aligned_srt_subtitle_file_path.exists():
                with open(aligned_subtitle_file_path, "r") as f:
                    aligned_subtitles = json.load(f)
                aligned_srt_subtitles = whisper_x_output_to_srt(aligned_subtitles["segments"])
                aligned_srt_subtitle_file_path.write_text(aligned_srt_subtitles)
                logging.info("Saved aligned subtitles in srt format in file %s", aligned_srt_subtitle_file_path)

            continue

        srt_subtitles = srt.parse(subtitle_file_path.read_text())

        logging.debug("Processing subtitles in file %s", subtitle_file_path)
        subtitles = join_subtitles_by_time(
            join_split_subtitles(iter_whisper_x_format(srt_subtitles)),
            target_time=target_time,
            max_time=max_time,
        )

        logging.debug("Aligning subtitles in file with WhisperX %s", subtitle_file_path)
        aligned_subtitles = align_subtitles_with_whisper_x(audio, subtitles, language_code=language_code)

        with open(aligned_subtitle_file_path, "w") as f:
            json.dump(aligned_subtitles, f)
        logging.info("Saved aligned subtitles in file %s", aligned_subtitle_file_path)

        if save_srt:
            aligned_srt_subtitle_file_path = out_directory / f"{subtitle_file_path.stem}_aligned.srt"
            aligned_srt_subtitles = whisper_x_output_to_srt(aligned_subtitles["segments"])
            aligned_srt_subtitle_file_path.write_text(aligned_srt_subtitles)
            logging.info("Saved aligned subtitles in srt format in file %s", aligned_srt_subtitle_file_path)

        if save_vtt:
            aligned_vtt_subtitle_file_path = out_directory / f"{subtitle_file_path.stem}_aligned.vtt"
            aligned_vtt_subtitles = whisperx.srt_to_vtt(aligned_srt_subtitles)
            aligned_vtt_subtitle_file_path.write_text(aligned_vtt_subtitles)
            logging.info("Saved aligned subtitles in vtt format in file %s", aligned_vtt_subtitle_file_path)


def align_all_subtitles_in_directories(
    input_directory: Path, output_directory: Path | None = None, save_srt: bool = False, save_vtt: bool = False
) -> None:
    base_output_dir = output_directory
    for directory in input_directory.glob("*/"):
        if base_output_dir:
            output_directory = base_output_dir / directory.name
        else:
            output_directory = None
        align_subtitles(directory, out_directory=output_directory, save_srt=save_srt, save_vtt=save_vtt)


def main():
    """Align all subtitles in the given directory.

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
    ``nb`` for Bokmål (``no`` since that is used to signify Bokmål by Whisper).

    After running the script, the aligned subtitles will be saved in the same directory as the original subtitle files,
    with the same filename as the original subtitle files, but with the suffix ``_aligned.json`` (e.g.
    ``some_subtitle_name_no_aligned.json``).

    The aligned subtitles are saved in the WhisperX format, which is a list of dictionaries with the keys ``start``,
    ``end``, and ``text``. If the --save_srt flag is set, the aligned subtitles will also be saved as srt files.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "input_directory", type=Path, help="The directory containing subdirectories with the audio and subtitle files."
    )
    parser.add_argument(
        "--output_directory", type=Path, help="The directory where the aligned subtitles will be saved."
    )
    parser.add_argument("--log_level", type=str, default="INFO", help="The log level for the logger.")
    parser.add_argument("--save_srt", action="store_true", help="Save the aligned subtitles as srt files as well.")
    parser.add_argument("--save_vtt", action="store_true", help="Save the aligned subtitles as vtt files as well.")

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    align_all_subtitles_in_directories(
        args.input_directory, output_directory=args.output_directory, save_srt=args.save_srt, save_vtt=args.save_vtt
    )
