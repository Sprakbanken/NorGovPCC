import json
import logging
from pathlib import Path

import numpy as np
import resampy
import soundfile as sf

from subtitled_videos_to_asr_dataset.log_config import setup_logger
from subtitled_videos_to_asr_dataset.types import AlignedSubtitles

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Set log colors
setup_logger(logger)


def wav_to_audio_array(
    audio_path: str, start_sec: float, end_sec: float, to_sampling_rate: int, to_mono: bool = True
) -> np.array:
    """Returns array representation of the sound between start_sec and end_sec in the wav_file.
    Downsamples and turns sound into mono audio if needed.
    """
    info = sf.info(audio_path)
    sampling_rate = info.samplerate

    start_i = int(sampling_rate * start_sec)
    end_i = int(sampling_rate * end_sec)

    audio, sampling_rate = sf.read(audio_path, start=start_i, stop=end_i)

    if to_mono and len(audio.shape) == 2 and audio.shape[1] > 1:
        audio = np.mean(audio, axis=1)

    if not sampling_rate >= to_sampling_rate:
        logger.error(f"Sampling rate of {audio_path} is smaller than language identification model")
        return np.array([])

    if sampling_rate != to_sampling_rate:
        audio = resampy.resample(audio, sampling_rate, to_sampling_rate)

    return audio


def get_subtitles_and_output_file(subtitle_file: Path) -> tuple[AlignedSubtitles, str]:
    out_file = subtitle_file.parent / f"{subtitle_file.stem}_with_language.json"

    if out_file.exists():
        logger.debug(f"out_file exists, reading subs from {out_file}")
        with out_file.open("rb") as f:
            subs = json.load(f)
    else:
        logger.debug(f"out_file does not exist, reading subs from {subtitle_file}")
        with subtitle_file.open("rb") as f:
            subs = json.load(f)

    return subs, out_file
