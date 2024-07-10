import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Generator

import numpy as np
import whisperx
from whisperx.asr import FasterWhisperPipeline

from subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.language_identification.utils import (
    get_subtitles_and_output_file,
)
from subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.utils import get_torch_device
from subtitled_videos_to_asr_dataset.log_config import setup_logger
from subtitled_videos_to_asr_dataset.types import (
    AlignedSubtitles,
    Language,
    Probability,
    Subtitle,
    SubtitleWithLanguage,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Set log colors
setup_logger(logger)

DEFAULT_WHISPER_MODEL = "large-v3"


def detect_language(model: FasterWhisperPipeline, audio: np.ndarray) -> tuple[Language, dict[Language, Probability]]:
    # Adapted from whisperx.asr.FasterWhisperPipeline.detected_language
    from whisperx.asr import N_SAMPLES, log_mel_spectrogram

    if audio.shape[0] < N_SAMPLES:
        print("Warning: audio is shorter than 30s, language detection may be inaccurate.")
    model_n_mels = model.model.feat_kwargs.get("feature_size")
    segment = log_mel_spectrogram(
        audio[:N_SAMPLES],
        n_mels=model_n_mels if model_n_mels is not None else 80,
        padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0],
    )
    encoder_output = model.model.encode(segment)
    results = model.model.model.detect_language(encoder_output)
    language_probabilities = dict((l[2:-2], p) for l, p in results[0])
    language_probabilities["no"] += language_probabilities.pop("nn")

    language = max(language_probabilities, key=language_probabilities.__getitem__)

    return language, language_probabilities


def load_audio_and_transcription(audio_path: Path) -> tuple[np.ndarray, AlignedSubtitles]:
    audio = whisperx.load_audio(audio_path, 16000)
    aligned_subtitle_path = next(audio_path.parent.glob("*_aligned.json"))
    with aligned_subtitle_path.open("rb") as f:
        subtitles = json.load(f)
    return audio, subtitles


def iter_subtitles(
    audio: np.ndarray,
    segments: list[Subtitle],
    sample_rate: int = 16000,
    window_size: int = 30,
    use_segment_end: bool = False,
) -> Generator[tuple[np.ndarray, Subtitle], None, None]:
    for segment in segments:
        start = segment["start"]
        start_index = int(start * sample_rate)

        if use_segment_end:
            end = segment["end"]
        else:
            end = start + window_size
        end_index = int(end * sample_rate)
        yield audio[start_index:end_index], segment


def identify_whisper(
    audio: np.ndarray, segments: list[Subtitle], model: FasterWhisperPipeline, model_name: str
) -> Generator[SubtitleWithLanguage, None, None]:
    for audio_window, segment in iter_subtitles(audio, segments):
        segment_with_language = segment.copy()

        detected_languages = segment_with_language.get("detected_spoken_language", [])

        # Only label if it isn't already labeled with this model
        already_detected = False
        for detected_language in detected_languages:
            if detected_language["model_id"] == model_name:
                already_detected = True
        if already_detected:
            continue

        # Detect language and add to detected_languages list
        language, language_probabilities = detect_language(model, audio_window)
        detected_languages.append(
            {"model_id": model_name, "detected_language": language, "model_certainty": language_probabilities[language]}
        )
        segment_with_language["detected_spoken_language"] = detected_languages
        yield segment_with_language


def add_language_to_subtitles(subtitles: AlignedSubtitles, audio: np.ndarray, model_name: str) -> AlignedSubtitles:
    device = get_torch_device("auto")
    model = whisperx.load_model(model_name, device)

    new_subtitles = subtitles.copy()
    new_subtitles["segments"] = list(identify_whisper(audio, new_subtitles["segments"], model, model_name))

    return new_subtitles


def detect_languages_all_audio_in_directories(model_name: str, directory: Path) -> None:
    for subdirectory in directory.glob("*/"):
        subtitle_files = list(subdirectory.glob("*_aligned.json"))
        audio_files = list(subdirectory.glob("*.mp3"))
        if len(audio_files) > 1:
            logger.warn(f"More than one audio file in {subdirectory}. Skipping subdirectory")
            continue

        if not subtitle_files:
            logger.warn(f"No aligned subtitle files in {subdirectory}. Skipping subdirectory")
            continue

        audio_path = audio_files[0]

        logger.debug(f"Detecting spoken language in {audio_path}")
        for subtitle_file in subtitle_files:
            subtitles, output_file = get_subtitles_and_output_file(subtitle_file)
            audio, subtitles = load_audio_and_transcription(audio_path)

            new_subtitles = add_language_to_subtitles(subtitles, audio, model_name)
            aligned_subtitle_path = next(subdirectory.glob("*_aligned.json"), None)
            if aligned_subtitle_path is None:
                logger.info("No aligned subtitle file found in %s", subdirectory)
                continue

            with output_file.open("w") as f:
                json.dump(new_subtitles, f)
            logger.info("Updated aligned subtitle file in %s", subdirectory)


def main():
    parser = ArgumentParser()
    parser.add_argument("data_directory", type=Path)
    parser.add_argument("--model_name", type=str, default=DEFAULT_WHISPER_MODEL)
    args = parser.parse_args()
    detect_languages_all_audio_in_directories(args.model_name, args.data_directory)


if __name__ == "__main__":
    main()
