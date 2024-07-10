"""Identifies the language in the audio files"""

import json
import logging
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

from subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.language_identification.utils import (
    get_subtitles_and_output_file,
    wav_to_audio_array,
)
from subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.utils import get_torch_device
from subtitled_videos_to_asr_dataset.log_config import setup_logger
from subtitled_videos_to_asr_dataset.types import Subtitle, SubtitleWithLanguage

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Set log colors
setup_logger(logger)

DEFAULT_HUGGINGFACE_MODEL = "facebook/mms-lid-256"


def detect_language(
    segments: list[Subtitle],
    audio_file: str,
    processor: AutoFeatureExtractor,
    device: Literal["cpu", "cuda", "auto"],
    model: Wav2Vec2ForSequenceClassification,
    model_id: str,
    desc: str,
) -> list[SubtitleWithLanguage]:
    segments = deepcopy(segments)
    for sub in tqdm(segments, desc=desc):
        detected_language = sub.get("detected_spoken_language", [])
        already_detected = [e for e in detected_language if e["model_id"] == model_id]
        if already_detected:
            continue

        audio_array = wav_to_audio_array(
            audio_file,
            start_sec=sub["start"],
            end_sec=sub["end"],
            to_sampling_rate=processor.sampling_rate,
            to_mono=True,
        )

        inputs = processor(audio_array, sampling_rate=processor.sampling_rate, return_tensors="pt").to(device)
        with torch.no_grad():
            try:
                logits = model(**inputs).logits
            except Exception:
                logger.exception("Failed at language detection, input type: %s", type(inputs))
                logger.info(f"audio array shape: {audio_array.shape}")
                continue

        certainties = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        label2id = {v: k for k, v in model.config.id2label.items()}
        certainties[label2id["nob"]] += certainties[label2id["nno"]]

        lang_id = np.argmax(certainties)
        certainty = certainties[lang_id]
        detected_language.append(
            {
                "model_id": model_id,
                "detected_language": model.config.id2label[lang_id],
                "model_certainty": float(certainty),
            }
        )
        sub["detected_spoken_language"] = detected_language

    return segments


def detect_languages_all_audio_in_directories(model_id: str, directory: Path) -> None:
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
    device = get_torch_device("auto")
    processor = AutoFeatureExtractor.from_pretrained(model_id, device_map="auto")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id).to(device)

    for subdir in directory.iterdir():
        subtitle_files = list(subdir.glob("*_aligned.json"))
        audio_files = list(subdir.glob("*.mp3"))
        if len(audio_files) > 1:
            logger.warn(f"More than one audio file in {subdir}. Skipping subdirectory")
            continue

        if not subtitle_files:
            logger.warn(f"No aligned subtitle files in {subdir}. Skipping subdirectory")
            continue

        audio_file = audio_files[0]
        logger.debug(f"Detecting spoken language in {audio_file}")
        for subtitle_file in subtitle_files:
            subs, out_file = get_subtitles_and_output_file(subtitle_file)

            logger.debug(f"Processing sound clips using subtitle indices from {subtitle_file}")
            subs["segments"] = detect_language(
                subs["segments"],
                audio_file,
                processor,
                device,
                model,
                model_id,
                desc=f"{subdir.name}_{subtitle_file.name.split('_')[1]}",
            )

            logger.debug(f"Writing detected language to {out_file}")
            with out_file.open("w+") as f:
                json.dump(subs, f, ensure_ascii=False, indent=4)

    logger.info(f"Finished detecting languages in {directory}")


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

    detect_languages_all_audio_in_directories(DEFAULT_HUGGINGFACE_MODEL, input_dir)


if __name__ == "__main__":
    main()
