import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from shutil import copy2
from typing import Any, TypedDict
import httpx

from subtitled_videos_to_asr_dataset.log_config import setup_logger

logger = logging.getLogger(__name__)
setup_logger(logger)

DEFAULT_CLIENT = httpx.Client(headers={"user_agent": "redirect-link-resolver-for-NorGovPCC"})

class DetectedSpokenLanguage(TypedDict):
    language: str
    confidence: float
    model_id: str


class Transcript(TypedDict):
    start: float
    end: float
    text: str
    detected_spoken_language: DetectedSpokenLanguage


class RawDatasetInfo(TypedDict):
    video_id: str
    video_url: str
    subtitle_language: str

    transcript: list[Transcript]


def get_redirect_url(url: str) -> str:
    response = DEFAULT_CLIENT.get(url)
    location = response.headers["location"]
    if location.startswith("http"):
        return location
    elif location.startswith("/"):
        return str(response.url.copy_with(path=location))
    else:
        raise NotImplementedError("Redirect location is not implemented")


def get_transcript(raw_data: dict[str, Any], speaker_language_model) -> list[Transcript]:
    return [
        {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "detected_spoken_language": next(
                detected_spoken_language
                for detected_spoken_language in segment["detected_spoken_language"]
                if detected_spoken_language["model_id"] == speaker_language_model
            ),
        }
        for segment in raw_data["segments"]
    ]


def get_json_data(video_dir: Path, speaker_language_model: str) -> RawDatasetInfo:
    video_id = video_dir.name
    video_url = get_redirect_url(f"https://www.regjeringen.no/{video_id}")
    vtt_files = list(video_dir.glob("*.vtt"))
    if len(vtt_files) > 1:
        raise ValueError(f"More than one vtt file in {video_dir}")

    language = vtt_files[0].stem.split("_")[0]

    with open(video_dir / f"{language}_aligned_with_language.json") as f:
        raw_data = json.load(f)
    transcript = get_transcript(raw_data, speaker_language_model)
    return {
        "video_id": video_id,
        "video_url": video_url,
        "subtitle_language": language,
        "transcript": transcript,
    }


def copy_files(
    from_dir: Path, to_dir: Path, already_copied_vtt_contents: set[str], speaker_language_model: str
) -> set[str]:
    logger.info("Copying files from %s to %s", from_dir, to_dir)
    vtt_files = list(from_dir.glob("*.vtt"))
    if len(vtt_files) != 1:
        raise ValueError(f"There can only be one VTT file in {from_dir}, not {len(vtt_files)}")

    vtt_file = vtt_files[0]

    # Check if we've already handled this VTT file
    vtt_content = vtt_file.read_text()
    files_with_same_content = already_copied_vtt_contents.get(vtt_content, [])
    if files_with_same_content:
        logger.warning(f"VTT file %s already copied, present in %s", vtt_file, files_with_same_content)
        return already_copied_vtt_contents | {vtt_content: files_with_same_content + [from_dir.name]}

    # Find Audio file and load the JSON data
    audio_files = list(from_dir.glob("*.wav"))
    if len(audio_files) != 1:
        raise ValueError(f"There can only be one WAV file in {from_dir}, not {len(audio_files)}")

    audio_file = audio_files[0]
    json_data = get_json_data(from_dir, speaker_language_model)

    # Copy the files
    to_dir.mkdir(parents=True, exist_ok=True)
    copy2(vtt_file, to_dir / vtt_file.name)
    copy2(audio_file, to_dir / "audio.wav")
    with open(to_dir / f"{from_dir.name}.json", "w") as f:
        json.dump(json_data, f)

    return already_copied_vtt_contents | {vtt_content: files_with_same_content + [from_dir.name]}


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset_split_json", type=Path, required=True)
    parser.add_argument("--speaker_language_model", type=str, default="large-v3")
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper())

    already_copied_vtt_contents = {}
    dataset_splits: dict[str, list[str]] = json.loads(args.dataset_split_json.read_text())
    for dataset_split, video_ids in dataset_splits.items():
        logger.info("Processing %s split", dataset_split)
        for video_id in video_ids:
            if not (video_dir := args.input_dir / video_id).is_dir():
                raise ValueError(f"Video directory {video_dir} does not exist")

            already_copied_vtt_contents = copy_files(
                video_dir,
                args.output_dir / dataset_split / video_id,
                already_copied_vtt_contents,
                args.speaker_language_model,
            )
