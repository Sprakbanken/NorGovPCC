import json
import logging
import uuid
from argparse import ArgumentParser
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import soundfile as sf
from datasets import load_dataset
from subtitled_videos_to_asr_dataset.log_config import setup_logger
from subtitled_videos_to_asr_dataset.types import Subtitle, SubtitleWithLanguage
from jinja2 import Template

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Set log colors
setup_logger(logger)


@dataclass(init=False)
class DatasetInfo:
    num_segments: int
    duration: float
    nno_count: int
    nno_count: int
    name: str

    @property
    def nno_fraction(self):
        return self.nno_count / self.num_segments

    @property
    def nob_fraction(self):
        return self.nob_count / self.num_segments

    def __init__(self, metadata: pd.DataFrame, name: str):
        """
        Initialize the CreateHuggingfaceDataset class.

        Parameters
        ----------
        metadata : pd.DataFrame
            A DataFrame with 'duration' and 'transcription_language' columns.

        Attributes
        ----------
        num_segments : int
            The number of segments in the dataset.
        duration : float
            The total duration of the dataset.
        nno_count : int
            The count of Norwegian Nynorsk segments.
        nob_count : int
            The count of Norwegian Bokmål segment.
        """
        self.num_segments = len(metadata)
        self.duration = metadata["duration"].sum()
        self.nno_count = metadata["transcription_language"].value_counts().get("nn", 0)
        self.nob_count = metadata["transcription_language"].value_counts().get("nb", 0)
        self.name = name


@dataclass(init=False)
class FullDatasetInfo(DatasetInfo):
    split_info: dict[str, DatasetInfo]

    def __init__(self, metadata: pd.DataFrame):
        super().__init__(metadata, name="Full data")

        split_names = metadata["file_name"].str.extract("data/(.*?)/", expand=False).fillna(self.name)

        self.split_info = {
            split_name: DatasetInfo(split_metadata, split_name)
            for split_name, split_metadata in metadata.groupby(split_names)
        }
        assert (
            self.name not in self.split_info or len(self.split_info) == 1
        ), "Full data should only be present if no splits are present"

        # Sort splits by names in reverse alphabetical order to get training split before test split
        self.split_info = {k: self.split_info[k] for k in sorted(self.split_info.keys(), reverse=True)}


def merge_subtitles_upto_n_sec(indexed_subtitles: list[tuple[int, Subtitle]], n: int = 30) -> list[Subtitle]:
    merged_subs = []
    prev_i = 0

    for i, sub in indexed_subtitles:
        if (
            merged_subs == [] or prev_i != i - 1
        ):  # current sub is not directly following the previous in original subtitle file
            merged_subs.append(sub)
            prev_i = i
            continue

        prev_sub = merged_subs[-1]
        merged_duration = sub["end"] - prev_sub["start"]
        if merged_duration <= n:
            prev_sub["end"] = sub["end"]
            prev_sub["text"] += " " + sub["text"]
        else:
            merged_subs.append(sub)
        prev_i = i

    return merged_subs


def remove_unwanted_subtitles(
    segments: list[SubtitleWithLanguage],
    preferred_langdetect_model: str = "large-v3",
    preferred_langcode: str = "no",
    max_segment_length: int = 30,
) -> list[tuple[int, SubtitleWithLanguage]]:
    segments_to_keep = []
    for i, seg in enumerate(segments):
        detected_lang = next(
            (det for det in seg["detected_spoken_language"] if det["model_id"] == preferred_langdetect_model), None
        )

        if not detected_lang:
            logger.warning(
                f"Could not find detected language with preferred language identification model for segment {i}"
            )
            continue

        if max_segment_length and seg["end"] - seg["start"] > max_segment_length:
            logger.debug(
                f"Segment {i} has length {seg['end'] - seg['start']}, which is longer than max segment length {max_segment_length}"
            )
            continue

        if detected_lang["detected_language"] != preferred_langcode:
            logger.debug(f"Detected language for segment {i}: {detected_lang['detected_language']}")
            continue

        segments_to_keep.append((i, seg))

    return segments_to_keep


def segments_to_audio_files_and_metadata(
    segments: list[SubtitleWithLanguage], audio_file: Path, audio_id: str, target_directory: Path
) -> dict[str, list]:
    metadata = defaultdict(list)
    samplerate = sf.info(audio_file).samplerate
    for i, seg in enumerate(segments):
        start_i = int(samplerate * seg["start"])
        end_i = int(samplerate * seg["end"])

        audio, samplerate = sf.read(audio_file, start=start_i, stop=end_i)

        target_audio_filename = target_directory / f"{audio_id}_{i}.mp3"
        sf.write(target_audio_filename, audio, samplerate)

        relative_filename = f"{target_audio_filename.parent.name}/{target_audio_filename.name}"

        metadata["file_name"].append(relative_filename)
        metadata["transcription"].append(seg["text"])
        metadata["duration"].append(seg["end"] - seg["start"])
    return metadata


def build_datasplit_directory_and_metadata(
    audio_directories: Iterable[Path],
    target_directory: Path,
    preferred_langdetect_model: str = "large-v3",
    preferred_langdetect_langcode: str = "no",
    merge_upto_n_seconds: int = 30,
    remove_segments_longer_than_n: bool = True,
) -> dict[str, list]:
    """Build a directory of short mp3-files from subtitle segments of larger mp3 files.

    Will filter out audio segments that are not on the correct language, and merge neighbouring segments upto n seconds and,
    if specified, remove too long segments.

    Assumes that each directory in audio_directories contain one .mp3-file and at least one subtitle file ending with 'aligned_with_language.json'

    Will build the following directory structure:
    ```
    target_directory
    ├── directory-name1_0.mp3
    ├── directory-name1_1.mp3
    ├── directory-name2_0.mp3
    ├── directory-name2_1.mp3
    ├── directory-name2_2.mp3
    └── directory-name2_3.mp3
    ```

    And return a dict containing metadata for each .mp3 file in target directory
    """

    metadata = defaultdict(list)

    for e in audio_directories:
        aligned_files = list(e.glob("*aligned_with_language.json"))
        if not aligned_files:
            logger.debug(f"Directory {e} has no aligned subtitle files, skipping")
            continue

        mp3_file = next(e.glob("*.mp3"))
        audio_id = e.name

        for e2 in aligned_files:
            subtitle_lang_code = e2.name.split("_")[-4]
            with e2.open() as f:
                logger.debug(f"Reading subtitle segments from {e2}")
                segments = json.load(f)["segments"]

            indexed_segments = remove_unwanted_subtitles(
                segments=segments,
                preferred_langdetect_model=preferred_langdetect_model,
                preferred_langcode=preferred_langdetect_langcode,
                max_segment_length=remove_segments_longer_than_n and merge_upto_n_seconds,
            )
            if merge_upto_n_seconds:
                segments = merge_subtitles_upto_n_sec(indexed_subtitles=indexed_segments, n=merge_upto_n_seconds)
            else:
                segments = zip(*indexed_segments)

            segments_metadata = segments_to_audio_files_and_metadata(
                segments=segments, audio_file=mp3_file, audio_id=audio_id, target_directory=target_directory
            )

            segments_metadata["transcription_language"] = [subtitle_lang_code] * len(segments)
            segments_metadata["detected_spoken_language"] = [preferred_langdetect_langcode] * len(segments)
            segments_metadata["language_identification_model"] = [preferred_langdetect_model] * len(segments)

            model_certainties = [
                next(
                    det["model_certainty"]
                    for det in seg["detected_spoken_language"]
                    if det["model_id"] == preferred_langdetect_model
                )
                for seg in segments
            ]
            segments_metadata["language_identification_model_certainty"] = model_certainties
            segments_metadata["source_audio_id"] = [audio_id] * len(segments)

            for k in segments_metadata:
                metadata[k] += segments_metadata[k]

    return metadata


def rename_and_shuffle(dataset_directory: Path):
    metadata_file = dataset_directory / "metadata.csv"
    df = pd.read_csv(metadata_file)
    new_df_filenames = []
    for e in df.file_name:
        p = dataset_directory / e
        new_name = f"{uuid.uuid4()}.mp3"
        p = p.rename(p.with_name(new_name))
        new_df_filenames.append(p.relative_to(dataset_directory))

    df["file_name"] = new_df_filenames
    df = df.sort_values("file_name")
    df = df.drop(columns=["source_audio_id"])
    df.to_csv(dataset_directory / "metadata.csv", index=False)


def push_to_hub(dataset_path: str, huggingface_dataset_name: str, token: str, private: bool):
    dataset = load_dataset(dataset_path)
    dataset.push_to_hub(huggingface_dataset_name, token=token, private=private)


def main():
    """Build a 🤗️ audio dataset and (optionally) upload it to the huggingface hub."""

    parser = ArgumentParser()
    parser.add_argument(
        "--input_directory",
        type=Path,
        help="The directory containing audio files and aligned and subtitles with language detection",
        required=True,
    )
    parser.add_argument(
        "--output_directory",
        type=Path,
        help="The directory where the huggingface dataset is saved.",
        default="hf_dataset/",
    )
    parser.add_argument(
        "--dataset_split_json",
        type=Path,
        default=None,
        help="A json-file with split name keys and list of subdirectory values",
    )
    parser.add_argument(
        "--langdetect_model",
        default="large-v3",
        help="Which language detection model results to use to filter language",
    )
    parser.add_argument(
        "--langdetect_langcode",
        default="no",
        help="Which detected language code to filter segments by (will only keep segments with this language code)",
    )
    parser.add_argument(
        "--merge_n_seconds",
        type=int,
        default=30,
        help="Merge shorter neighbouring segments up to merge_n_seconds seconds if possible (does not merge segments if merge_n_seconds is 0)",
    )
    parser.add_argument(
        "--remove_longer_segments",
        action="store_true",
        default=False,
        help="If flagged: throw away segments longer than n seconds (from merge_n_seconds)",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push dataset to huggingface hub if true",
        default=False,
    )

    parser.add_argument(
        "--hf_repo_id",
        help="Dataset repo name on huggingface hub\nFormat: user_id/dataset_name",
        required=False,
    )

    parser.add_argument(
        "--hf_private", action="store_true", help="Makes dataset private on huggingface hub if flagged", required=False
    )

    parser.add_argument(
        "--hf_token",
        help="Huggingface token with write access",
        required=False,
    )

    parser.add_argument("--log_level", default="INFO", help="The log level")

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    if not args.input_directory.exists():
        logger.critical(f"Input directory {args.input_directory} does not exist")
        exit()

    if args.push:
        empty_hf_args = ["--" + k for k, v in vars(args).items() if k.startswith("hf") and v is None]
        if empty_hf_args:
            req_list = " and ".join(empty_hf_args)
            logger.critical(f"--push requires {req_list}")
            exit()

    # create dataset directory
    data_dir = args.output_directory / "data"
    data_dir.mkdir(exist_ok=True, parents=True)

    if args.dataset_split_json:
        dataset_splits: dict[str, list[str]] = json.loads(args.dataset_split_json.read_text())
        metadata = defaultdict(list)
        for split_name, directory_ids in dataset_splits.items():
            directory_id_paths = [args.input_directory / e for e in directory_ids]
            target_directory = data_dir / split_name
            target_directory.mkdir(exist_ok=True)
            split_metadata = build_datasplit_directory_and_metadata(
                target_directory=target_directory,
                audio_directories=directory_id_paths,
                preferred_langdetect_model=args.langdetect_model,
                preferred_langdetect_langcode=args.langdetect_langcode,
                merge_upto_n_seconds=args.merge_n_seconds,
                remove_segments_longer_than_n=args.remove_longer_segments,
            )

            for k in split_metadata:
                metadata[k].extend(split_metadata[k])
        metadata["file_name"] = ["data/" + e for e in metadata["file_name"]]
    else:
        metadata = build_datasplit_directory_and_metadata(
            target_directory=data_dir,
            audio_directories=args.input_directory.iterdir(),
            preferred_langdetect_model=args.langdetect_model,
            preferred_langdetect_langcode=args.langdetect_langcode,
            merge_upto_n_seconds=args.merge_n_seconds,
            remove_segments_longer_than_n=args.remove_longer_segments,
        )

    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(args.output_directory / "metadata.csv", index=False)

    # Create data card markdown file
    dataset_info = FullDatasetInfo(metadata_df)
    template_file = Path(__file__).parent / "data_card.md.j2"
    template = Template(template_file.read_text())
    data_card = template.render(dataset_info=dataset_info)
    data_card_file = args.output_directory / "README.md"
    data_card_file.write_text(data_card)

    logger.info("Finished creating audio files and metadata.csv for huggingface dataset")

    rename_and_shuffle(args.output_directory)

    # check that dataset loads
    dataset = load_dataset(str(args.output_directory))

    logger.info(f"Successfully built 🤗️ dataset in {args.output_directory}")

    if args.push:
        dataset.push_to_hub(args.hf_repo_id, token=args.hf_token, private=args.hf_private)
        logger.info(f"Pushed dataset to https://huggingface.co/datasets/{args.hf_repo_id}")
