import logging
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.align_subtitles
import subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.compress_audio
import subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.compute_transcribed_difference
import subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.language_identification
import subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.transcribe_audio


def main():
    parser = ArgumentParser()
    parser.add_argument("input_directory", help="The directory containing the subtitle files", type=Path)
    parser.add_argument("--log_level", default="INFO", help="The log level")
    parser.add_argument("--save_srt", action="store_true", help="Save the aligned subtitles as srt files as well.")

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    try:
        subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.transcribe_audio.transcribe_using_all_models(
            args.input_directory,
            infer_language_from_filename=False,
        )

        subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.compute_transcribed_difference.compute_all_transcribed_differences_in_directories(
            args.input_directory,
            partial(
                subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.compute_transcribed_difference.get_transcription_file,
                model="nb-whisper-small",
                include_language=False,
            ),
        )
        subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.compute_transcribed_difference.compute_all_transcribed_differences_in_directories(
            args.input_directory,
            partial(
                subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.compute_transcribed_difference.get_transcription_file,
                model="nb-whisper-medium",
                include_language=False,
            ),
        )
        subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.compute_transcribed_difference.compute_all_transcribed_differences_in_directories(
            args.input_directory,
            partial(
                subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.compute_transcribed_difference.get_transcription_file,
                model="nb-whisper-large",
                include_language=False,
            ),
        )

        subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.align_subtitles.align_all_subtitles_in_directories(
            args.input_directory, args.save_srt
        )

        subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.compress_audio.compress_audio_files_in_directories(
            args.input_directory, overwrite=False
        )

        DEFAULT_WHISPER_MODEL = (
            subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.language_identification.DEFAULT_WHISPER_MODEL
        )
        subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.language_identification.whisper_detect_languages(
            DEFAULT_WHISPER_MODEL, args.input_directory
        )
        DEFAULT_HUGGINGFACE_MODEL = subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.language_identification.DEFAULT_HUGGINGFACE_MODEL
        subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.language_identification.huggingface_detect_languages(
            DEFAULT_HUGGINGFACE_MODEL, args.input_directory
        )
    except Exception:
        logging.exception("An error occurred while processing the audio and subtitles")
        raise
