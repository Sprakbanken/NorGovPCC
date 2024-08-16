import pytest
from subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.compute_transcribed_difference import calculate_subtitle_difference


@pytest.mark.parametrize(
    "subtitle1, subtitle2, expected_difference",
    [
        ("Some subtitles", "", 14),
        ("", "Some subtitles", 14),
        ("Some", "Some subtitles", 10),
        ("SOME", "some", 4),
        ("Some subtitles", "Some subtitles", 0),
    ],
)
def test_known_subtitle_differences(subtitle1, subtitle2, expected_difference):
    """Known subtitles should expected differences."""
    assert calculate_subtitle_difference(subtitle1, subtitle2) == expected_difference
