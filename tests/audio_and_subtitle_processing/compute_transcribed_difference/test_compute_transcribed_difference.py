import json

import pytest
import srt
from subtitled_videos_to_asr_dataset.audio_and_subtitle_processing.compute_transcribed_difference import compute_transcribed_difference


@pytest.fixture
def example_srt_text_content():
    return """\
1
00:21:36,865 --> 00:21:39,231
Hello, this is a subtitle\n
over two lines.
2
00:21:39,234 --> 00:21:41,987
This is another subtitle\n
that also spans two lines."""


@pytest.fixture
def example_srt_content(example_srt_text_content):
    return srt.parse(example_srt_text_content)


@pytest.fixture
def example_whisperx_content():
    return [
        {"start": 21.36865, "end": 21.39231, "text": "Hello, this is a subtitle over two lines."},
        {"start": 21.39234, "end": 21.41987, "text": "This is another subtitle that also spans two lines."},
    ]


def test_no_error(tmp_path, example_srt_content, example_whisperx_content):
    """No error should be found when the subtitles are the same."""
    srt_file = tmp_path / "subtitle_no.srt"
    srt_file.write_text(srt.compose(example_srt_content), "utf-8")
    json_file = tmp_path / "whisperx_transcribed_no.json"
    with open(json_file, "w") as f:
        json.dump({"segments": example_whisperx_content}, f)
    compute_transcribed_difference(tmp_path, (lambda x, y: json_file))

    error_file = tmp_path / "autotranscript_error_no.json"
    assert error_file.exists()

    with open(error_file, "r") as error_file:
        distance = json.load(error_file)[0]["levenshtein_distance"]
    assert distance == 0


def test_one_insert(tmp_path, example_srt_content, example_whisperx_content):
    """The distance should be 1 when one character is inserted."""
    srt_file = tmp_path / "subtitle_no.srt"
    srt_file.write_text(srt.compose(example_srt_content), "utf-8")
    json_file = tmp_path / "whisperx_transcribed_no.json"
    example_whisperx_content_insert = example_whisperx_content.copy()
    example_whisperx_content_insert[0]["text"] = example_whisperx_content_insert[0]["text"] + "H"

    with open(json_file, "w") as f:
        json.dump({"segments": example_whisperx_content_insert}, f)
    compute_transcribed_difference(tmp_path, (lambda x, y: json_file))

    error_file = tmp_path / "autotranscript_error_no.json"
    assert error_file.exists()

    with open(error_file, "r") as error_file:
        distance = json.load(error_file)[0]["levenshtein_distance"]
    assert distance == 1


def test_one_delete(tmp_path, example_srt_content, example_whisperx_content):
    """The distance should be 1 when one character is deleted."""
    srt_file = tmp_path / "subtitle_no.srt"
    srt_file.write_text(srt.compose(example_srt_content))
    json_file = tmp_path / "whisperx_transcribed_no.json"
    example_whisperx_content_delete = example_whisperx_content.copy()
    example_whisperx_content_delete[0]["text"] = example_whisperx_content[0]["text"][1:]

    with open(json_file, "w") as f:
        json.dump({"segments": example_whisperx_content_delete}, f)
    compute_transcribed_difference(tmp_path, (lambda x, y: json_file))

    error_file = tmp_path / "autotranscript_error_no.json"
    assert error_file.exists()

    with open(error_file, "r") as error_file:
        distance = json.load(error_file)[0]["levenshtein_distance"]
    assert distance == 1


def test_one_edit(tmp_path, example_srt_content, example_whisperx_content):
    """The distance should be 1 when one character is edited."""
    srt_file = tmp_path / "subtitle_no.srt"
    srt_file.write_text(srt.compose(example_srt_content), "utf-8")
    json_file = tmp_path / "whisperx_transcribed_no.json"
    example_whisperx_content_edit = example_whisperx_content.copy()
    example_whisperx_content_edit[0]["text"] = "A" + example_whisperx_content[0]["text"][1:]

    with open(json_file, "w") as f:
        json.dump({"segments": example_whisperx_content_edit}, f)
    compute_transcribed_difference(tmp_path, (lambda x, y: json_file))

    error_file = tmp_path / "autotranscript_error_no.json"
    assert error_file.exists()

    with open(error_file, "r") as error_file:
        distance = json.load(error_file)[0]["levenshtein_distance"]
    assert distance == 1
