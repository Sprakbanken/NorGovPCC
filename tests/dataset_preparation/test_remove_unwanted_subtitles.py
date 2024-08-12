from subtitled_videos_to_asr_dataset.dataset_preparation.create_huggingface_dataset import remove_unwanted_subtitles

subtitle_1 = {
    "start": 0,
    "end": 10,
    "text": "foo bar",
    "detected_spoken_language": [{"model_id": "test_model", "detected_language": "no", "model_certainty": 50.5}],
}

subtitle_2 = {
    "start": 10,
    "end": 15,
    "text": "bar baz",
    "detected_spoken_language": [{"model_id": "test_model", "detected_language": "no", "model_certainty": 33.33}],
}

subtitle_3 = {
    "start": 15,
    "end": 20,
    "text": "spruddel",
    "detected_spoken_language": [{"model_id": "test_model", "detected_language": "en", "model_certainty": 66.67}],
}


def test_empty():
    assert remove_unwanted_subtitles([]) == []


def test_single_remove():
    assert remove_unwanted_subtitles([subtitle_1], preferred_langdetect_model="supermodell") == []
    assert (
        remove_unwanted_subtitles([subtitle_1], preferred_langdetect_model="test_model", preferred_langcode="sw") == []
    )
    assert (
        remove_unwanted_subtitles(
            [subtitle_1], preferred_langdetect_model="test_model", preferred_langcode="no", max_segment_length=5
        )
        == []
    )


def test_single_keep():
    assert remove_unwanted_subtitles([subtitle_1], preferred_langdetect_model="test_model") == [(0, subtitle_1)]


def test_multiple():
    segments = [subtitle_1, subtitle_2, subtitle_3]
    assert remove_unwanted_subtitles(segments, preferred_langdetect_model="test_model", preferred_langcode="no") == [
        (0, subtitle_1),
        (1, subtitle_2),
    ]
    assert remove_unwanted_subtitles(segments, preferred_langdetect_model="test_model", preferred_langcode="en") == [
        (2, subtitle_3)
    ]
    assert remove_unwanted_subtitles(
        segments, preferred_langdetect_model="test_model", preferred_langcode="no", max_segment_length=5
    ) == [(1, subtitle_2)]
