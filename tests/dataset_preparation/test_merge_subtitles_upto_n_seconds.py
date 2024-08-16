from subtitled_videos_to_asr_dataset.dataset_preparation.create_huggingface_dataset import merge_subtitles_upto_n_sec


def test_empty_list():
    assert merge_subtitles_upto_n_sec([]) == []


def test_single_item():
    subtitle = {"start": 10, "end": 25, "text": ""}
    assert merge_subtitles_upto_n_sec([(0, subtitle)]) == [subtitle]


def test_no_merge_cause_index():
    subtitle1 = {"start": 10, "end": 12, "text": ""}
    subtitle2 = {"start": 14, "end": 16, "text": ""}
    assert merge_subtitles_upto_n_sec([(1, subtitle1), (3, subtitle2)]) == [subtitle1, subtitle2]


def test_no_merge_cause_length():
    subtitle1 = {"start": 10, "end": 25, "text": ""}
    subtitle2 = {"start": 30, "end": 45, "text": ""}
    assert merge_subtitles_upto_n_sec([(1, subtitle1), (2, subtitle2)], n=30) == [subtitle1, subtitle2]


def test_merge():
    subtitle1 = {"start": 0, "end": 5, "text": "foo"}
    subtitle2 = {"start": 10, "end": 20, "text": "bar baz"}
    assert merge_subtitles_upto_n_sec([(0, subtitle1), (1, subtitle2)]) == [
        {"start": 0, "end": 20, "text": "foo bar baz"}
    ]


def test_merge_multiple():
    subtitle1 = {"start": 0, "end": 5, "text": "foo"}
    subtitle2 = {"start": 10, "end": 20, "text": "bar baz"}
    subtitle3 = {"start": 20, "end": 25, "text": "beep boop"}
    subtitle4 = {"start": 0, "end": 5, "text": "foo"}
    subtitle5 = {"start": 20, "end": 25, "text": "fighters"}
    subtitle6 = {"start": 27, "end": 29, "text": "yeah"}

    assert merge_subtitles_upto_n_sec(
        [(0, subtitle1), (1, subtitle2), (3, subtitle3), (109, subtitle4), (110, subtitle5), (111, subtitle6)]
    ) == [
        {"start": 0, "end": 20, "text": "foo bar baz"},
        subtitle3,
        {"start": 0, "end": 29, "text": "foo fighters yeah"},
    ]
