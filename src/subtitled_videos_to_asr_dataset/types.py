from typing_extensions import Literal, NotRequired, TypedDict


class Subtitle(TypedDict):
    start: float
    end: float
    text: str


Language = Literal["no", "nn", "en"] | str
Probability = float


class LanguageScore(TypedDict):
    model_id: str
    detected_language: Language
    model_certainty: Probability


class SubtitleWithLanguage(Subtitle):
    detected_spoken_language: list[LanguageScore]


class AlignedSubtitles(TypedDict):
    segments: list[Subtitle]
    word_segments: list[Subtitle]
