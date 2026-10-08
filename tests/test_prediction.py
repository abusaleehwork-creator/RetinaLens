from __future__ import annotations

import pytest

from src.predict import load_saved_pipeline, predict_from_payload, validate_input


@pytest.fixture
def sample_payload() -> dict[str, float]:
    return {
        'quality': 1,
        'pre_screening': 1,
        'ma1': 22,
        'ma2': 22,
        'ma3': 22,
        'ma4': 19,
        'ma5': 18,
        'ma6': 14,
        'exudate1': 49.895756,
        'exudate2': 17.775994,
        'exudate3': 5.27092,
        'exudate4': 5.27092,
        'exudate5': 0.018632,
        'exudate6': 0.006864,
        'exudate7': 0.003923,
        'exudate8': 0.003923,
        'macula_opticdisc_distance': 0.486903,
        'opticdisc_diameter': 0.100025,
        'am_fm_classification': 1,
    }


def test_validate_input_accepts_complete_payload(sample_payload: dict[str, float]) -> None:
    frame = validate_input(sample_payload)
    assert list(frame.columns) == [
        'quality', 'pre_screening', 'ma1', 'ma2', 'ma3', 'ma4', 'ma5', 'ma6', 'exudate1', 'exudate2',
        'exudate3', 'exudate4', 'exudate5', 'exudate6', 'exudate7', 'exudate8', 'macula_opticdisc_distance',
        'opticdisc_diameter', 'am_fm_classification'
    ]
    assert frame.shape == (1, 19)


def test_predict_from_payload_returns_class_and_probability(sample_payload: dict[str, float]) -> None:
    model = load_saved_pipeline()
    prediction, probability = predict_from_payload(sample_payload, model=model)
    assert isinstance(prediction, int)
    assert 0 <= prediction <= 1
    assert 0.0 <= probability <= 1.0


def test_invalid_input_raises_error() -> None:
    with pytest.raises(ValueError):
        validate_input({'quality': 1})
