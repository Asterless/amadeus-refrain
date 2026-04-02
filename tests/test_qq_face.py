"""QQ Face emoji mapping tests."""

from src.constants.qq_face import QQ_FACE, face_to_text


def test_known_face_ids() -> None:
    assert QQ_FACE[0] == "惊讶"
    assert QQ_FACE[14] == "微笑"
    assert QQ_FACE[178] == "捂脸"


def test_face_to_text_known() -> None:
    assert face_to_text(14) == "[微笑]"
    assert face_to_text(178) == "[捂脸]"


def test_face_to_text_unknown() -> None:
    assert face_to_text(99999) == "[表情]"


def test_mapping_has_common_faces() -> None:
    """Ensure the mapping covers the most commonly used QQ faces."""
    common_ids = [0, 1, 2, 4, 5, 6, 9, 10, 11, 12, 13, 14, 21, 32, 49, 53, 78, 79]
    for fid in common_ids:
        assert fid in QQ_FACE, f"Missing common face id {fid}"
