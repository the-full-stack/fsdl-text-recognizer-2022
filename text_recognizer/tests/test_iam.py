"""Test for data.iam module."""
from text_recognizer.data.iam import IAM


def test_iam_parsed_words():
    iam = IAM()
    for iam_id in iam.all_ids:
        assert len(iam.word_strings_by_id[iam_id]) == len(iam.word_regions_by_id[iam_id])


def test_iam_parsed_lines():
    iam = IAM()
    for iam_id in iam.all_ids:
        assert len(iam.line_strings_by_id[iam_id]) == len(iam.line_regions_by_id[iam_id])


def test_iam_data_splits():
    iam = IAM()
    assert not set(iam.train_ids) & set(iam.validation_ids)
    assert not set(iam.train_ids) & set(iam.test_ids)
    assert not set(iam.validation_ids) & set(iam.test_ids)
