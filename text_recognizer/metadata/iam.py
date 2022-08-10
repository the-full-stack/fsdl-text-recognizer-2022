import text_recognizer.metadata.shared as shared

RAW_DATA_DIRNAME = shared.DATA_DIRNAME / "raw" / "iam"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
DL_DATA_DIRNAME = shared.DATA_DIRNAME / "downloaded" / "iam"
EXTRACTED_DATASET_DIRNAME = DL_DATA_DIRNAME / "iamdb"

DOWNSAMPLE_FACTOR = 2  # If images were downsampled, the regions must also be.
LINE_REGION_PADDING_X = 8  # add this many pixels around the exact coordinates
LINE_REGION_PADDING_Y = 8  # add this many pixels around the exact coordinates
WORD_REGION_PADDING_X = 4
WORD_REGION_PADDING_Y = 2

PUNCTUATIONS = [
    "!",
    '"',
    "#",
    "&",
    "'",
    "(",
    ")",
    "*",
    "+",
    ",",
    "-",
    ".",
    "/",
    ":",
    ";",
    "?",
]
