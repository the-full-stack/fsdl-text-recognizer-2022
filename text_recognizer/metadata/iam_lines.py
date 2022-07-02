import text_recognizer.metadata.emnist as emnist
import text_recognizer.metadata.shared as shared

PROCESSED_DATA_DIRNAME = shared.DATA_DIRNAME / "processed" / "iam_lines"

CHAR_WIDTH = emnist.INPUT_SHAPE[0]
IMAGE_HEIGHT = 56
IMAGE_WIDTH = 2048  # rounding up IAMLines empirical maximum width to a power of two

DIMS = (1, IMAGE_HEIGHT, IMAGE_WIDTH)
OUTPUT_DIMS = (89, 1)

MAPPING = emnist.MAPPING
