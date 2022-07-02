import text_recognizer.metadata.emnist as emnist
import text_recognizer.metadata.shared as shared

PROCESSED_DATA_DIRNAME = shared.DATA_DIRNAME / "processed" / "emnist_lines2"

CHAR_HEIGHT, CHAR_WIDTH = emnist.DIMS[1:3]
DIMS = (emnist.DIMS[0], CHAR_HEIGHT, None)  # width variable, depends on maximum sequence length

MAPPING = emnist.MAPPING

IMAGE_HEIGHT, IMAGE_WIDTH = 56, 1024
IMAGE_X_PADDING = 28
MAX_OUTPUT_LENGTH = 89
