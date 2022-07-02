"""Provide an image of handwritten text and get back out a string!"""
import argparse
import json
import logging
import os
from pathlib import Path
from time import time
from uuid import uuid4

import gradio as gr
from PIL import ImageStat
import requests

from text_recognizer.paragraph_text_recognizer import ParagraphTextRecognizer
import text_recognizer.util as util

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # do not use GPU

logging.basicConfig(level=logging.INFO)
LOG_DIR = Path("logs")  # where do we log images?
IMAGE_LOG_DIR = LOG_DIR / "images"
IMAGE_LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PORT = 11700


def main(args):
    predictor = PredictorBackend(url=args.model_url)
    frontend = make_frontend(predictor.run)
    frontend.launch(
        server_name="0.0.0.0",  # make server accessible, binding all interfaces  # noqa: S104
        server_port=args.port,  # set a port to bind to, failing if unavailable
        share=True,  # should we create a (temporary) public link on gradio.app?
        favicon_path="app_gradio/1f95e.png",  # what icon should we display in the address bar?
    )


def make_frontend(fn):
    examples_dir = Path("text_recognizer") / "tests" / "support" / "paragraphs"
    example_fnames = [elem for elem in os.listdir(examples_dir) if elem.endswith(".png")]
    example_paths = [examples_dir / fname for fname in example_fnames]

    examples = [[str(path)] for path in example_paths]

    readme = """# Text Recognizer"""

    # build a basic browser interface to a Python function
    frontend = gr.Interface(
        fn=fn,  # which Python function are we interacting with?
        outputs="text",  # what output widgets does it need? the default text widget
        # what input widgets does it need? we configure an image widget
        inputs=gr.inputs.Image(type="pil", invert_colors=False, label="Handwritten Text"),
        theme="default",  # what theme should we use? provide css kwarg to set your own style
        title="📝 Text Recognizer",  # what should we display at the top of the page?
        thumbnail="app_gradio/1f95e.png",  # what should we display when the link is shared, e.g. on social media?
        description=__doc__,  # what should we display just above the interface?
        article=readme,  # what long-form content should we display below the interface?
        examples=examples,  # which potential inputs should we provide?
        cache_examples=False,  # should we cache those inputs for faster inference? slows down start
        flagging_options=["incorrect", "unusable", "offensive", "other"],  # what options do users have for feedback?
    )

    return frontend


class PredictorBackend:
    """Interface to a backend that serves predictions.

    To communicate with a backend accessible via a URL,
    provide the url kwarg.

    Otherwise, runs a predictor locally.
    """

    def __init__(self, url=None):
        if url is not None:
            self.url = url
            self._predict = self._predict_from_endpoint
        else:
            model = ParagraphTextRecognizer()
            self._predict = model.predict

    def run(self, image):
        pred, metrics = self._predict_with_metrics(image)
        self._log_inference(pred, metrics)
        return pred

    def _predict_with_metrics(self, image):
        pred = self._predict(image)

        fname = IMAGE_LOG_DIR / f"{int(time())}-{uuid4()}.png"
        image.save(fname)
        stats = ImageStat.Stat(image)
        metrics = {
            "image_mean_intensity": stats.mean,
            "image_median": stats.median,
            "image_extrema": stats.extrema,
            "image_area": image.size[0] * image.size[1],
            "image_filename": fname,
            "pred_length": len(pred),
        }
        return pred, metrics

    def _predict_from_endpoint(self, image):
        encoded_image = util.encode_b64_image(image)

        headers = {"Content-type": "application/json"}
        payload = json.dumps({"image": "data:image/png;base64," + encoded_image})

        response = requests.post(self.url, data=payload, headers=headers)
        pred = response.json()["pred"]

        return pred

    def _log_inference(self, pred, metrics):
        for key, value in metrics.items():
            logging.info(f"METRIC {key} {value}")
        logging.info(f"PRED >begin\n{pred}\nPRED >end")


def _make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_url",
        default=None,
        type=str,
        help="Identifies a URL to which to send image data. Data is base64-encoded, converted to a utf-8 string, and then set via a POST request. Default is None, which instead sends the data to a model created locally.",
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        type=int,
        help="Port on which to expose this server. Default is {DEFAULT_PORT}",
    )

    return parser


if __name__ == "__main__":
    parser = _make_parser()
    args = parser.parse_args()
    main(args)
