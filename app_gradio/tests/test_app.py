import os
import signal

from app_gradio import app


os.environ["CUDA_VISIBLE_DEVICES"] = ""


def test_local_run():
    """A quick test to make sure we can build the application locally."""
    backend = app.PredictorBackend()
    frontend = app.make_frontend(fn=backend.run)

    signal.setitimer(signal.ITIMER_REAL, 10)  # set a timer for ten seconds, then
    try:
        frontend.launch(share=False, debug=True)  # run the server
    except TimeoutSignal:  # and after the timer goes off
        return  # we've passed the test -- server ran for 10s without error


# For details on the below strategy for terminating Python code
#   see StackOverflow discussion here: https://stackoverflow.com/a/25027182


class TimeoutSignal(Exception):
    """A simple class to represent the timeout above."""


def handle_timeout(signum, frame):
    """Converts a system signal to a TimeoutSignal."""
    raise TimeoutSignal


signal.signal(signal.SIGALRM, handle_timeout)
