import os
import sys
import contextlib
import warnings
from IPython.display import display
from ipywidgets import Output


@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = Output()
            try:
                with output:
                    display(output, display_id='suppress_output')
                    yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            output.clear_output(wait=True)
