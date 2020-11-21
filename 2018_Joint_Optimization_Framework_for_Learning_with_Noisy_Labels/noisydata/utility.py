import os
import csv
import inspect
import datetime
import pdb, traceback, sys

JST = datetime.timezone(datetime.timedelta(hours=+9), 'JST')


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):

    def __init__(self, output_dir=os.path.join(os.path.dirname(__file__), "../log")):
        self.output_filename = os.path.join(
            output_dir, "{}.log".format(datetime.datetime.now(JST).strftime("%Y%m%d_%H%M%S")))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            self.logging("create directory : {}".format(output_dir))

    def logging(self, message):
        frameinfo = inspect.stack()[1]
        output_message = "[{} (function {} in file {} at line {})] {}".format(
            datetime.datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S"), frameinfo.function,
            os.path.relpath(frameinfo.filename), frameinfo.lineno, message)
        print(output_message)
        if hasattr(self, "output_filename") and self.output_filename is not None:
            with open(self.output_filename, 'a') as f:
                f.write("{}\n".format(output_message))


def run_debug(func):
    """Start pdb debugger at where the `func` throw Exception.
    """
    try:
        res = func()
        return res
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
