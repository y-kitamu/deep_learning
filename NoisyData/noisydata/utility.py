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


class Logging(metaclass=Singleton):

    def __init__(self, output_filename=os.path.join(os.path.dirname(__file__), "../log", "log.txt")):
        self.output_filename = output_filename
        dirname = os.path.dirname(output_filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            self.logging("create directory : {}".format(dirname))

    def logging(self, message):
        frameinfo = inspect.stack()[1]
        output_message = "[{} (function {} in file {} at line {})] {}".format(
            datetime.datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S"),
            frameinfo.function, os.path.relpath(frameinfo.filename), frameinfo.lineno, message
        )
        print(output_message)
        if self.output_filename is not None:
            with open(self.output_filename, 'a') as f:
                f.write("{}\n".format(output_message))


class CSVLogger:
    def __init__(self, trainer, metrics, output_filename=None):
        self.trainer = trainer
        self.metrics = metrics
        self.output_filename = output_filename
        if output_filename is None:
            self.output_filename = os.path.join(
                os.path.dirname(__file__), "../log",
                "{}.csv".format(datetime.datetime.now(JST).strftime("%Y%m%d_%H%M%S")))

        dirname = os.path.dirname(self.output_filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with open(self.output_filename, 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(metrics)

    def logging(self):
        with open(self.output_filename, 'a') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                getattr(self.trainer, metric) for metric in self.metrics
            ])


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
