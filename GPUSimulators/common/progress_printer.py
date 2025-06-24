import logging
import time

import numpy as np


def time_string(seconds):
    seconds = int(max(seconds, 1))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    periods = [('h', hours), ('m', minutes), ('s', seconds)]
    return_string = ' '.join('{}{}'.format(value, name)
                             for name, value in periods
                             if value)
    return return_string


def progress_bar(step, total_steps, width=30):
    progress = np.round(width * step / total_steps).astype(np.int32)
    progressbar = "0% [" + "#" * progress + "=" * (width - progress) + "] 100%"
    return progressbar


class ProgressPrinter(object):
    """
    Small helper class for creating a progress bar
    """

    def __init__(self, total_steps, print_every=5):
        self.logger = logging.getLogger(__name__)
        self.start = time.time()
        self.total_steps = total_steps
        self.print_every = print_every
        self.next_print_time = self.print_every
        self.last_step = 0
        self.secs_per_iter = None

    def get_print_string(self, step):
        elapsed = time.time() - self.start
        if elapsed > self.next_print_time:
            dt = elapsed - (self.next_print_time - self.print_every)
            dsteps = step - self.last_step
            steps_remaining = self.total_steps - step

            if dsteps == 0:
                return None

            self.last_step = step
            self.next_print_time = elapsed + self.print_every

            if not self.secs_per_iter:
                self.secs_per_iter = dt / dsteps
            self.secs_per_iter = 0.2 * self.secs_per_iter + 0.8 * (dt / dsteps)

            remaining_time = steps_remaining * self.secs_per_iter

            return (f"{progress_bar(step, self.total_steps)}. "
                    + f"Total: {time_string(elapsed + remaining_time)}, "
                    + f"elapsed: {time_string(elapsed)}, "
                    + f"remaining: {time_string(remaining_time)}")

        return None
