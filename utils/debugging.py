import sys

import torch

from ipdb import post_mortem, contextmanager


@contextmanager
def iexd():
    """Drop into ipdb if an exception is raised in rank zero of a distributed context."""

    if (
        # Only rank zero should drop into ipdb.
        (0 == torch.distributed.get_rank())
        if torch.distributed.is_initialized()
        else True
    ):
        try:
            yield
        except Exception as e:
            print(e.__repr__(), file=sys.stderr)
            _, _, tb = sys.exc_info()
            post_mortem(tb)
        # TODO: wait do we need pass?
    else:
        yield
