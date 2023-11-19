import sys

import ipdb

import torch

from decorator import contextmanager


@contextmanager
def launch_ipdb_on_exception_distributed():
    """Drop into ipdb if an exception is raised in rank zero of a distributed context."""

    try:
        yield
    except Exception:
        # Only the rank zero proc should drop into ipdb.
        # Still need to catch the exception on other ranks,
        # though, or the program will crash while debugging.
        if (
            (0 == torch.distributed.get_rank())
            if torch.distributed.is_initialized()
            else True
        ):
            _, m, tb = sys.exc_info()
            print(m.__repr__(), file=sys.stderr)
            ipdb.post_mortem(tb)
    finally:
        # Re-raise the exception so the program crashes.
        raise


iexd = launch_ipdb_on_exception_distributed()
