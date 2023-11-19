import sys
import ipdb
import torch
from decorator import contextmanager


def is_rank_zero():
    return (
        (0 == torch.distributed.get_rank())
        if torch.distributed.is_initialized()
        else True
    )


def barrier():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def debug_distributed():
    ipdb.set_trace(cond=is_rank_zero())

    barrier()  # Make sure all procs are synced up before continuing.


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
            _, message, traceback = sys.exc_info()
            print(message.__repr__(), file=sys.stderr)
            ipdb.post_mortem(traceback)
    finally:
        barrier()  # Make sure all procs are synced up before continuing.


iexd = launch_ipdb_on_exception_distributed()
