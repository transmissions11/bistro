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
            _, message, traceback = sys.exc_info()
            print(message.__repr__(), file=sys.stderr)
            ipdb.post_mortem(traceback)
    finally:
        # Make sure all procs are synced up before exiting.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

iexd = launch_ipdb_on_exception_distributed()


