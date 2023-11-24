import os

from git import Repo

from datetime import datetime


def get_clean_commit_msg():
    # Often commit messages will start with feat:, fix:, etc.
    # We don't want that in our run names, so we'll strip it out.
    return Repo(".").head.commit.message.strip().split(": ", 1)[-1]


def get_safe_ckpt_dirpath(project: str, run_name: str):
    ckpt_dirpath = f'checkpoints/trained/{project}/{run_name.replace(" ", "-")}'

    if not os.path.exists(ckpt_dirpath):
        return ckpt_dirpath
    else:
        # Append the current datetime to the checkpoint path to avoid the collision.
        ckpt_dirpath = f'{ckpt_dirpath}-{datetime.now().strftime("%m-%d+%H-%M-%S")}'

        if os.path.exists(ckpt_dirpath):  # This shouldn't happen, kick it to the user.
            raise RuntimeError(f"Checkpoint dirpath {ckpt_dirpath} already exists.")

        return ckpt_dirpath
