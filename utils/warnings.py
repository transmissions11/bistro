import warnings


def suppress_uncontrollable_warnings():
    """Filter incorrect or "out of our control" warnings."""
    warnings.filterwarnings(
        "ignore",
        message=r".*DtypeTensor constructors are no longer recommended.*",
        module="wandb",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*_histc_cuda does not have a deterministic implementation.*",
        module="wandb",
    )


def elevate_important_warnings():
    """Elevate warnings we want to treat as errors."""
    warnings.filterwarnings("error", message=r".*Checkpoint directory .+ not empty.*")
