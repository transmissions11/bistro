import warnings


def suppress_uncontrollable_warnings():
    """Filter incorrect or "out of our control" warnings."""

    warnings.filterwarnings(
        "ignore",
        # https://github.com/wandb/wandb/issues/6227
        message=r".*DtypeTensor constructors are no longer recommended.*",
        module="wandb",
    )
    warnings.filterwarnings(
        "ignore",
        # internal wandb error with deterministic CUDA
        message=r".*_histc_cuda does not have a deterministic implementation.*",
        module="wandb",
    )
    warnings.filterwarnings(
        "ignore",
        # https://github.com/Lightning-AI/pytorch-lightning/issues/12862
        message=r".*Using `DistributedSampler` with the dataloaders",
        module="lightning",
    )


def elevate_important_warnings():
    """Elevate warnings we want to treat as errors."""

    warnings.filterwarnings(
        "error", message=r".*Checkpoint directory .+ not empty.*", module="lightning"
    )
