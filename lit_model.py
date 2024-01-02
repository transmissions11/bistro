import time
import torch

import lightning as L

from pathlib import Path

from typing import Optional

from lit_gpt import Config, Tokenizer

from utils.loss import compute_loss
from utils.inference import inference_model
from utils.tensors import find_subtensor_end
from utils.curriculum import CurriculumCollate
from utils.padding import strip_right_pad, ignored_tkn
from utils.vicuna import VICUNA_END_OF_USER_PROMPT_SEQUENCE
from utils.hard_prompting import (
    get_non_ascii_tkns,
    get_hard_prompt_gradients,
    create_hard_prompt_candidates,
    test_hard_prompt_candidates,
    insert_hard_prompt_into_template,
)

from model import GPT


class LitModel(L.LightningModule):
    def __init__(
        self,
        model_config: Config,
        tokenizer: Tokenizer,
        ########################################
        hard_prompt_tkn: int,
        num_hard_prompt_tkns: int,
        topk: int,
        candidate_batch_size: int,
        num_candidate_batches: int,
        expansion_loss_threshold: float,
        only_ascii_tkns: bool,
        ########################################
        curriculum_collate: CurriculumCollate,
        checkpoint_path: Optional[Path] = None,
    ):
        super().__init__()

        self.model = None  # This will get set in configure_model.

        # Assign these manually as they don't pickle well
        # or shouldn't be saved via save_hyperparameters.
        self.curriculum_collate = curriculum_collate
        self.checkpoint_path = checkpoint_path

        self.automatic_optimization = False  # We'll handle it ourselves.

        # logger=False since we already log hparams manually in train.py.
        self.save_hyperparameters(
            ignore=["curriculum_collate", "checkpoint_path"], logger=False
        )

        ####################################################################

        self.register_buffer(
            "not_allowed_tokens",
            get_non_ascii_tkns(tokenizer) if only_ascii_tkns else None,
        )

        self.register_buffer(
            "current_hard_prompt",
            tokenizer.encode("✅579 * 640 = 370,560✅", bos=False).to(torch.int64),
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        inputs, targets = batch["inputs"], batch["targets"]  # (b, t), (b, t)

        import random

        for _ in range(100):
            n = random.randint(1, 100)
            import time

            start_time = time.perf_counter()
            with torch.inference_mode():
                loss = compute_loss(
                    self.model,
                    # .repeat(n,1) -> (b * n, t * 1)
                    input_ids=inputs.repeat(n, 1),
                    target_ids=targets.repeat(n, 1),
                    # reduction="none",
                )
            end_time = time.perf_counter()
            print(
                f"n: {n}, loss: {loss} — Computation time: {end_time - start_time} seconds"
            )

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        pass

    def configure_optimizers(self):
        ...  # We don't need an optimizer.

    def configure_model(self):
        # Ensure this function is idempotent, as
        # the trainer may call it multiple times.
        if self.model is not None:
            return

        # Can't use self.print here since it will try
        # to print via the not-yet-existent prog bar.
        def g0_print(msg: str) -> None:
            if self.trainer.is_global_zero:
                print(msg)

            return time.time()  # For timing convenience.

        t0 = g0_print("Initializing model...")
        self.model = GPT(config=self.hparams.model_config)
        g0_print(f"Initialized model in {time.time() - t0:.3f}s.")

        if self.checkpoint_path is not None:
            t0 = g0_print(f"Loading checkpoint weights from {self.checkpoint_path}...")
            self.model.load_state_dict(
                # mmap=True requires checkpoint_path to be a str.
                torch.load(str(self.checkpoint_path), mmap=True),
                strict=False,
                assign=True,
            )
            g0_print(f"Loaded checkpoint weights in {time.time() - t0:.3f}s.")

        g0_print("Done loading & configuring model.")
