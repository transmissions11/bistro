import time
import torch

import lightning as L

from pathlib import Path

from typing import Optional

from lit_gpt import Config, Tokenizer

from utils.loss import compute_loss
from utils.inference import inference_model
from utils.tensors import find_subtensor_end
from utils.padding import strip_right_pad, ignored_tkn
from utils.vicuna import VICUNA_END_OF_USER_PROMPT_SEQUENCE
from utils.hard_prompting import (
    get_non_ascii_tkns,
    get_hard_prompt_gradients,
    create_hard_prompt_candidates,
    clean_hard_prompt_candidates,
    test_hard_prompt_candidates,
)

from model import GPT


class LitModel(L.LightningModule):
    def __init__(
        self,
        model_config: Config,
        tokenizer: Tokenizer,
        #######################################
        hard_prompt_tkn: int,
        num_hard_prompt_tkns: int,
        only_ascii_tkns: bool = True,
        # Where 1 means no accumulation.
        accumulate_grad_batches: int = 1,
        #######################################
        checkpoint_path: Optional[Path] = None,
    ):
        super().__init__()

        self.model = None  # This will get set in configure_model.

        # Assign these manually as they don't pickle well
        # or shouldn't be saved via save_hyperparameters.
        self.checkpoint_path = checkpoint_path

        self.automatic_optimization = False  # We'll handle it ourselves.

        # logger=False since we already log hparams manually in train.py.
        self.save_hyperparameters(ignore=["checkpoint_path"], logger=False)

        #####################################################################

        # TODO: benchmark this
        self.register_buffer(
            "not_allowed_tokens",
            get_non_ascii_tkns(tokenizer) if only_ascii_tkns else None,
        )

        self.register_buffer(
            "current_hard_prompt",
            torch.tensor(
                [hard_prompt_tkn] * num_hard_prompt_tkns,
                dtype=torch.int64,
            ),
        )

        assert (
            self.current_hard_prompt.size(0) == num_hard_prompt_tkns
        ), f"hard prompt size mismatch {self.current_hard_prompt.size(0)} != {num_hard_prompt_tkns}"

        #####################################################################

        self.hard_prompt_step = 0.0

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        self.print(f"-------- START HARD PROMPT STEP {self.hard_prompt_step} --------")

        # fmt: off
        input_ids = torch.tensor([
                    [ 319, 13563,  1546,   263, 12758,  1404,   322,   385, 23116, 21082,
        20255, 29889,   450, 20255,  4076,  8444, 29892, 13173, 29892,   322,
         1248,   568,  6089,   304,   278,  1404, 29915, 29879,  5155, 29889,
         3148,  1001, 29901,  1128, 31681, 31681, 31681, 31681, 31681, 31681,
        31681, 31681, 31681, 31681, 31681, 31681, 31681, 31681, 31681, 31681,
        31681, 31681, 31681, 29871, 29946, 29953, 29906,   334, 29871, 29945,
        29955, 29929,   353,   319,  1799,  9047, 13566, 29901, 29871, 29906,
        29953, 29955, 29892, 29946, 29929, 29947],
                    [  319, 13563,  1546,   263, 12758,  1404,   322,   385, 23116, 21082,
        20255, 29889,   450, 20255,  4076,  8444, 29892, 13173, 29892,   322,
         1248,   568,  6089,   304,   278,  1404, 29915, 29879,  5155, 29889,
         3148,  1001, 29901,  1128, 31681, 31681, 31681, 31681, 31681, 31681,
        31681, 31681, 31681, 31681, 31681, 31681, 31681, 31681, 31681, 31681,
        31681, 31681, 31681, 29871, 29946, 29953, 29906,   334, 29871, 29945,
        29955, 29929,   353,   319,  1799,  9047, 13566, 29901, 29871, 29906,
        29953, 29955, 29892, 29946, 29929, 29947]
        ]).to(self.device)

        target_ids = torch.tensor([
                    [   -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
           -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
           -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
           -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
           -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
           -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
           -1,    -1,    -1,    -1,    -1,    -1,    -1, 29871, 29906, 29953,
        29955, 29892, 29946, 29929, 29947,     2],
                    [   -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
           -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
           -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
           -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
           -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
           -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
           -1,    -1,    -1,    -1,    -1,    -1,    -1, 29871, 29906, 29953,
        29955, 29892, 29946, 29929, 29947,     2]
        ]).to(self.device)
        # fmt: on

        self.print(
            "SHAPES",
            f"input_ids: {input_ids.size()}",
            f"target_ids: {target_ids.size()}",
        )

        self.print("--------------------------BOTH-------------------------")

        loss = compute_loss(
            # reduce=False to get the loss for each sequence in the batch.
            self.model,
            input_ids=input_ids,
            target_ids=target_ids,
            reduction="none",
        ).view(2, -1)

        self.print(loss)

        self.print("--------------------------FIRST-------------------------")

        loss = compute_loss(
            # reduce=False to get the loss for each sequence in the batch.
            self.model,
            input_ids=input_ids[0].unsqueeze(0),
            target_ids=target_ids[0].unsqueeze(0),
            reduction="none",
        )

        self.print(loss)

        ####################################################################

        self.log(
            "hard_prompt_step",
            # We need to specify float32 or we'll get annoying precision issues.
            # Bug report: https://github.com/Lightning-AI/lightning/issues/18984
            torch.tensor(self.hard_prompt_step, dtype=torch.float32),
        )
        self.hard_prompt_step += 1.0

        ####################################################################

        if self.hard_prompt_step == 1.0:
            raise ValueError("DONE")

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        inputs, targets = batch["inputs"], batch["targets"]

        candidate_losses = test_hard_prompt_candidates(
            self.model,
            hard_prompt_candidates=self.current_hard_prompt.unsqueeze(0),
            hard_prompt_tkn=self.hparams.hard_prompt_tkn,
            input_ids=inputs,
            target_ids=targets,
        )

        loss = torch.min(candidate_losses)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

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

    def on_train_start(self):
        self.print("\nResetting model caches for training...\n")
        self.model.reset_caches()
