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

        self.register_buffer(
            "accumulated_grads",
            torch.zeros(
                self.hparams.num_hard_prompt_tkns,
                self.hparams.tokenizer.vocab_size,
                dtype=torch.float64,
            ),
            persistent=False,
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        # TODO: make sure i didnt mess anything up in the big commit (30e3365)

        ####################

        inputs, targets = batch["inputs"], batch["targets"]

        # TODO: ablate these for performance

        # TODO; we could support input ids with batches here, just mean properly

        # Compute and accumulate the gradients for the hard prompt.
        # .type_as() is needed to upcast the gradients to the
        # higher precision type used by self.accumulated_grads.
        self.accumulated_grads += self.all_gather(
            get_hard_prompt_gradients(
                self.model,
                current_hard_prompt=self.current_hard_prompt,
                hard_prompt_tkn=self.hparams.hard_prompt_tkn,
                input_ids=inputs,
                target_ids=targets,
            ).type_as(self.accumulated_grads)
        ).mean(dim=0)

        # We need to use batch_idx + 1 here since batch_idx starts at 0, which
        # would cause the first batch to trigger an update before accumulating.
        if (batch_idx + 1) % self.hparams.accumulate_grad_batches == 0:
            # Use the accumulated gradients for the update.
            # + 1 because grad accumulation steps are on top of the normal step.
            # e.g. if grad_accumulation_steps=1, two batches are needed to update.
            grads = self.accumulated_grads / self.hparams.accumulate_grad_batches

            self.accumulated_grads.zero_()  # Reset the accumulated gradients.

            candidate_batch_size = 1
            num_candidate_batches_to_test = 2

            num_candidates = candidate_batch_size * num_candidate_batches_to_test

            # TODO: support grad accumulation iters essentially (split into multiple batches)
            hard_prompt_candidates = create_hard_prompt_candidates(
                current_hard_prompt=self.current_hard_prompt,
                hard_prompt_grads=grads,
                num_candidates=num_candidates,  # TODO: find a good value and make this configurable
                not_allowed_tokens=self.not_allowed_tokens,
                topk=50,
            )

            # TODO: make sure cands are all in the same place

            hard_prompt_candidates = clean_hard_prompt_candidates(
                self.hparams.tokenizer,
                current_hard_prompt=self.current_hard_prompt,
                hard_prompt_candidates=hard_prompt_candidates,
            )

            # TODO: ensure every proc has the same cands

            candidate_losses = self.all_gather(
                test_hard_prompt_candidates(
                    self.model,
                    candidate_batch_size=candidate_batch_size,
                    hard_prompt_candidates=hard_prompt_candidates,
                    hard_prompt_tkn=self.hparams.hard_prompt_tkn,
                    input_ids=inputs,
                    target_ids=targets,
                )
            ).mean(dim=0)

            min_loss_candidate_idx = torch.argmin(candidate_losses).item()
            min_loss = candidate_losses[min_loss_candidate_idx]

            # TODO: have rank zero do this? hm can test w/ print
            self.current_hard_prompt = hard_prompt_candidates[min_loss_candidate_idx]

            self.log("train_loss", min_loss)

            self.log(
                "hard_prompt_step",
                # We need to specify float32 or we'll get annoying precision issues.
                # Bug report: https://github.com/Lightning-AI/lightning/issues/18984
                torch.tensor(self.hard_prompt_step, dtype=torch.float32),
            )
            self.hard_prompt_step += 1.0

            for i in range(2):
                self.print(
                    f"CAND {i}",
                    self.hparams.tokenizer.decode(hard_prompt_candidates[i]),
                )
            self.print("CAND LOSSES", candidate_losses)
            # if self.hard_prompt_step == 3.0:
            #     raise ValueError("DONE")

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

        # # Log a few sample inferences from the validation set to W&B.
        # if batch_idx == 0:
        #     tokenizer = self.hparams.tokenizer

        #     prompt_end_tkns = tokenizer.encode(
        #         VICUNA_END_OF_USER_PROMPT_SEQUENCE, device=self.device
        #     )

        #     def process_val_sample(sample, target):
        #         sample = strip_right_pad(sample)
        #         target = strip_right_pad(target)

        #         input_ids = sample[: find_subtensor_end(sample, prompt_end_tkns) + 1]

        #         return (
        #             tokenizer.decode(input_ids),
        #             tokenizer.decode(
        #                 inference_model(
        #                     self.model,
        #                     input_ids,
        #                     temperature=0.00,  # Sample 100% greedily.
        #                     max_new_tokens=100,  # Should hit an eos token first.
        #                     eos_id=tokenizer.eos_id,
        #                 )
        #             ),
        #             # Note: target != ignored_tkn strips away ignored_tkn tokens entirely,
        #             # which may lead to confusion if ignored_tkn is used between real tokens.
        #             tokenizer.decode(target[target != ignored_tkn]),
        #         )

        #     self.logger.log_text(
        #         key="val_samples",
        #         columns=["input", "output", "target"],
        #         data=[
        #             process_val_sample(inputs[i], targets[i])
        #             for i in range(len(inputs))  # Full batch.
        #         ],
        #     )

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

        # Lightning calls .to() on all registered buffers during setup which
        # will set the dtype to the default dtype. We need to reset it here.
        # Bug report: https://github.com/Lightning-AI/lightning/issues/18982
        self.accumulated_grads = self.accumulated_grads.type(torch.float64)
