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

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        inputs, targets = batch["inputs"], batch["targets"]

        # TODO: ablate these for performance

        hard_prompt_grads = get_hard_prompt_gradients(
            self.model,
            current_hard_prompt=self.current_hard_prompt,
            hard_prompt_tkn=self.hparams.hard_prompt_tkn,
            input_ids=inputs,
            target_ids=targets,
        )

        # TODO: limit to ascii

        # TODO: support grad accum iters essentially (split into multiple batches)
        hard_prompt_candidates = create_hard_prompt_candidates(
            current_hard_prompt=self.current_hard_prompt,
            hard_prompt_grads=hard_prompt_grads,
            batch_size=100,  # TODO: FIND A GOOD VALUE!!!! MAKE THIS CONFIG
            not_allowed_tokens=self.not_allowed_tokens,
            topk=10,
        )

        hard_prompt_candidates = clean_hard_prompt_candidates(
            self.hparams.tokenizer,
            current_hard_prompt=self.current_hard_prompt,
            hard_prompt_candidates=hard_prompt_candidates,
        )

        (min_loss, best_candidate_idx) = test_hard_prompt_candidates(
            self.model,
            hard_prompt_candidates=hard_prompt_candidates,
            hard_prompt_tkn=self.hparams.hard_prompt_tkn,
            input_ids=inputs,
            target_ids=targets,
        )

        self.current_hard_prompt = hard_prompt_candidates[best_candidate_idx]

        self.log("train_loss", min_loss, prog_bar=True)

        if batch_idx % 20 == 0:
            print("PROMPT", self.hparams.tokenizer.decode(self.current_hard_prompt))

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        # inputs, targets = batch["inputs"], batch["targets"]
        # loss = compute_loss(self.model, input_ids=inputs, target_ids=targets)

        # self.log(
        #     "val_loss",
        #     # Need to upcast precision as types like bfloat16
        #     # have very low precision with larger values (~256+)
        #     # that results in inaccurate accumulation w/ on_epoch.
        #     # https://github.com/Lightning-AI/lightning/issues/18620
        #     loss.to(torch.float32),
        #     on_epoch=True,
        #     prog_bar=True,
        #     sync_dist=True,
        # )

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
        ...

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
        self.print(f"\nResetting model caches for training...\n")
        self.model.reset_caches()
