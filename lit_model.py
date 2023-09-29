import time
import torch

from torch.nn import functional as F

import lightning as L

from pathlib import Path

from typing import Optional

from lit_gpt import Config, Tokenizer

from utils.inference import inference_model
from utils.tensors import find_subtensor_end
from utils.padding import strip_right_pad, ignored_tkn
from utils.vicuna import VICUNA_END_OF_USER_PROMPT_SEQUENCE

from model import GPT


class LitModel(L.LightningModule):
    def __init__(
        self,
        model_config: Config,
        tokenizer: Tokenizer,
        ###############################
        # If None, will use random weights.
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

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        inputs, targets = batch["inputs"], batch["targets"]
        loss = self.compute_loss(inputs, targets)

        # embed_weights = self.model.transformer.wte.weight

        # # TODO: separate slice of input_ids

        # one_hot = torch.zeros(
        #     input_ids.shape[0],
        #     embed_weights.shape[0],
        #     device=self.device,
        #     dtype=embed_weights.dtype,
        # )

        # one_hot.scatter_(
        #     1,
        #     input_ids.unsqueeze(1),
        #     torch.ones(
        #         one_hot.shape[0], 1, device=self.device, dtype=embed_weights.dtype
        #     ),
        # )

        # one_hot.requires_grad_()

        # input_embeds = (one_hot @ embed_weights).unsqueeze(0)

        # # now stitch it together with the rest of the embeddings
        # embeds = self.model.transformer.wte(input_ids.unsqueeze(0)).detach()
        # full_embeds = torch.cat(
        #     [
        #         embeds[:, : input_slice.start, :],
        #         input_embeds,
        #         embeds[:, input_slice.stop :, :],
        #     ],
        #     dim=1,
        # )

        # loss = self.compute_loss(targets, input_embeds=full_embeds)

        # # loss.backward()
        # self.manual_backward(loss)

        # grad = one_hot.grad.clone()

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        inputs, targets = batch["inputs"], batch["targets"]
        loss = self.compute_loss(inputs, targets)

        self.log(
            "val_loss",
            # Need to upcast precision as types like bfloat16
            # have very low precision with larger values (~256+)
            # that results in inaccurate accumulation w/ on_epoch.
            # https://github.com/Lightning-AI/lightning/issues/18620
            loss.to(torch.float32),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Log a few sample inferences from the validation set to W&B.
        if batch_idx == 0:
            tokenizer = self.hparams.tokenizer

            prompt_end_tkns = tokenizer.encode(
                VICUNA_END_OF_USER_PROMPT_SEQUENCE, device=self.device
            )

            def process_val_sample(sample, target):
                sample = strip_right_pad(sample)
                target = strip_right_pad(target)

                input_ids = sample[: find_subtensor_end(sample, prompt_end_tkns) + 1]

                return (
                    tokenizer.decode(input_ids),
                    tokenizer.decode(
                        inference_model(
                            self.model,
                            input_ids,
                            temperature=0.00,  # Sample 100% greedily.
                            max_new_tokens=100,  # Should hit an eos token first.
                            eos_id=tokenizer.eos_id,
                        )
                    ),
                    # Note: target != ignored_tkn strips away ignored_tkn tokens entirely,
                    # which may lead to confusion if ignored_tkn is used between real tokens.
                    tokenizer.decode(target[target != ignored_tkn]),
                )

            self.logger.log_text(
                key="val_samples",
                columns=["input", "output", "target"],
                data=[
                    process_val_sample(inputs[i], targets[i])
                    for i in range(len(inputs))  # Full batch.
                ],
            )

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

    def compute_loss(self, inputs, targets):
        logits = self.model(input_ids=inputs)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignored_tkn
        )
