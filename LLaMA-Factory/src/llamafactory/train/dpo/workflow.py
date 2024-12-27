# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/examples/scripts/dpo.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, List, Optional

from ...data import Ul2PairwiseDataCollatorWithPadding, PairwiseDataCollatorWithPadding, get_dataset
from ...extras.constants import IGNORE_INDEX
from ...extras.ploting import plot_loss
from ...hparams import ModelArguments
from ...model import load_model, load_tokenizer
from ..trainer_utils import extend_vocab, get_extended_embeddings_state_dict, create_modelcard_and_push, create_ref_model
from .trainer import CustomDPOTrainer
import torch

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments


def run_dpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]

    # please finish extending vocab before tokenizing dataset
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if finetuning_args.stage == "ul2_dpo":
        model, tokenizer = extend_vocab(model, tokenizer, finetune_new_vocab=finetuning_args.ul2_finetune_embedding)

        if model_args.adapter_name_or_path:
            print("model args", model_args.adapter_name_or_path)
            state_dict_path = f"{model_args.adapter_name_or_path[0]}/global_step{model_args.adapter_name_or_path[0].split('-')[-1]}/mp_rank_00_model_states.pt"
            state_dict = torch.load(state_dict_path)["module"]
            # load extra parameters for embedding layer
            embedding = model.get_input_embeddings()
            embedding_state_dict = get_extended_embeddings_state_dict(state_dict, extra_param_name="extra_token_embeddings")

            embedding.load_state_dict(embedding_state_dict, strict=False)

            collator_class = Ul2PairwiseDataCollatorWithPadding
    else:
        collator_class = PairwiseDataCollatorWithPadding

    dataset_module = get_dataset(model_args, data_args, training_args, stage="rm", **tokenizer_module)
    
    data_collator = collator_class(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )

    # Create reference model
    if finetuning_args.use_ref_model:
        if finetuning_args.ref_model is None and (not training_args.do_train):  # use the model itself
            ref_model = model
        elif finetuning_args.stage == "ul2_dpo" and model_args.adapter_name_or_path:
            ref_tokenizer_module = load_tokenizer(model_args)
            ref_tokenizer = ref_tokenizer_module["tokenizer"]
            ref_model = load_model(
                ref_tokenizer, model_args, finetuning_args, is_trainable=False
            )

            ref_model, _ = extend_vocab(ref_model, ref_tokenizer, finetune_new_vocab=False)
            ref_embedding = ref_model.get_input_embeddings()

            print("ref_embedding type", type(ref_embedding))
            for name, _ in ref_embedding.named_parameters():
                print(name)
            print("embedding state dict keys", embedding_state_dict.keys())

            ref_embedding.load_state_dict(embedding_state_dict, strict=False)
        else:
            ref_model = create_ref_model(model_args, finetuning_args)
    else:
        ref_model = None

    # Update arguments
    training_args.remove_unused_columns = False  # important for pairwise dataset

    # Initialize our Trainer
    trainer = CustomDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
    )
    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "rewards/accuracies"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        if id(model) == id(ref_model):  # unable to compute rewards if reference model is the model itself
            remove_keys = [key for key in metrics.keys() if "rewards" in key]
            for key in remove_keys:
                metrics.pop(key)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
