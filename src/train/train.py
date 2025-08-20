#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""

import argparse
import logging
import os
from itertools import chain
from pathlib import Path
import re
import json

import datasets
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
)
from transformers.utils.versions import require_version
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.utils.data import DataLoader
from datasets.distributed import split_dataset_by_node

import time
from accelerate import InitProcessGroupKwargs
from datetime import timedelta
import evaluate
from src.eval.zeroshot_retrieval import prepare_dataset, evaluate_retrieval
import glob

import hashlib
import base64

logger = get_logger(__name__)
require_version(
    "datasets>=2.14.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def estimate_flops(batch_size, seq_len, num_layers, d_model):
    """
    Transformer の FLOPs を近似計算
    """
    # FLOPs の理論値（Self-Attention + FFN の合計）
    flops = 2 * num_layers * (4 * d_model**2 + 2 * seq_len * d_model) * batch_size

    return flops


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def get_latest_checkpoint(output_dir):
    """
    Get the latest checkpoint from the output directory
    """
    checkpoint_dirs = sorted(
        glob.glob(os.path.join(output_dir, "step_*")), key=natural_key
    )
    if checkpoint_dirs:
        return checkpoint_dirs[-1]
    return None


def generate_run_id(text, length=8):
    hash_digest = hashlib.sha256(text.encode()).digest()
    base32 = base64.b32encode(hash_digest).decode("utf-8").lower().rstrip("=")
    return base32[:length]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a Masked Language Modeling task"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default="llm-jp-corpus",
        help="The directory containing the training data.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--validation_dir",
        type=str,
        default="llm-jp-corpus-validation",
        help="The directory containing the validation data.",
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."
        ),
    )
    parser.add_argument(
        "--line_by_line",
        type=bool,
        default=False,
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Ratio of tokens to mask for masked language modeling loss",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_token", type=str, help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="The entity to use when pushing to the Hub.",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default=None,
        help="The project name to use when pushing to the Hub.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="The experiment name to use when pushing to the Hub.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help="The number of steps before evaluating the model.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="The number of steps before logging the loss.",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=10000,
        help="The buffer size to use for shuffling.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-6,
        help="The epsilon to use in Adam.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 to use in Adam.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.98,
        help="The beta2 to use in Adam.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="The maximum gradient norm.",
    )

    args = parser.parse_args()

    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError(
                "Need an `output_dir` to create a repo when `--push_to_hub` is passed."
            )

    return args


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


metric = evaluate.load("accuracy")


def compute_metrics(logits, labels):
    preds = preprocess_logits_for_metrics(logits, labels)
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return metric.compute(predictions=preds, references=labels)


def main():
    args = parse_args()

    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    kwargs = [InitProcessGroupKwargs(timeout=timedelta(seconds=6000))]
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
        kwargs_handlers=kwargs,
    )

    # generate run id from exp_name like now45qh4
    assert args.exp_name is not None, "exp_name is required"
    run_id = generate_run_id(args.exp_name)
    accelerator.init_trackers(
        project_name=args.project_name,
        init_kwargs={
            "wandb": {
                "entity": args.entity,
                "name": args.exp_name,
                "id": run_id,
                "resume": "allow",
            }
        },
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(
                repo_name, exist_ok=True, token=args.hub_token
            ).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.dataset_name is not None:
        raw_datasets = load_dataset(
            args.dataset_name, args.dataset_config_name, streaming=True
        )
        raw_datasets["validation"] = load_dataset(
            "json",
            data_dir=args.validation_dir,
            split="train",
        )

    else:
        raw_datasets = load_dataset(
            "json",
            data_dir=args.train_dir,
            streaming=True,
        )

        raw_datasets["validation"] = load_dataset(
            "json",
            data_dir=args.validation_dir,
        )

    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name, trust_remote_code=args.trust_remote_code
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path, trust_remote_code=args.trust_remote_code
        )
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            use_fast=not args.use_slow_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=not args.use_slow_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(
            config, trust_remote_code=args.trust_remote_code
        )
    # torch._dynamo.config.optimize_ddp=False
    # model = torch.compile(model)
    num_layers = model.config.num_hidden_layers  # レイヤー数
    d_model = model.config.hidden_size  # 隠れ層の次元

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        print("Resize embedding")
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    text_column_name = "text"

    # keep only "text" column
    raw_datasets = raw_datasets.select_columns([text_column_name])

    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            max_seq_length = 1024
    else:
        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the "
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    if args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            # examples[text_column_name] = [
            #     line
            #     for line in examples[text_column_name]
            #     if len(line) > 0 and not line.isspace()
            # ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        with accelerator.main_process_first():
            # filter None
            raw_datasets = raw_datasets.filter(
                lambda x: x.get(text_column_name) is not None
            )
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                remove_columns=[text_column_name],
                batched=True,
            )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(
                examples[text_column_name], return_special_tokens_mask=True
            )

        with accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                remove_columns=[text_column_name],
                batched=True,
            )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {
                k: list(chain(*examples[k])) for k in examples.keys()
            }
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [
                    t[i : i + max_seq_length]
                    for i in range(0, total_length, max_seq_length)
                ]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/process#map

        with accelerator.main_process_first():
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )

    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=args.mlm_probability
    )

    # DataLoaders creation:
    # shuffle
    train_dataset = train_dataset.shuffle(seed=42, buffer_size=args.buffer_size)

    # distributed
    if os.environ.get("RANK") is not None:
        train_dataset = split_dataset_by_node(
            train_dataset,
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
        )

    train_dataloader = StatefulDataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.preprocessing_num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.preprocessing_num_workers,
        pin_memory=True,
    )
    if args.resume_from_checkpoint:
        checkpoint_path = None
        if args.resume_from_checkpoint == "latest":
            checkpoint_path = get_latest_checkpoint(args.output_dir)
        else:
            checkpoint_path = args.resume_from_checkpoint
        if checkpoint_path is not None:
            dataloader_state = torch.load(
                os.path.join(checkpoint_path, "dataloader_state.pth")
            )
            train_dataloader.load_state_dict(dataloader_state)
            logger.info(f"Loaded dataloader state from {checkpoint_path}")
            logger.info(f"train_dataset state: {train_dataloader.state_dict()}")
        else:
            logger.info(f"No checkpoint found in {args.resume_from_checkpoint}")
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon,
        betas=(args.adam_beta1, args.adam_beta2),
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    # if accelerator.distributed_type == DistributedType.TPU:
    #     model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config[
            "lr_scheduler_type"
        ].value
        accelerator.init_trackers("mlm_no_trainer", experiment_config)

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    grad_norm = None
    total_batches = 0
    total_loss = 0.0
    step_times = []  # 1ステップの時間を記録するリスト
    completed_steps = 0
    epoch = 0
    total_seen_tokens = 0
    total_seen_tokens_without_padding = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        checkpoint_path = None
        if args.resume_from_checkpoint == "latest":
            checkpoint_path = get_latest_checkpoint(args.output_dir)
        else:
            checkpoint_path = args.resume_from_checkpoint
        if checkpoint_path is not None:
            accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
            accelerator.load_state(checkpoint_path)
            training_difference = checkpoint_path.split("/")[-1]
            completed_steps = int(training_difference.replace("step_", ""))
            # load current epoch and total token seen
            with open(os.path.join(checkpoint_path, "custom_state.json")) as f:
                custom_state = json.load(f)
            epoch = custom_state["epoch"]
            total_seen_tokens = custom_state["total_seen_tokens"]
            total_seen_tokens_without_padding = custom_state[
                "total_seen_tokens_without_padding"
            ]
        else:
            accelerator.print(f"No checkpoint found in {args.resume_from_checkpoint}")

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    accelerator.print(f"Completed steps: {completed_steps}")

    model.train()
    active_dataloader = train_dataloader

    # Total FLOPs
    flops = (
        estimate_flops(
            batch_size=args.per_device_train_batch_size * accelerator.num_processes,
            seq_len=max_seq_length,
            num_layers=num_layers,
            d_model=d_model,
        )
        * args.gradient_accumulation_steps
    )

    # query_sentences, corpus_sentences = prepare_dataset()
    seen_tokens_per_device = 0
    seen_tokens_without_padding_per_device = 0
    while completed_steps < args.max_train_steps:
        for _, batch in enumerate(active_dataloader):
            start_time = time.perf_counter()  # ステップ開始時刻を記録
            with accelerator.accumulate(model):
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                total_loss += loss.detach()
                total_batches += 1
                seen_tokens_per_device += batch["input_ids"].numel()
                seen_tokens_without_padding_per_device += (
                    (batch["input_ids"] != tokenizer.pad_token_id).sum().item()
                )
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # ステップ時間計測
            step_time = time.perf_counter() - start_time
            step_times.append(step_time)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                avg_loss = accelerator.gather(total_loss).mean().item() / total_batches
                avg_grad_norm = (
                    accelerator.gather(grad_norm.clone().detach()).mean().item()
                )
                total_seen_tokens += (
                    accelerator.gather(
                        torch.tensor(seen_tokens_per_device, device=accelerator.device)
                    )
                    .sum()
                    .item()
                )
                total_seen_tokens_without_padding += (
                    accelerator.gather(
                        torch.tensor(
                            seen_tokens_without_padding_per_device,
                            device=accelerator.device,
                        )
                    )
                    .sum()
                    .item()
                )
                if completed_steps % args.logging_steps == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    avg_step_time = sum(step_times) / len(
                        step_times
                    )  # 平均ステップ時間
                    step_times = []  # 記録後にリセット
                    accelerator.log(
                        {
                            "train/loss": avg_loss,
                            "train/lr": lr,
                            "train/grad_norm": avg_grad_norm,
                            "train/time_per_step": avg_step_time,
                            "train/flops": flops,
                            "train/total_flops": flops * completed_steps,
                            "train/epoch": epoch,
                            "train/total_seen_tokens": total_seen_tokens,
                            "train/total_seen_tokens_without_padding": total_seen_tokens_without_padding,
                        },
                        step=completed_steps,
                    )
                    logger.info(
                        f"Step {completed_steps}, loss: {loss}, lr: {lr}, grad_norm: {grad_norm}, time_per_step: {avg_step_time}, flops: {flops}, total_flops: {flops * completed_steps}, epoch: {epoch}, total_seen_tokens: {total_seen_tokens}, total_seen_tokens_without_padding: {total_seen_tokens_without_padding}"
                    )
                total_batches = 0
                total_loss = 0.0
                seen_tokens_per_device = 0
                seen_tokens_without_padding_per_device = 0

                # validation
                if completed_steps % args.validation_steps == 0:
                    if accelerator.is_main_process:
                        model.eval()
                        val_loss = 0.0
                        val_acc = 0.0
                        batch_num = 0
                        for val_batch in val_dataloader:
                            batch_num += 1
                            val_batch = {
                                k: v.to(accelerator.device)
                                for k, v in val_batch.items()
                            }
                            with torch.no_grad():
                                outputs = model(**val_batch)
                                val_loss += outputs.loss
                                val_acc += compute_metrics(
                                    outputs.logits, val_batch["labels"]
                                )["accuracy"]

                        val_loss /= batch_num
                        val_acc /= batch_num
                        logger.info(
                            f"Validation loss: {val_loss}, Validation accuracy: {val_acc}"
                        )
                        accelerator.log(
                            {"val/loss": val_loss, "val/accuracy": val_acc},
                            step=completed_steps,
                        )
                        model.train()

            if isinstance(checkpointing_steps, int):
                if (
                    completed_steps % checkpointing_steps == 0
                    and accelerator.sync_gradients
                ):
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    # save dataloader state
                    state_dict = active_dataloader.state_dict()
                    logger.info(f"state_dict: {state_dict}")
                    torch.save(
                        state_dict, os.path.join(output_dir, "dataloader_state.pth")
                    )
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        output_dir,
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                    )
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(output_dir)
                        # save current epoch and total token seen
                        custom_state = {
                            "epoch": epoch,
                            "total_seen_tokens": total_seen_tokens,
                            "total_seen_tokens_without_padding": total_seen_tokens_without_padding,
                        }
                        with open(
                            os.path.join(output_dir, "custom_state.json"), "w"
                        ) as f:
                            json.dump(custom_state, f)
                        # evaluate retrieval
                        # recall, mrr = evaluate_retrieval(
                        #     output_dir, query_sentences, corpus_sentences
                        # )
                        # accelerator.log(
                        #     {"test/recall": recall, "test/mrr": mrr},
                        #     step=completed_steps,
                        # )
            if completed_steps >= args.max_train_steps:
                break

        epoch += 1

    model.eval()
    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        # accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )


if __name__ == "__main__":
    main()
