from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import os
import math
from tqdm import tqdm
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import random


def compute_pseudo_perplexity(
    input_id, model, tokenizer, mask_num: int, max_mask_batch_size: int = 5
) -> float:
    batch_size, seq_len = input_id.shape
    input_id = input_id.to(model.device)

    # Generate mask positions for each batch element
    mask_positions = torch.randint(
        1, seq_len - 1, (batch_size, mask_num), device=input_id.device
    )

    # Initialize the loss list
    loss_list = []

    # Process in smaller batches to avoid OOM
    for mask_batch_start in range(0, mask_num, max_mask_batch_size):
        mask_batch_end = min(mask_batch_start + max_mask_batch_size, mask_num)
        mask_batch_size = mask_batch_end - mask_batch_start

        # Expand input_id for masking
        masked_inputs = input_id.unsqueeze(1).repeat(
            1, mask_batch_size, 1
        )  # Shape: [batch_size, mask_batch_size, seq_len]
        labels = input_id.unsqueeze(1).repeat(
            1, mask_batch_size, 1
        )  # Copy for loss computation

        # Apply mask at specified positions
        for i in range(batch_size):
            masked_inputs[
                i,
                torch.arange(mask_batch_size),
                mask_positions[i, mask_batch_start:mask_batch_end],
            ] = tokenizer.mask_token_id

        # Reshape for model input
        masked_inputs = masked_inputs.view(
            batch_size * mask_batch_size, seq_len
        )  # Flatten batch
        labels = labels.view(batch_size * mask_batch_size, seq_len)

        # Forward pass
        with torch.no_grad():
            outputs = model(masked_inputs)

        # Extract logits at masked positions
        logits = outputs.logits.view(
            batch_size, mask_batch_size, seq_len, -1
        )  # Reshape back
        selected_logits = logits[
            torch.arange(batch_size).unsqueeze(1),
            torch.arange(mask_batch_size),
            mask_positions[:, mask_batch_start:mask_batch_end],
        ]

        # Extract correct target labels
        target_labels = labels.view(batch_size, mask_batch_size, seq_len)[
            torch.arange(batch_size).unsqueeze(1),
            torch.arange(mask_batch_size),
            mask_positions[:, mask_batch_start:mask_batch_end],
        ]

        # Compute loss for this batch
        loss = torch.nn.functional.cross_entropy(
            selected_logits.view(-1, selected_logits.shape[-1]),
            target_labels.view(-1),
            reduction="mean",
        )
        loss_list.append(loss.item())

    # Compute perplexity as the exponential of the average loss
    perplexity = math.exp(sum(loss_list) / len(loss_list))
    print(f"Loss list: {loss_list}")
    print(f"Perplexity: {perplexity}")

    return perplexity


def prepare_dataset(ds, tokenizer, max_seq_length, num_examples: int = 2000):
    bin_1 = []  # [0, 1024]
    bin_2 = []  # [1024, 2048]
    bin_3 = []  # [2048, 4096]
    bin_4 = []  # [4096, 8192]
    quarter_num = num_examples // 4

    for i, example in enumerate(ds):
        if (
            len(bin_1) >= quarter_num
            and len(bin_2) >= quarter_num
            and len(bin_3) >= quarter_num
            and len(bin_4) >= quarter_num
        ):
            break
        print(
            f"bin_1: {len(bin_1)}, bin_2: {len(bin_2)}, "
            f"bin_3: {len(bin_3)}, bin_4: {len(bin_4)}"
        )

        input_ids = tokenizer(
            example["text"],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_seq_length,
        )["input_ids"]

        seq_len = input_ids.shape[1]

        if seq_len <= 1024 and len(bin_1) < quarter_num:
            bin_1.append(input_ids[:, :max_seq_length])
        elif 1024 < seq_len <= 2048 and len(bin_2) < quarter_num:
            bin_2.append(input_ids[:, :max_seq_length])
        elif 2048 < seq_len <= 4096 and len(bin_3) < quarter_num:
            bin_3.append(input_ids[:, :max_seq_length])
        elif 4096 < seq_len < 8192 and len(bin_4) < quarter_num:
            bin_4.append(input_ids[:, :max_seq_length])

    sequences = bin_1 + bin_2 + bin_3 + bin_4
    # shuffle to avoid minibatch bias (for efficiency)

    random.shuffle(sequences)
    print(
        f"bin_1: {len(bin_1)}, bin_2: {len(bin_2)}, "
        f"bin_3: {len(bin_3)}, bin_4: {len(bin_4)}"
    )
    return sequences


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="llm-jp/llm-jp-modernbert-base")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--num_examples", type=int, default=2000)
    parser.add_argument("--mask_num", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(42)
    plt.rcParams.update({"font.size": 20})

    model = AutoModelForMaskedLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()

    # Load dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train")
    ds = ds.shuffle(seed=42)

    long_sequences = prepare_dataset(
        ds, tokenizer, args.max_seq_length, args.num_examples
    )

    # plot sequence length distribution
    plt.figure(figsize=(10, 8))
    plt.hist([seq.shape[1] for seq in long_sequences], bins=50)
    plt.xlabel("Sequence length")
    plt.ylabel("Count")
    plt.title("Sequence length distribution")
    plt.grid()
    plt.savefig("sequence_length_distribution.png")

    # Compute pseudo-perplexity for each sequence
    pseudo_perplexities = []
    for seq in tqdm(long_sequences):
        pseudo_perplexities.append(
            compute_pseudo_perplexity(seq, model, tokenizer, args.mask_num)
        )

    # shuffle

    # plot sequence length vs pseudo-perplexity

    output_dir = os.path.join(args.output_dir, "pseudo_perplexity", args.model)
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.scatter(
        [seq.shape[1] for seq in long_sequences],
        pseudo_perplexities,
        alpha=0.5,
    )
    # plt.legend()
    # vertical and horizontal lines
    plt.xlabel("Sequence length")
    plt.ylabel("Pseudo-perplexity")
    plt.ylim(0, 20)
    # plt.title(args.model.replace("speed/", ""))
    plt.grid()
    #
    plt.savefig(
        os.path.join(output_dir, "pseudo_perplexity.png"),
    )
