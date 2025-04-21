import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from matplotlib import pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
import os
from src.eval.alignment_and_uniformity import prepare_dataset


def compute_alignment(z1, z2):
    """
    z1, z2: (batch_size, hidden_dim) の正例ペア
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    return (z1 - z2).norm(p=2, dim=1) ** 2


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=2000)
    parser.add_argument(
        "--model_list",
        type=str,
        nargs="+",
        default=[
            # "cl-nagoya/ruri-large-v2",
            # "tohoku-nlp/bert-base-japanese-v3",
            # "sbintuitions/modernbert-ja-130m",
            "llm-jp/llm-jp-modernbert-base",
        ],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(42)
    plt.rcParams.update({"font.size": 20})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    positive_pairs, random_pairs = prepare_dataset(args.num_examples)

    result_dict = {}

    model_list = args.model_list
    model_list = [
        "speed/llm-jp-modernbert-base-v4-ja-stage1-0k",
        "speed/llm-jp-modernbert-base-v4-ja-stage1-4k",
        "speed/llm-jp-modernbert-base-v4-ja-stage1-15k",
        #     # "speed/llm-jp-modernbert-base-v4-ja-stage1-50k",
        "speed/llm-jp-modernbert-base-v4-ja-stage1-100k",
        #     "speed/llm-jp-modernbert-base-v4-ja-stage1-200k",
        #     "speed/llm-jp-modernbert-base-v4-ja-stage1-300k",
        "speed/llm-jp-modernbert-base-v4-ja-stage1-400k",
        #     "speed/llm-jp-modernbert-base-v4-ja-stage1-500k",
        "speed/llm-jp-modernbert-base-v4-ja-stage2-200k",
        "cl-nagoya/ruri-large-v2",
        "tohoku-nlp/bert-base-japanese-v3",
        "sbintuitions/modernbert-ja-130m",
    ]
    for model_id in model_list:
        print(f"Model: {model_id}")
        model = SentenceTransformer(model_id).to(device)
        z1 = model.encode([pair[0] for pair in random_pairs], convert_to_tensor=True)
        z2 = model.encode([pair[1] for pair in random_pairs], convert_to_tensor=True)
        alignment_score = compute_alignment(z1, z2)

        z1 = model.encode([pair[0] for pair in positive_pairs], convert_to_tensor=True)
        z2 = model.encode([pair[1] for pair in positive_pairs], convert_to_tensor=True)
        positive_alignment_score = compute_alignment(z1, z2)
        plt.figure(figsize=(8, 6))
        sns.histplot(alignment_score.cpu().numpy(), bins=50, kde=True, label="Random")
        sns.histplot(
            positive_alignment_score.cpu().numpy(), bins=50, kde=True, label="Positive"
        )
        plt.legend()
        # plt.title(model_id)
        plt.xlabel("Alignment score")
        plt.ylabel("Frequency")
        plt.ylim(0, 600)
        plt.xlim(0, 2)
        plt.tight_layout()
        if "speed" in model_id:
            model_id = model_id.replace("speed", "llm-jp")
        os.makedirs(os.path.join("results/sentence_sim_dist", model_id), exist_ok=True)
        plt.savefig(
            os.path.join("results/sentence_sim_dist", model_id, "distribution.png")
        )
        plt.close()
