import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from matplotlib import pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
import json
from adjustText import adjust_text
import os


def compute_alignment(z1, z2):
    """
    z1, z2: (batch_size, hidden_dim) の正例ペア
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    return ((z1 - z2).norm(p=2, dim=1) ** 2).mean().item()


def compute_uniformity(embeddings, t=2.0):
    embeddings = F.normalize(embeddings, dim=1)  # (1) L2正規化
    sq_norm = (embeddings**2).sum(dim=1, keepdim=True)  # (2) 各ベクトルの2乗ノルム
    dist_squared = (
        sq_norm + sq_norm.T - 2 * torch.matmul(embeddings, embeddings.T)
    )  # (3) 距離二乗計算
    mask = ~torch.eye(
        embeddings.size(0), dtype=torch.bool, device=embeddings.device
    )  # (4) 対角成分マスク
    exp_term = torch.exp(-t * dist_squared[mask])  # (5) e^{-t * dist^2}
    return torch.log(exp_term.mean()).item()  # (6) log(平均値)


def prepare_dataset(num_examples: int = 2000):
    miracl_ds = load_dataset(
        "miracl/miracl", "ja", trust_remote_code=True, split="train"
    )
    wiki_ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train").shuffle(
        seed=42
    )
    positive_pairs = []
    for data in miracl_ds:
        positive_passages = data["positive_passages"]
        positive_sentences = [entry["text"] for entry in positive_passages]
        if len(positive_sentences) < 2:
            continue
        positive_pairs.append((positive_sentences[0], positive_sentences[1]))

    positive_pairs = positive_pairs[:num_examples]
    corpus_sentences = []
    for i, example in enumerate(wiki_ds):
        if i >= num_examples * 2:
            break
        corpus_sentences.append(example["text"])
    random_pairs = []
    for i in range(num_examples):
        random_pairs.append((corpus_sentences[i], corpus_sentences[num_examples + i]))
    return positive_pairs, random_pairs


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument(
        "--model_list",
        type=str,
        nargs="+",
        default=[
            "cl-nagoya/ruri-large-v2",
            "tohoku-nlp/bert-base-japanese-v3",
            "sbintuitions/modernbert-ja-130m",
            "llm-jp/llm-jp-modernbert-base",
        ],
    )
    parser.add_argument("--num_examples", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(42)
    plt.rcParams.update({"font.size": 20})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    positive_pairs, random_pairs = prepare_dataset(args.num_examples)

    result_dict = {}
    model_list = args.model_list
    # model_list = [
    #     "speed/llm-jp-modernbert-base-v4-ja-stage1-0k",
    #     "speed/llm-jp-modernbert-base-v4-ja-stage1-4k",
    #     "speed/llm-jp-modernbert-base-v4-ja-stage1-15k",
    #     # "speed/llm-jp-modernbert-base-v4-ja-stage1-50k",
    #     "speed/llm-jp-modernbert-base-v4-ja-stage1-100k",
    #     # "speed/llm-jp-modernbert-base-v4-ja-stage1-200k",
    #     "speed/llm-jp-modernbert-base-v4-ja-stage1-300k",
    #     # "speed/llm-jp-modernbert-base-v4-ja-stage1-400k",
    #     "speed/llm-jp-modernbert-base-v4-ja-stage1-500k",
    #     "speed/llm-jp-modernbert-base-v4-ja-stage2-200k",
    # ]

    for model_id in model_list:
        print(f"Model: {model_id}")
        model = SentenceTransformer(model_id).to(device)
        z1 = model.encode([pair[0] for pair in positive_pairs], convert_to_tensor=True)
        z2 = model.encode([pair[1] for pair in positive_pairs], convert_to_tensor=True)

        alignment_score = compute_alignment(z1, z2)

        z1 = model.encode([pair[0] for pair in random_pairs], convert_to_tensor=True)
        z2 = model.encode([pair[1] for pair in random_pairs], convert_to_tensor=True)
        uniformity_score = compute_uniformity(torch.cat([z1, z2], dim=0))

        print(f"Alignment:  {alignment_score:.4f}")
        print(f"Uniformity: {uniformity_score:.4f}")

        result_dict[model_id] = {
            "alignment": alignment_score,
            "uniformity": uniformity_score,
        }

    # Plot Alignment vs. Uniformity

    data = []
    for model_id, scores in result_dict.items():
        miracl_result_path = os.path.join(
            args.output_dir, "miracl", model_id, "results.json"
        )
        if not os.path.exists(miracl_result_path):
            recall = 0
        else:
            with open(miracl_result_path, "r") as f:
                data_json = json.load(f)
                # {"Recall@10": 0.6723940435280642, "MRR@10": 0.47000740922562306}
            recall = data_json["Recall@10"]
        data.append((model_id, scores["alignment"], scores["uniformity"], recall))

    # 散布図データ抽出
    labels, aligns, uniforms, scores = zip(*data)

    # プロット設定
    plt.figure(figsize=(10, 8))
    norm = plt.Normalize(min(scores), max(scores))
    cmap = sns.color_palette("magma", as_cmap=True)
    sc = plt.scatter(
        uniforms, aligns, c=scores, cmap=cmap, norm=norm, s=100, edgecolors="k"
    )

    texts = []

    for label, x, y in zip(labels, uniforms, aligns):
        label = label.split("/")[-1]
        label = "-".join(label.split("-")[-2:])
        texts.append(plt.text(x, y, label))

    adjust_text(texts, arrowprops=dict(arrowstyle="->", color="r", lw=0.5))

    # 軸ラベル
    plt.xlabel(r"$\ell_{uniform}$")
    plt.ylabel(r"$\ell_{align}$")

    # カラーバー（スコア用）(上部に配置)
    cbar = plt.colorbar(sc, orientation="horizontal", pad=0.1)

    cbar.set_label("Recall@10 on miracl")
    plt.tight_layout()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig("Alignment_vs_Uniformity.png")
