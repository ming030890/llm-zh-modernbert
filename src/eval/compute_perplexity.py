from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import math
import random
from argparse import ArgumentParser


def calculate_perplexity(text, model, tokenizer, max_length=512, verbose=False):
    random.seed(0)
    torch.manual_seed(0)
    # トークン化
    tokens = tokenizer(
        text, return_tensors="pt", max_length=max_length, truncation=True
    )
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    # マスクされたトークンを作成
    masked_indices = torch.randint(1, input_ids.size(1) - 1, (1,))
    labels = input_ids.clone()
    ground_truth = input_ids[0, masked_indices].item()
    labels[:, masked_indices] = tokenizer.mask_token_id  # マスクトークンに置き換え

    # モデルの出力
    with torch.no_grad():
        outputs = model(labels, attention_mask=attention_mask, labels=input_ids)

    # filled mask token
    filled = tokenizer.decode(torch.argmax(outputs.logits[0, masked_indices]).item())
    topk = 5
    topk_filled = [
        tokenizer.decode(i)
        for i in torch.topk(outputs.logits[0, masked_indices], topk).indices[0].tolist()
    ]

    # 損失を取得
    loss = outputs.loss.item()

    # パープレキシティを計算
    perplexity = math.exp(loss)
    accuracy = (
        1
        if input_ids[0, masked_indices].item()
        == torch.argmax(outputs.logits[0, masked_indices]).item()
        else 0
    )
    if accuracy == 0 and verbose:
        print("labeled text:", tokenizer.decode(labels[0].tolist()))
        print(f"Ground truth: {tokenizer.decode([ground_truth])}")
        print(f"Filled: {filled}")
        print(f"Top-{topk} filled: {topk_filled}")

    return perplexity, accuracy


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--dataset", type=str, default="wikipedia")
    parser.add_argument("--num_texts", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 日本語BERTモデルの読み込み
    model_list = [
        "tohoku-nlp/bert-base-japanese-v3",
        "sbintuitions/modernbert-ja-130m",
        "speed/llm-jp-modernbert-base-stage1",
        "speed/llm-jp-modernbert-base-v4-ja-stage1-500k",
        "speed/llm-jp-modernbert-base-v3-ja-en-stage1-500k",
    ]
    # Wikipedia日本語データセットの読み込み
    # match statement
    match args.dataset:
        case "wikipedia":
            ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train")
        case "tinystory":
            ds = load_dataset("kai271/TinyStories-Japanese", split="train")
        case "arxiv":
            ds = load_dataset("speed/arxiver_ja", split="train")
            ds = ds.rename_column("abstract_ja", "text")

    ds = ds.filter(lambda x: len(x["text"]) > 5)
    texts = ds["text"]

    texts = texts[: args.num_texts]

    def calc_score_model(model_name, verbose=False):
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        ppls = []
        accuracies = []
        for text in tqdm(texts):
            ppl, accuracy = calculate_perplexity(
                text, model, tokenizer, max_length=args.max_length, verbose=verbose
            )
            ppls.append(ppl)
            accuracies.append(accuracy)

        ppl = sum(ppls) / len(ppls)
        accuracy = sum(accuracies) / len(accuracies)
        return ppl, accuracy

    for model_name in model_list:
        ppl, accuracy = calc_score_model(model_name, verbose=args.verbose)
        print(f"Model: {model_name}")
        print(f"Perplexity: {ppl}")
        print(f"Accuracy: {accuracy}")
        print()
