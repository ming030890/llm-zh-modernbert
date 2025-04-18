from datasets import load_dataset
import torch
from sentence_transformers import SentenceTransformer
from argparse import ArgumentParser
import os
import json


def prepare_dataset():
    ds = load_dataset("miracl/miracl", "ja", trust_remote_code=True, split="train")
    query_sentences = []
    corpus_sentences = []
    for data in ds:
        positive_passages = data["positive_passages"]
        positive_sentences = [entry["text"] for entry in positive_passages]
        if len(positive_sentences) < 2:
            continue
        query_sentences.append(positive_sentences[0])
        corpus_sentences.append(positive_sentences[1])
    return query_sentences, corpus_sentences


def evaluate_retrieval(model_name: str, query_sentences, corpus_sentences, k=10):
    model = SentenceTransformer(model_name)
    corpus_embeddings = model.encode(corpus_sentences, convert_to_tensor=True)
    query_embeddings = model.encode(query_sentences, convert_to_tensor=True)
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    corpus_embeddings = torch.nn.functional.normalize(corpus_embeddings, p=2, dim=1)
    similarity_scores = torch.mm(query_embeddings, corpus_embeddings.t())
    recall = 0
    mrr = 0
    for i, query in enumerate(query_sentences):
        topk = torch.topk(similarity_scores[i], k=k).indices.tolist()
        if i in topk:
            recall += 1
        for rank, idx in enumerate(topk, start=1):
            if idx == i:
                mrr += 1 / rank
                break

    recall /= len(query_sentences)
    mrr /= len(query_sentences)
    return recall, mrr


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="llm-jp/llm-jp-modernbert-base")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    query_sentences, corpus_sentences = prepare_dataset()

    recall, mrr = evaluate_retrieval(
        args.model, query_sentences, corpus_sentences, k=args.k
    )
    print(f"Model: {args.model}")
    print(f"Recall@{args.k}: {recall:.3f}")
    print(f"MRR@{args.k}: {mrr:.3f}")
    output_dir = os.path.join(args.output_dir, "miracl", args.model)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json_data = {"Recall@10": recall, "MRR@10": mrr}
        json.dump(json_data, f)
