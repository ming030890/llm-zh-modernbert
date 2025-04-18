from transformers import ModernBertConfig, ModernBertForMaskedLM, AutoTokenizer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--tokenizer_name",
    type=str,
    default="llm-jp-modernbert-base",
    help="tokenizer name to use",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="llm-jp-modernbert-base",
    help="model name that will be saved",
)

if __name__ == "__main__":
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    print("vocab size:", tokenizer.vocab_size)
    # num of special tokens
    print("num of special tokens:", tokenizer.num_special_tokens_to_add())
    # special tokens
    print("special tokens:", tokenizer.all_special_tokens)

    config = ModernBertConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=8192,
        num_attention_heads=12,
        num_hidden_layers=22,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        global_rope_theta=10000.0,
        reference_compile=True,
    )

    model = ModernBertForMaskedLM(config)

    print("model parameters", model.num_parameters())

    # save model and tokenizer
    if "/" in args.model_name:
        model_name = args.model_name.split("/")[-1]
        model.save_pretrained(model_name)
        tokenizer.save_pretrained(model_name)
    else:
        model.save_pretrained(args.model_name)
        tokenizer.save_pretrained(args.model_name)
