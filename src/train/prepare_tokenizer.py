from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument(
    "--tokenizer_name",
    type=str,
    default="llm-jp-modernbert-base",
    help="tokenizer name to use",
)

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("llm-jp/llm-jp-3-1.8b")

tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
    single=["<CLS|LLM-jp>", "$A:0", "<SEP|LLM-jp>"],
    pair=["<CLS|LLM-jp>", "$A:0", "<SEP|LLM-jp>", "$B:1", "<SEP|LLM-jp>:1"],
    special_tokens=[
        ("<CLS|LLM-jp>", tokenizer.backend_tokenizer.token_to_id("<CLS|LLM-jp>")),
        ("<SEP|LLM-jp>", tokenizer.backend_tokenizer.token_to_id("<SEP|LLM-jp>")),
    ],
)

# Remove specific configs for the original tokenizer
tokenizer.init_kwargs.pop("add_bos_token")
tokenizer.init_kwargs.pop("add_eos_token")


tokenizer.save_pretrained(args.tokenizer_name)
# fix tokenizer_config.json
with open(f"{args.tokenizer_name}/tokenizer_config.json", "r") as f:
    tokenizer_config = json.load(f)

tokenizer_config["model_input_names"] = ["input_ids", "attention_mask"]

with open(f"{args.tokenizer_name}/tokenizer_config.json", "w", encoding="utf-8") as f:
    out_str = (
        json.dumps(tokenizer_config, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    )
    f.write(out_str)


# Test
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
text = "こんにちは"
print(f"tokenizer({text}) -> {tokenizer(text)}")
print(
    f"tokenizer.decode({tokenizer(text)['input_ids']}) -> {tokenizer.decode(tokenizer(text)['input_ids'])}"
)
