from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train")
# keep only "text" column
ds = ds.select_columns(["text"])
ds = ds.train_test_split(test_size=0.1, seed=42)
print(ds)
train_ds = ds["train"].shuffle(seed=42).select(range(100000))
test_ds = ds["test"].shuffle(seed=42).select(range(1000))
train_ds.to_json("dataset/wiki_ja_nano/train/00000.json", force_ascii=False)
test_ds.to_json("dataset/wiki_ja_nano/test/00000.json", force_ascii=False)
