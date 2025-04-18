import json
import glob
import pandas as pd
from argparse import ArgumentParser


def extract_epoch_and_lr(file):
    file = file.split("/")[-1]
    epoch = file.split("_")[2].split("e")[-1]
    lr = file.split("_")[3].split("lr")[-1].replace(".json", "")
    return int(epoch), float(lr)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output_format", type=str, default="markdown")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_list = [
        "tohoku-nlp/bert-base-japanese-v3",
        "sbintuitions/modernbert-ja-130m",
        "sbintuitions/modernbert-ja-310m",
        "speed/llm-jp-modernbert-base-stage1",
        "speed/llm-jp-modernbert-base-stage2",
        "speed/llm-jp-modernbert-base-v4-ja-stage1-4k",
        "speed/llm-jp-modernbert-base-v4-ja-stage1-15k",
        "speed/llm-jp-modernbert-base-v4-ja-stage1-50k",
        "speed/llm-jp-modernbert-base-v4-ja-stage1-100k",
        "speed/llm-jp-modernbert-base-v4-ja-stage1-200k",
        "speed/llm-jp-modernbert-base-v4-ja-stage1-300k",
        "speed/llm-jp-modernbert-base-v4-ja-stage1-400k",
        "speed/llm-jp-modernbert-base-v4-ja-stage1-500k",
        "speed/llm-jp-modernbert-base-v4-ja-stage2-200k",
    ]

    metric_mapping = {
        "JSTS": ["eval_pearson"],
        "JNLI": ["eval_accuracy"],
        "JCoLA": ["eval_accuracy"],
        "miracl": ["Recall@10"],
    }

    task_list = list(metric_mapping.keys())

    data = []

    for model in model_list:
        row = {"Model": model}
        for task in task_list:
            metrics = metric_mapping[task]
            path = f"results/{task}/{model}"
            best_metric = "N/A"

            try:
                if task == "miracl":
                    files = glob.glob(path + "/results.json")
                else:
                    files = glob.glob(path + "/all_results_e*_lr*.json")

                for file in files:
                    with open(file, "r") as f:
                        data_json = json.load(f)
                        primary_metric = metrics[0]
                        current_value = data_json.get(primary_metric, 0)

                        if best_metric == "N/A" or current_value > best_metric:
                            best_metric = (
                                round(current_value, 4)
                                if isinstance(current_value, (int, float))
                                else "N/A"
                            )

                row[task] = best_metric
            except Exception:
                row[task] = "N/A"

        scores_jglue = [row.get(task, "N/A") for task in ["JSTS", "JNLI", "JCoLA"]]
        numeric_scores_jglue = [
            score for score in scores_jglue if isinstance(score, (int, float))
        ]
        row["Avg(JGLUE)"] = (
            round(sum(numeric_scores_jglue) / len(numeric_scores_jglue), 4)
            if numeric_scores_jglue
            else "N/A"
        )

        scores_all = scores_jglue + [row.get("miracl", "N/A")]
        numeric_scores_all = [
            score for score in scores_all if isinstance(score, (int, float))
        ]
        row["Avg"] = (
            round(sum(numeric_scores_all) / len(numeric_scores_all), 4)
            if numeric_scores_all
            else "N/A"
        )

        data.append(row)

    df = pd.DataFrame(
        data,
        columns=["Model", "JSTS", "JNLI", "JCoLA", "Avg(JGLUE)"],  # , "miracl", "Avg"]
    )
    # rename speed to llm-jp
    df["Model"] = df["Model"].str.replace("speed", "llm-jp")
    # 100 point
    df["JSTS"] = df["JSTS"] * 100
    df["JNLI"] = df["JNLI"] * 100
    df["JCoLA"] = df["JCoLA"] * 100
    df["Avg(JGLUE)"] = df["Avg(JGLUE)"] * 100

    print(df.to_markdown(index=False, tablefmt="github", floatfmt=".1f"))
    with open("leaderboard.md", "w") as f:
        if args.output_format == "markdown":
            f.write(df.to_markdown(index=False, tablefmt="github", floatfmt=".1f"))
        else:
            f.write(
                df.to_latex(
                    index=False,
                    float_format="%.1f",
                    column_format="l" + "c" * len(df.columns),
                )
            )
