import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor
import pandas as pd


def cosine_similarity(e1: Tensor, e2: Tensor) -> Tensor:
    e1_norm = torch.nn.functional.normalize(e1, p=2, dim=1)
    e2_norm = torch.nn.functional.normalize(e2, p=2, dim=1)
    return torch.mm(e1_norm, e2_norm.T)  # 行列積で一括計算


dataset = [
    {
        "query": "今日は朝から天気が良く、久しぶりに公園を散歩することにしました。澄み切った青空の下、子供たちが元気に遊んでいる様子を見ながらベンチに座って読書をしました。風が心地よく、春の訪れを感じました。",
        "related": [
            "晴れた日に公園を歩くと、気持ちがリフレッシュできます。特に春や秋は、空気が澄んでいて散歩には最適な季節です。",
            "公園にはランニングコースがあり、多くの人が健康のために走っています。ジョギングをしながら、季節の移り変わりを感じることができます。",
            "公園には多くの種類の花が植えられており、四季折々の風景を楽しむことができます。",
        ],
    },
    {
        "query": "東京から大阪まで新幹線を利用しました。移動時間が短縮されるだけでなく、車内で仕事をしたり、のんびりと景色を眺めたりすることもできるので、飛行機よりも快適に感じました。",
        "related": [
            "新幹線は時間に正確で、長距離移動を短時間で行うことができるため、多くのビジネスマンに利用されています。",
            "日本の新幹線は世界的に見ても正確な運行スケジュールを持ち、高い評価を受けています。車内の座席も広く、快適な移動手段として人気です。",
            "東京と大阪を結ぶ新幹線は、特にビジネス用途で利用されることが多く、早朝や夕方は混雑することがよくあります。",
            "新幹線のグリーン車は、広々とした座席と静かな環境が特徴で、移動中に仕事をするのに適しています.",
        ],
    },
    {
        "query": "最近、機械学習の勉強を本格的に始めましたが、特にニューラルネットワークの仕組みがとても奥深く、数学的な知識が求められることを痛感しました。微分や線形代数の復習をしながら、モデルの学習プロセスを理解しようとしています。",
        "related": [
            "機械学習の基礎を学ぶには、線形代数や微分積分の理解が不可欠です。数学的な背景があると、ニューラルネットワークの仕組みを深く理解しやすくなります。",
            "機械学習アルゴリズムの最適化には、大量の計算資源が必要になります。特にディープラーニングのトレーニングには、高性能なGPUが求められます。",
            "数学の知識があると、機械学習モデルの構造をより深く理解することができ、適切なハイパーパラメータの設定も容易になります.",
        ],
    },
    {
        "query": "京都に旅行し、歴史的な寺社仏閣を巡りながら伝統的な町並みを楽しみました。特に清水寺や金閣寺は圧巻で、写真を撮るのが楽しかったです。抹茶スイーツを食べながら、ゆったりとした時間を過ごしました。",
        "related": [
            "京都は歴史と伝統が色濃く残る街で、多くの観光客が訪れます。特に紅葉の季節には、神社仏閣が美しく彩られます。",
            "京都には美味しい和菓子を楽しめるお店がたくさんあります。特に抹茶スイーツは観光客にも人気です。",
            "多くの観光客が京都を訪れる理由の一つは、美しい街並みと歴史的な建造物にあります。特に秋の紅葉シーズンは、どこを歩いても風情があります。",
            "神社やお寺は日本の伝統文化を象徴する建物であり、歴史的価値が高いです。",
        ],
    },
    {
        "query": "昨日の夜ご飯はカレーライスでした。家庭的な味がして、とても美味しかったのでまた作ってみようと思います。",
        "related": [
            "カレーライスは、日本の家庭料理として親しまれています。野菜や肉をカレー粉で煮込んだもので、ご飯にかけて食べると美味しいです。",
            "カレーライスの具材は様々で、野菜や肉、魚などを使ってバリエーション豊かに楽しむことができます。",
            "カレーライスは、スパイスの効いたカレーソースをご飯にかけた料理で、日本人にとって親しみやすい味わいです。",
        ],
    },
]

query_sentences = [data["query"] for data in dataset]
corpus_sentences = [related for data in dataset for related in data["related"]]
ground_truth = {
    i: [corpus_sentences.index(related) for related in data["related"]]
    for i, data in enumerate(dataset)
}

model_list = [
    "cl-nagoya/ruri-large-v2",
    "tohoku-nlp/bert-base-japanese-v3",
    "sbintuitions/modernbert-ja-130m",
    "speed/llm-jp-modernbert-base-stage1",
    "speed/llm-jp-modernbert-base-v4-ja-stage1-500k",
    "speed/llm-jp-modernbert-base-v3-ja-en-stage1-500k",
]

top_k = 4  # 上位何件取得するか

retrieves_per_model = {}

for model_name in model_list:
    print(f"\n=== Model: {model_name} ===")
    model = SentenceTransformer(model_name)

    # すべての文をエンコード
    query_embeddings = model.encode(query_sentences, convert_to_tensor=True)
    corpus_embeddings = model.encode(corpus_sentences, convert_to_tensor=True)

    # 類似度を計算
    similarity_matrix = cosine_similarity(query_embeddings, corpus_embeddings)

    # 各クエリに対し最も近いK個を取得
    for i, query in enumerate(query_sentences):
        top_indices = similarity_matrix[i].topk(top_k).indices
        top_sentences = [corpus_sentences[j] for j in top_indices]
        for rank, (sent, score) in enumerate(
            zip(top_sentences, similarity_matrix[i][top_indices]), 1
        ):
            retrieves_per_model.setdefault(model_name, []).append(
                (query, sent, score.item(), rank)
            )


top_k = 4  # 上位何件取得するか

retrieves_per_model = {}

for model_name in model_list:
    print(f"\n=== Model: {model_name} ===")
    model = SentenceTransformer(model_name)

    # すべての文をエンコード
    query_embeddings = model.encode(query_sentences, convert_to_tensor=True)
    corpus_embeddings = model.encode(corpus_sentences, convert_to_tensor=True)

    # 類似度を計算
    similarity_matrix = cosine_similarity(query_embeddings, corpus_embeddings)

    # 各クエリに対し最も近いK個を取得
    for i, query in enumerate(query_sentences):
        top_indices = similarity_matrix[i].topk(top_k).indices
        top_sentences = [corpus_sentences[j] for j in top_indices]
        for rank, (sent, score) in enumerate(
            zip(top_sentences, similarity_matrix[i][top_indices]), 1
        ):
            retrieves_per_model.setdefault(model_name, []).append(
                (query, rank, score.item(), sent)
            )

# **表形式で表示**
df_dict = {}

for model_name, retrieves in retrieves_per_model.items():
    df = pd.DataFrame(
        retrieves, columns=["Query", "Rank", "Score", "Retrieved Sentence"]
    )
    df_dict[model_name] = df

# **Excelに保存**
excel_path = "retrieval_results.xlsx"
with pd.ExcelWriter(excel_path) as writer:
    for model_name, df in df_dict.items():
        df.to_excel(writer, sheet_name=model_name.replace("/", "_"), index=False)

print(f"結果を {excel_path} に保存しました。")
