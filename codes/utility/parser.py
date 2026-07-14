import argparse


def parse_args(flags=False):
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--data_path", nargs="?", default="data/", help="Input data path."
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument(
        "--dataset",
        nargs="?",
        default="MenClothing",
        help="Choose a dataset from {Toys_and_Games, Beauty, MenClothing, WomenClothing}",
    )
    parser.add_argument(
        "--verbose", type=int, default=5, help="Interval of evaluation."
    )
    parser.add_argument("--epoch", type=int, default=1000, help="Number of epoch.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size.")
    parser.add_argument(
        "--regs", nargs="?", default="[1e-03,1e-5]", help="Regularizations."
    )
    parser.add_argument("--lr", type=float, default=5e-05, help="Learning rate.")
    parser.add_argument("--embed_size", type=int, default=64, help="Embedding size.")
    parser.add_argument(
        "--feat_embed_dim", type=int, default=64, help="Feature embedding size."
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Coefficient of self node features."
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Coefficient of fine-grained interest matching.",
    )
    parser.add_argument(
        "--core",
        type=int,
        default=5,
        help="5-core for warm-start; 0-core for cold start.",
    )
    parser.add_argument(
        "--n_layers", type=int, default=1, help="Number of graph conv layers."
    )
    parser.add_argument("--use_cross_attention", default=True, action="store_false", help="Enable cross-attention mechanism.")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads in the cross-attention layer.")
    parser.add_argument("--attention_dim", type=int, default=256, help="Dimensionality of cross-attention projection."
    )
    parser.add_argument("--use_contrastive", default=False, action="store_true", help="Enable contrastive learning.")
    parser.add_argument("--contrastive_margin", type=float, default=0.5, help="Margin for contrastive loss.")
    parser.add_argument("--margin", type=float, default=0.5, help="Margin for contrastive loss in MONET model."
    )
    parser.add_argument("--temperature", type=float, default=0.25, help="Temperature for triplet loss."
    )
    
    parser.add_argument('--lambda_mmd', type=float, default=0.1, help='Weight for MMD loss'
    )

    parser.add_argument("--use_mlp", default=True, action="store_false", help="Enable MLP module for final prediction.")

    parser.add_argument("--has_norm", default=True, action="store_false")
    parser.add_argument("--target_aware", default=True, action="store_false")
    parser.add_argument(
        "--agg",
        type=str,
        default="concat",
        help="Choose a dataset from {sum, weighted_sum, concat, fc}",
    )
    parser.add_argument("--cf", default=False, action="store_true")
    parser.add_argument(
        "--cf_gcn",
        type=str,
        default="LightGCN",
        help="Choose a dataset from {MeGCN, LightGCN}",
    )
    parser.add_argument("--lightgcn", default=False, action="store_true")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id")
    parser.add_argument(
        "--Ks", nargs="?", default="[10, 20]", help="K value of ndcg/recall @ k"
    )
    parser.add_argument(
        "--test_flag",
        nargs="?",
        default="part",
        help="Specify the test type from {part, full}, indicating whether the reference is done in mini-batch",
    )

    if flags:
        attribute_dict = dict(vars(parser.parse_args()))
        print("*" * 32 + " Experiment setting " + "*" * 32)
        for k, v in attribute_dict.items():
            print(k + " : " + str(v))
        print("*" * 32 + " Experiment setting " + "*" * 32)
    return parser.parse_args()
