import argparse
import sys

datasets = ["pawsx", "xnli", "siqa", "copa", "siqa_nd_copa", "marc"]
stochasticities = ["DiffInit", "DiffOrder", "DiffInitOrder", "DiffData", "AllDiff"]


def build_parser(args=[]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--stochasticity", default="DiffInit", choices=stochasticities
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="pawsx",
        choices=datasets,
        help="Which dataset to run experiments on",
    )
    parser.add_argument(
        "-m",
        "--mmlm",
        default="bert-base-multilingual-uncased",
        help="Name or path of the pretrained-mmlm to fine-tune",
    )
    parser.add_argument(
        "-n",
        "--n_models",
        type=int,
        default=2,
        help="Number of ensemble models to train, Only relevant while running main_multiple.py",
    )
    parser.add_argument(
        "--train_lang", default="en", type=str, help="Language to use for training"
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=-1,
        help="Maximum number of training examples to use. Default = -1, which means the entire dataset will be used",
    )
    parser.add_argument(
        "--load_train_en_only", dest="load_train_en_only", action="store_true"
    )
    parser.add_argument(
        "--no-load_train_en_only", dest="load_train_en_only", action="store_false"
    )
    parser.set_defaults(load_train_en_only=True)
    parser.add_argument(
        "--data_dir", default="data/", help="Path where data is located",
    )
    parser.add_argument(
        "--save_dir", default="out/", help="Path where to store the results",
    )
    parser.add_argument(
        "--plot_dir", default="plots/", help="Path where to store the results",
    )
    parser.add_argument(
        "--device", default="cuda", help="Device on which to run experiments"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Whether to run the code in debug mode"
    )
    parser.add_argument(
        "--cal_bins",
        default=10,
        type=int,
        help="Number of bins to use while calculating callibration scores",
    )
    parser.add_argument(
        "--cal_lang",
        default="en",
        type=str,
        help="Language Dev Data to use to calibrate",
    )
    parser.add_argument(
        "--cal_size", default=-1, type=int, help="Amount of data to use for scaling"
    )
    parser.add_argument(
        "--temp_scaling",
        dest="temp_scaling",
        action="store_true",
        help="Whether to use temperature scaling for calibrating the model",
    )
    parser.add_argument(
        "--label_smoothing",
        dest="label_smoothing",
        action="store_true",
        help="Whether to use label smoothing for calibrating the model",
    )
    parser.add_argument(
        "--alpha_smoothing", type=float, default=0.0, help="Label Smoothing Parameter"
    )
    parser.add_argument(
        "--adv_tgt_lang",
        type=str,
        default="hi",
        help="Language(s) on which to calibrate the model",
    )
    parser.add_argument("--adv_lambda", type=float, default=1)

    # Few Shot Learning Parameters
    parser.add_argument(
        "--few_shot_learning",
        action="store_true",
        help="Whether to perform few shot learning on target language",
    )
    parser.add_argument(
        "--train_config",
        default="",
        type=str,
        help="Training config specifying the amount of data to be used in each language for training. Must follow the format lang1:size1,...,langn:sizen",
    )
    parser.add_argument(
        "--few_shot_lang",
        default="hi",
        type=str,
        help="Langauge to use for few-shot learning. Only relevant when train_config=''",
    )
    parser.add_argument(
        "--few_shot_size",
        default=1000,
        type=int,
        help="Amount of data to use for few-shot",
    )
    parser.add_argument(
        "--curiculum_learning",
        action="store_true",
        help="Whether to train using curiculum learning",
    )
    parser.add_argument("--lr_curr", type=float, default=3e-5)
    parser.add_argument("--num_epochs_curr", type=int, default=1)
    parser.add_argument("--batch_size_curr", type=int, default=8)

    parser.add_argument("--use_dev_for_train_for_non_en", action="store_true")

    # Training Hyper-parameters
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--clip_grad_norm", type=float, default=0)
    parser.add_argument("--re_train", action="store_true")
    # For logging purposes
    parser.add_argument("--train_mode", default="", type=str)

    # SIQA/COPA paramerers
    parser.add_argument("--pretrained_model", default="", type=str)
    parser.add_argument("--merge_dev_sets", action="store_true")
    config = parser.parse_args(args)

    return config
