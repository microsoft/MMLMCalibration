import logging
import os
import random
import sys

import numpy as np
import torch
import wandb
from src.config import build_parser
from src.dataset import init_sentence_pair_dataset
from src.model import TempScaledSentencePairClassifier
from src.performance_nd_calibration import get_acc_nd_cal_errors
from src.trainer import evaluate, train_fancy
from src.utils.helper import create_dirs, load_model, save_model
from src.utils.load_data import load_data
from torch.utils.data import DataLoader

logging.basicConfig(format="%(asctime)s %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    config = build_parser(sys.argv[1:])

    # Set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    # Create Directories
    if config.train_lang == "en":
        run_id = f"{config.dataset}_{config.mmlm}_MaxLength{config.max_length}_BatchSize{config.batch_size}_LR{config.lr}_EPOCHS{config.num_epochs}_SEED{config.seed}_TrainSize{config.max_train_samples}_AlphaSmoothing{str(config.alpha_smoothing).replace('.','')}_debug{config.debug}"
    else:
        run_id = f"{config.dataset}_{config.mmlm}_TrainLang{config.train_lang}_MaxLength{config.max_length}_BatchSize{config.batch_size}_LR{config.lr}_EPOCHS{config.num_epochs}_SEED{config.seed}_TrainSize{config.max_train_samples}_AlphaSmoothing{str(config.alpha_smoothing).replace('.','')}_debug{config.debug}"
    ckpt_dir = f"{config.save_dir}/checkpoints/{run_id}"
    create_dirs(ckpt_dir)

    # IMP! change this according to your wandb settings and set mode="online"
    wandb_config = vars(config)
    wandb.init(
        project="MMLMCalibration",
        reinit=True,
        entity="<AddYourEntity>",
        config=wandb_config,
        mode="disabled",
    )

    logger.info(f"Loading {config.dataset} Dataset!")
    if config.train_lang == "en":
        train_df, val_df, testlang2df, vallang2df = load_data(
            config.dataset,
            config.data_dir,
            debug=config.debug,
            max_train_samples=config.max_train_samples,
        )
    else:
        trainlang2df, _, testlang2df, vallang2df = load_data(
            config.dataset,
            config.data_dir,
            debug=config.debug,
            max_train_samples=config.max_train_samples,
            load_train_en_only=False,
        )
        train_df = trainlang2df[config.train_lang]
        val_df = vallang2df[config.train_lang]

    logger.info("Initializing Datasets")
    train_dataset = init_sentence_pair_dataset(
        train_df,
        max_length=config.max_length,
        tokenizer_variant=config.mmlm,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    val_dataset = init_sentence_pair_dataset(
        val_df,
        max_length=config.max_length,
        tokenizer_variant=config.mmlm,
    )
    val_loader = DataLoader(val_dataset, batch_size=config.eval_batch_size)

    logger.info("Initializing Model")
    # import pdb
    # pdb.set_trace()
    model = TempScaledSentencePairClassifier(
        noutputs=train_df["label"].nunique(), model_type=config.mmlm
    )

    ckpt_file = f"{ckpt_dir}/model.pt"
    if not os.path.exists(ckpt_file) or config.re_train:
        logger.info("Training Model")
        model, val_acc = train_fancy(
            model,
            train_loader,
            val_loader,
            logger,
            loss_fn="ce",
            lr=config.lr,
            num_epochs=config.num_epochs,
            device=config.device,
            label_smoothing=config.alpha_smoothing,
            clip_grad_norm=config.clip_grad_norm,
            eval_every=config.eval_every,
        )
        wandb.log({"val_acc": val_acc})
        logger.info(f"Validation Accuracy: {val_acc}")
        save_model(
            model, ckpt_file,
        )
    else:
        logger.info("Found Trained Model. Loading..........")
        model = load_model(model, ckpt_file)
        model.to(config.device)
        val_acc = evaluate(model, val_loader, device=config.device)
        wandb.log({"val_acc": val_acc})
        logger.info(f"Validation Accuracy: {val_acc}")

    if config.few_shot_learning:
        logger.info(f"Running Few-Shot on language {config.few_shot_lang} with {config.few_shot_size} examples")
        train_dataset_tgt = init_sentence_pair_dataset(
            vallang2df[config.few_shot_lang].iloc[:config.few_shot_size],
            max_length=config.max_length,
            tokenizer_variant=config.mmlm,
        )
        train_loader_tgt = DataLoader(
            train_dataset_tgt, batch_size=config.batch_size_curr, shuffle=True
        )
        model, val_acc = train_fancy(
            model,
            train_loader_tgt,
            val_loader,
            logger,
            loss_fn="ce",
            lr=config.lr_curr,
            num_epochs=config.num_epochs_curr,
            device=config.device,
            label_smoothing=config.alpha_smoothing,
            clip_grad_norm=config.clip_grad_norm,
            eval_every=config.eval_every,
        )
        wandb.log({"val_acc": val_acc})
        logger.info(f"Validation Accuracy after Few-Shot: {val_acc}")

    if config.temp_scaling:
        logger.info(f"Running Temperature Scaling with {config.cal_lang} data")
        cal_dev_dataset = init_sentence_pair_dataset(
            vallang2df[config.cal_lang].iloc[:config.cal_size],
            max_length=config.max_length,
            tokenizer_variant=config.mmlm,
        )
        cal_dev_loader = DataLoader(
            cal_dev_dataset,
            batch_size=config.eval_batch_size
         )
        model.set_temperature(
            cal_dev_loader, device=config.device
        )
        wandb.log({"Temperature": model.temperature.mean().item()})
        logger.info(f"Temperature: {model.temperature.mean().item()}")

    logger.info("Computing Accuracies and Calibration Errors on Test Datasets")
    lang2acc, lang2ece, _ = get_acc_nd_cal_errors(
        model, testlang2df, config, log=True
    )
    logger.info(f"Language Wise Accuracy: {lang2acc}")
    logger.info(f"Language Wise ECE: {lang2ece}")


if __name__ == "__main__":
    main()


