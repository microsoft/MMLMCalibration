import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import wandb
from src.config import build_parser
from src.dataset import init_mcq_task_dataset, init_mixed_mcq_dataset
from src.model import TempScaledMultipleChoiceClassifier
from src.performance_nd_calibration import get_acc_nd_cal_errors
from src.trainer import evaluate, train_fancy
from src.utils.helper import create_dirs, load_model, save_model
from src.utils.load_data import load_copa_dataset, load_siqa_dataset
from torch.utils.data import DataLoader

logging.basicConfig(format="%(asctime)s %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    config = build_parser(sys.argv[1:])
    config.fine_tuned = False
    # Set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    # Create Directories
    run_id = f"{config.dataset}_{config.mmlm}_MaxLength{config.max_length}_BatchSize{config.batch_size}_LR{config.lr}_EPOCHS{config.num_epochs}_SEED{config.seed}_TrainSize{config.max_train_samples}_MaxGradNorm{str(config.clip_grad_norm).replace('.','_')}_AlphaSmoothing{str(config.alpha_smoothing).replace('.','')}_debug{config.debug}"
    ckpt_dir = f"{config.save_dir}/checkpoints/{run_id}"
    ckpt_file = f"{ckpt_dir}/model.pt"
    create_dirs(ckpt_dir)
    config.run_id = run_id

    wandb_config = vars(config)
    # IMP! change this according to your wandb settings and set mode="online"
    wandb.init(
        project=f"MMLMCalibration",
        reinit=True,
        entity="<AddYourEntity>",
        config=wandb_config,
        mode="disabled",
    )
    logger.info(f"Loading {config.dataset} Dataset!")
    if config.dataset == "siqa":
        train_df, dev_df = load_siqa_dataset(
            config.max_train_samples, debug=config.debug
        )
    elif config.dataset == "copa":
        train_df, dev_df, lang2train_df, lang2dev_df, lang2test_df = load_copa_dataset(
            config.max_train_samples,
            debug=config.debug,
            use_dev_for_train_for_non_en=config.use_dev_for_train_for_non_en,
        )
    elif config.dataset == "siqa_nd_copa":
        train_df_siqa, dev_df_siqa = load_siqa_dataset(
            config.max_train_samples, debug=config.debug
        )
        train_df, dev_df, lang2train_df, lang2dev_df, lang2test_df = load_copa_dataset(
            config.max_train_samples,
            debug=config.debug,
            use_dev_for_train_for_non_en=config.use_dev_for_train_for_non_en,
        )
    else:
        raise NotImplementedError

    logger.info("Initializing Model")
    model = TempScaledMultipleChoiceClassifier(model_type=config.mmlm)


    logger.info("Initializing Datasets")
    if config.dataset != "siqa_nd_copa":
        train_dataset = init_mcq_task_dataset(
            train_df,
            dataset=config.dataset,
            max_length=config.max_length,
            tokenizer_variant=config.mmlm,
            max_num_choices=3 if config.dataset == "siqa" else 2,
        )

    else:
        train_dataset = init_mixed_mcq_dataset(
            train_df_siqa,
            train_df,
            max_length=config.max_length,
            tokenizer_variant=config.mmlm,
        )
    if config.dataset == "copa" and config.merge_dev_sets:
        dev_df = pd.concat(
            [lang_dev_df for lang_dev_df in lang2dev_df.values()], axis=0
        ).reset_index()
    dev_dataset = init_mcq_task_dataset(
        dev_df,
        dataset=config.dataset,
        max_length=config.max_length,
        tokenizer_variant=config.mmlm,
        max_num_choices=3 if config.dataset == "siqa" else 2,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.eval_batch_size)

    if config.re_train or not os.path.exists(ckpt_file):
        pretrained_model_path = (
            f"{config.save_dir}/checkpoints/{config.pretrained_model}/model.pt"
        )
        if config.pretrained_model != "" and os.path.exists(pretrained_model_path):
            logger.info("Found Model Trained on SiQA")
            config.fine_tuned = True
            wandb.config.update({"fine-tuned": True})
            model = load_model(model, pretrained_model_path)
            model.to(config.device)

        logger.info("Training Model")
        model, val_acc = train_fancy(
            model,
            train_loader,
            dev_loader,
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
        logger.info(f"Found Fine-tuned model at {ckpt_dir}")
        model = load_model(model, ckpt_file)
        model.to(config.device)
        val_acc = evaluate(model, dev_loader, device=config.device)
        wandb.log({"val_acc": val_acc})
        logger.info(f"Validation Accuracy: {val_acc}")

    if config.few_shot_learning:
        logger.info(f"Running Few-Shot on language {config.few_shot_lang} with {config.few_shot_size} examples")
        train_dataset_tgt = init_mcq_task_dataset(
            lang2dev_df[config.few_shot_lang],
            dataset=config.dataset,
            max_length=config.max_length,
            tokenizer_variant=config.mmlm,
            max_num_choices=3 if config.dataset == "siqa" else 2,
        )
        train_loader_tgt = DataLoader(
            train_dataset_tgt, batch_size=config.batch_size_curr, shuffle=True
        )
        # ckpt_tgt_file = f"{ckpt_dir}/model_few_shot_{config.few_shot_lang}TrainSize{len(train_dataset_tgt)}_fine_tune.pt"
        # if not os.path.exists(ckpt_tgt_file):
        model, val_acc = train_fancy(
            model,
            train_loader_tgt,
            dev_loader,
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
        logger.info("Learning a Temperature Scale for Improving Calibration")
        cal_dev_dataset = init_mcq_task_dataset(
            lang2dev_df[config.cal_lang],
            dataset=config.dataset,
            max_length=config.max_length,
            tokenizer_variant=config.mmlm,
            max_num_choices=3 if config.dataset == "siqa" else 2,
        )
        cal_dev_loader = DataLoader(cal_dev_dataset, batch_size=config.eval_batch_size)
        model.set_temperature(cal_dev_loader, device=config.device)
        wandb.log({"Temperature": model.temperature.mean().item()})
        logger.info(f"Temperature: {model.temperature.mean().item()}")

    if "copa" in config.dataset:
        logger.info("Computing Accuracies and Calibration Errors on Test Datasets")
        lang2acc, lang2ece = get_acc_nd_cal_errors(
            model, lang2test_df, config, log=True, calculate_cace=False
        )
        logger.info(f"Language Wise Accuracy: {lang2acc}")
        logger.info(f"Language Wise ECE: {lang2ece}")

if __name__ == "__main__":
    main()
