
import numpy as np
import tqdm
import wandb
from src.calibration import (
    get_cace,
    get_cace_from_preds_nd_labels,
    get_callibration_error,
    get_callibration_error_from_preds_nd_labels,
)
from src.dataset import (
    MultipleChoiceDataset,
    SentencePairDataset,
    SentencePairMultilingualDataset,
    init_mcq_task_dataset,
)
from src.trainer import predict, train
from torch.utils.data import DataLoader


def get_acc_nd_cal_errors(model, testlang2df, config, log=True, calculate_cace=True):
    lang2acc, lang2cace, lang2ece = {}, {}, {}
    langs = list(testlang2df.keys())
    for lang in tqdm.tqdm(langs):
        test_df = testlang2df[lang]
        if config.dataset in ["siqa", "copa", "siqa_nd_copa"]:
            test_dataset = init_mcq_task_dataset(
                test_df,
                dataset=config.dataset,
                max_length=config.max_length,
                tokenizer_variant=config.mmlm,
                max_num_choices=3 if config.dataset == "siqa" else 2,
            )
        else:
            test_dataset = SentencePairDataset(
                test_df["sentence1"].values.tolist(),
                test_df["sentence2"].values.tolist(),
                test_df["label"].values.tolist(),
                max_length=config.max_length,
                tokenizer_variant=config.mmlm,
            )
        test_loader = DataLoader(test_dataset, batch_size=config.eval_batch_size)
        # import pdb
        # pdb.set_trace()
        pred_probs = predict(model, test_loader, device="cuda")
        preds = np.argmax(pred_probs, axis=-1)
        labels = np.array(test_dataset.labels).astype(float)
        test_acc = (
            preds == labels
        ).mean()  # evaluate(model, test_loader, device=config.device,)

        # pdb.set_trace()
        test_ece = get_callibration_error_from_preds_nd_labels(
            pred_probs, labels, M=config.cal_bins,
        )

        if calculate_cace:
            test_cace, _, _ = get_cace_from_preds_nd_labels(
                pred_probs, labels, M=config.cal_bins,
            )

        lang2acc[lang] = test_acc
        lang2ece[lang] = test_ece
        if calculate_cace:
            lang2cace[lang] = test_cace

        if log:
            wandb.log({f"{lang}_test_acc": test_acc})
            wandb.log({f"{lang}_test_ece": test_ece})

            if calculate_cace:
                wandb.log({f"{lang}_test_cace": test_cace})
    if log:
        wandb.log(
            {
                f"avg_test_acc": np.mean(list(lang2acc.values())),
                f"avg_test_ece": np.mean(list(lang2ece.values())),
            }
        )
        if calculate_cace:
            wandb.log(
                {f"avg_test_cace": np.mean(list(lang2cace.values())),}
            )
    if calculate_cace:
        return lang2acc, lang2ece, lang2cace
    else:
        return lang2acc, lang2ece
