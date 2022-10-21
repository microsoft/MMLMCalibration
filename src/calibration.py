import pdb
from imp import is_builtin

import numpy as np

# from src.netcal.metrics.ECE import ECE
from netcal.metrics import ECE
from src.trainer import predict
from torch.utils.data import DataLoader


def get_callibration_error_from_preds_nd_labels(pred_probs, labels, M):
    ece_calculator = ECE(bins=M)
    return ece_calculator.measure(pred_probs, labels)

    # is_binary = len(pred_probs.shape) == 1
    # pred_probs_flat = []
    # labels_oh_flat = []
    # if is_binary:
    #     for pred_prob, y in zip(pred_probs, labels):
    #         pred_probs_flat += [1 - pred_prob, pred_prob]
    #         if y == 1:
    #             labels_oh_flat += [0, 1]
    #         else:
    #             labels_oh_flat += [1, 0]

    # else:
    #     for pred_prob, y in zip(pred_probs, labels):
    #         pred_probs_flat += pred_prob.tolist()
    #         label_oh = np.zeros(len(pred_prob))
    #         label_oh[int(y)] = 1
    #         labels_oh_flat += label_oh.tolist()

    # # y_hats = (pred_probs > 0.5).astype(float)
    # # p_hats = [prob if y == 1 else 1 - prob for prob,y in zip(pred_probs, labels)]

    # confidences = []
    # accuracies = []
    # ece = 0
    # num_bins_present = 0
    # for m in range(1, M + 1):
    #     int_start = (m - 1) / M
    #     int_end = m / M

    #     bin_probs_labels = [
    #         (prob, y)
    #         for prob, y in zip(pred_probs_flat, labels_oh_flat)
    #         if prob > int_start and prob <= int_end
    #     ]

    #     if len(bin_probs_labels) == 0:
    #         continue

    #     bin_probs, bin_labels = list(map(list, zip(*bin_probs_labels)))

    #     bin_probs = np.array(bin_probs)
    #     bin_labels = np.array(bin_labels)

    #     confidence = np.mean(bin_probs)
    #     accuracy = (bin_labels).mean()

    #     confidences.append(confidence)
    #     accuracies.append(accuracy)

    #     ece += (len(bin_probs) / len(pred_probs_flat)) * abs(confidence - accuracy)
    #     num_bins_present += 1

    # ece = ece

    # return ece, confidences, accuracies


def get_callibration_error(
    test_dataset, modelA, modelB=None, M=10, batch_size=8, device="cuda"
):

    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    pred_probs = predict(modelA, test_loader, device="cuda")
    if modelB is not None:
        pred_probs = (pred_probs + predict(modelB, test_loader, device="cuda")) / 2
    labels = np.array(test_dataset.labels).astype(float)
    return get_callibration_error_from_preds_nd_labels(pred_probs, labels, M)


def get_callibration_error_nmodels(
    test_dataset, models, M=10, batch_size=8, device="cuda"
):

    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    pred_probs_all = [
        predict(model, test_loader, device=device)[
            np.newaxis,
        ]
        for model in models
    ]
    pred_probs_all = np.concatenate(pred_probs_all, axis=0)
    pred_probs = pred_probs_all.mean(axis=0)
    labels = np.array(test_dataset.labels).astype(float)

    return get_callibration_error_from_preds_nd_labels(pred_probs, labels, M)


def get_cace_from_preds_nd_labels(pred_probs, labels, M):

    is_binary = len(pred_probs.shape) == 1
    n_classes = 2 if is_binary else pred_probs.shape[-1]
    confidences = []
    accuracies = []
    cace = 0

    for m in range(1, M + 1):
        int_start = (m - 1) / M
        int_end = m / M
        acc_num = 0
        acc_deno = 0
        q_weight = 0
        for k in range(n_classes):
            num_frac = 0
            deno_frac = 0
            if is_binary:
                pred_probs_k = pred_probs if k == 1 else 1 - pred_probs
            else:
                pred_probs_k = pred_probs[:, k]

            for pred_prob, label in zip(pred_probs_k, labels):
                if label == k and (pred_prob > int_start and pred_prob <= int_end):
                    num_frac += 1

                if pred_prob > int_start and pred_prob <= int_end:
                    deno_frac += 1

            q_weight += deno_frac
            num_frac = num_frac / len(labels)
            deno_frac = deno_frac / len(labels)

            acc_num += num_frac
            acc_deno += deno_frac

        if acc_num != 0:
            accuracy = acc_num / acc_deno
        else:
            accuracy = 0
        confidence = (int_start + int_end) / 2
        q_weight = q_weight / len(labels)
        confidences.append(confidence)
        accuracies.append(accuracy)

        cace += q_weight * abs(confidence - accuracy)

    return cace, confidences, accuracies


def get_cace(test_dataset, modelA, modelB=None, M=10, batch_size=8, device="cuda"):
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    pred_probs = predict(modelA, test_loader, device="cuda")
    if modelB is not None:
        pred_probs = (pred_probs + predict(modelB, test_loader, device="cuda")) / 2
    labels = np.array(test_dataset.labels).astype(float)
    return get_cace_from_preds_nd_labels(pred_probs, labels, M=M)
