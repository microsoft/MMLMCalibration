import copy
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import wandb
from src.calibration_losses import _ECELoss
from src.model import (
    TempScaledMultipleChoiceClassifier,
    TempScaledSentencePairClassifier,
)


def evaluate(model, test_loader, device="cuda"):

    accuracy = 0
    model = model.eval()
    model = model.to(device)

    with torch.no_grad():
        for batch in test_loader:
            try:
                input_ids, mask, labels = batch
                choice_mask = None
            except:
                input_ids, mask, choice_mask, labels = batch
                choice_mask = choice_mask.to(device)

            input_ids = input_ids.to(device)
            mask = mask.to(device)
            labels = labels.numpy()
            if choice_mask is None:
                logits = model(input_ids, mask)
            else:
                logits = model(input_ids, mask, choice_mask)
            if isinstance(logits, tuple):
                logits = logits[0]
            if len(logits.shape) > 1:
                pred_labels = logits.argmax(dim=-1).detach().cpu().numpy()
            else:
                pred_labels = (logits > 0.5).float().detach().cpu().numpy()

            accuracy += (labels == pred_labels).astype(float).mean()

    accuracy = accuracy / len(test_loader)

    return accuracy


def predict(model, test_loader, device="cuda"):

    pred_probs_all = []

    with torch.no_grad():
        for batch in test_loader:
            try:
                input_ids, mask, labels = batch
                choice_mask = None
            except:
                input_ids, mask, choice_mask, labels = batch
                choice_mask = choice_mask.to(device)

            input_ids = input_ids.to(device)
            mask = mask.to(device)
            labels = labels.numpy()
            if choice_mask is None:
                logits = model(input_ids, mask)
            else:
                logits = model(input_ids, mask, choice_mask)

            if isinstance(logits, tuple):
                logits = logits[0]
            if isinstance(model, TempScaledSentencePairClassifier) or isinstance(
                model, TempScaledMultipleChoiceClassifier
            ):
                logits = model.temperature_scale(logits)
            if len(logits.shape) > 1:
                pred_probs = F.softmax(logits, dim=-1)
            else:
                pred_probs = F.sigmoid(logits)
            pred_probs_all.append(pred_probs)

    pred_probs_all = torch.cat(pred_probs_all, axis=0).detach().cpu().numpy()

    return pred_probs_all


def train(
    model,
    train_loader,
    val_loader,
    loss_fn="bce",
    lr=1e-5,
    num_epochs=3,
    device="cuda",
    label_smoothing=0.0,
):

    best_val_accuracy = float("-inf")
    best_model = None

    epoch_loss = 0
    model = model.to(device)

    if loss_fn == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        epoch_loss = 0
        model = model.train()

        for train_batch in tqdm.tqdm(train_loader):
            optimizer.zero_grad()

            try:
                input_ids, mask, labels = train_batch
                choice_mask = None
            except:
                input_ids, mask, choice_mask, labels = train_batch
                choice_mask = choice_mask.to(device)
            # Transfer the features, mask and labels to device
            input_ids = input_ids.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            if choice_mask is None:
                preds = model(input_ids, mask)
            else:
                preds = model(input_ids, mask, choice_mask)
            if loss_fn == "bce":
                loss = criterion(preds, labels.float())
            else:
                loss = criterion(preds, labels)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(train_loader)
        val_accuracy = evaluate(model, val_loader, device=device)

        # Model selection
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = copy.deepcopy(model)  # Create a copy of model

        print(
            f"Epoch {epoch} completed | Average Training Loss: {epoch_loss} | Validation Accuracy: {val_accuracy}"
        )
        wandb.log(
            {"training_loss": epoch_loss, "validation_accuracy": val_accuracy}, epoch
        )

    best_model.zero_grad()
    model.zero_grad()
    return best_model, best_val_accuracy


def train_fancy(
    model,
    train_loader,
    val_loader,
    logger = None,
    loss_fn="bce",
    lr=1e-5,
    num_epochs=3,
    device="cuda",
    label_smoothing=0.0,
    clip_grad_norm=0,
    eval_every=500,
):
    global_step = 0
    best_val_accuracy = float("-inf")
    best_model = None

    epoch_loss = 0
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    running_loss = 0
    for epoch in range(num_epochs):

        for train_batch in tqdm.tqdm(train_loader):
            global_step += 1
            model = model.train()
            optimizer.zero_grad()

            try:
                input_ids, mask, labels = train_batch
                choice_mask = None
            except:
                input_ids, mask, choice_mask, labels = train_batch
                choice_mask = choice_mask.to(device)
            # Transfer the features, mask and labels to device
            input_ids = input_ids.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            if choice_mask is None:
                preds = model(input_ids, mask)
            else:
                preds = model(input_ids, mask, choice_mask)
            loss = criterion(preds, labels)
            loss.backward()
            if clip_grad_norm != 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=clip_grad_norm
                )
            optimizer.step()

            running_loss += loss.item()
            wandb.log({"training_loss": loss.item()}, global_step)
            if global_step % eval_every == 0:

                val_accuracy = evaluate(model, val_loader, device=device)
                wandb.log({"validation_accuracy": val_accuracy}, global_step)
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_model = copy.deepcopy(model)  # Create a copy of model
                
                if logger is None:
                    print(
                        f"Step {global_step}: Training Loss: {running_loss} | Validation Accuracy = {val_accuracy}"
                    )
                else:
                    logger.info(
                        f"Step {global_step}: Training Loss: {running_loss} | Validation Accuracy = {val_accuracy}"
                    )
                running_loss = 0

    if best_model is None:
        best_model = copy.deepcopy(model)
        best_val_accuracy = evaluate(model, val_loader, device=device)
        wandb.log({"validation_accuracy": best_val_accuracy}, global_step)

    best_model.zero_grad()
    model.zero_grad()
    return best_model, best_val_accuracy


def adv_train(
    model,
    train_loader,
    val_loader,
    lr=1e-5,
    num_epochs=3,
    device="cuda",
    label_smoothing=0.0,
):
    best_val_accuracy = float("-inf")
    best_model = None

    epoch_loss = 0
    model = model.to(device)

    if label_smoothing != 0:
        model.cls_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        epoch_loss = 0
        model = model.train()

        for train_batch in tqdm.tqdm(train_loader):
            optimizer.zero_grad()

            input_ids, mask, labels, langs = train_batch

            if (labels == -1).all():
                continue

            # Transfer the features, mask, labels and langs to device
            input_ids = input_ids.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            langs = langs.to(device)

            # Get the loss
            loss = model.get_loss(input_ids, mask, labels, langs)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(train_loader)
        val_accuracy = evaluate(model, val_loader, device=device)

        # Model selection
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = copy.deepcopy(model)  # Create a copy of model

        print(
            f"Epoch {epoch} completed | Average Training Loss: {epoch_loss} | Validation Accuracy: {val_accuracy}"
        )

    best_model.zero_grad()
    model.zero_grad()
    return best_model, best_val_accuracy
