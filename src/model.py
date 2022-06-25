import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.calibration_losses import _ECELoss
from transformers import AutoModel


class SentencePairClassifier(nn.Module):
    def __init__(self, noutputs=1, model_type="bert-base-multilingual-uncased"):

        super(SentencePairClassifier, self).__init__()

        self.mmlm_layer = AutoModel.from_pretrained(model_type)
        self.hidden_size = self.mmlm_layer.config.hidden_size
        self.output_layer = nn.Linear(self.hidden_size, noutputs)
        self.output_size = noutputs

    def forward(self, input_ids, attn_mask=None):
        mmlm_output = self.mmlm_layer(input_ids, attention_mask=attn_mask).pooler_output
        output = self.output_layer(mmlm_output)

        return output.squeeze(-1)


class TempScaledSentencePairClassifier(SentencePairClassifier):
    def __init__(self, noutputs=1, model_type="bert-base-multilingual-uncased"):
        super(TempScaledSentencePairClassifier, self).__init__(
            noutputs=noutputs, model_type=model_type
        )
        self.temperature = nn.Parameter(torch.ones(1))

    def get_logits(self, input_ids, attn_mask=None):
        return self(input_ids, attn_mask)

    def predict_proba(self, input_ids, attn_mask=None):
        logits = self.get_logits(input_ids, attn_mask)
        probs = F.softmax(self.temperature_scale(logits), dim=-1)
        return probs

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(
            logits.size(0), logits.size(1)
        )
        return logits / temperature

    def set_temperature(self, val_loader, device="cuda"):
        self.temperature.data[0] = 1.5
        is_binary_cls = True if self.output_size == 1 else False

        if is_binary_cls:
            nll_criterion = nn.BCEWithLogitsLoss().to(device)
        else:
            nll_criterion = nn.CrossEntropyLoss().to(device)

        ece_criterion = _ECELoss().to(device)

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input_ids, mask, labels in val_loader:
                input_ids = input_ids.to(device)
                mask = mask.to(device)
                logits = self.get_logits(input_ids, mask)
                logits_list.append(logits)
                labels_list.append(labels)
            logits = torch.cat(logits_list).to(device)
            labels = torch.cat(labels_list).to(device)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print(
            "Before temperature - NLL: %.3f, ECE: %.3f"
            % (before_temperature_nll, before_temperature_ece)
        )

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(
            self.temperature_scale(logits), labels
        ).item()
        after_temperature_ece = ece_criterion(
            self.temperature_scale(logits), labels
        ).item()
        print("Optimal temperature: %.3f" % self.temperature.item())
        print(
            "After temperature - NLL: %.3f, ECE: %.3f"
            % (after_temperature_nll, after_temperature_ece)
        )



class MultipleChoiceClassifier(nn.Module):
    def __init__(self, model_type="bert-base-multilingual-uncased"):

        super(MultipleChoiceClassifier, self).__init__()

        self.mmlm_layer = AutoModel.from_pretrained(model_type)
        self.hidden_size = self.mmlm_layer.config.hidden_size
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1),
        )

    def forward(
        self, input_ids, attn_mask=None, choice_mask=None,
    ):
        batch_size, num_choices, seq_len = input_ids.shape
        input_ids_bc_merged = input_ids.reshape(-1, seq_len)
        if attn_mask is not None:
            attn_mask = attn_mask.reshape(-1, seq_len)
        mmlm_output = self.mmlm_layer(
            input_ids_bc_merged, attention_mask=attn_mask
        ).pooler_output
        scores = self.output_layer(mmlm_output)
        scores = scores.reshape(batch_size, num_choices)
        if choice_mask is not None:
            choice_mask = choice_mask + (1 - choice_mask) * (-1e19)
            scores = scores * choice_mask
        # pdb.set_trace()
        return scores


class TempScaledMultipleChoiceClassifier(MultipleChoiceClassifier):
    def __init__(self, model_type="bert-base-multilingual-uncased"):
        super(TempScaledMultipleChoiceClassifier, self).__init__(model_type=model_type)
        self.temperature = nn.Parameter(torch.ones(1))

    def get_logits(self, input_ids, attn_mask=None, choice_mask=None):
        return self(input_ids, attn_mask, choice_mask)

    def predict_proba(self, input_ids, attn_mask=None):
        logits = self.get_logits(input_ids, attn_mask)
        probs = F.softmax(self.temperature_scale(logits), dim=-1)
        return probs

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(
            logits.size(0), logits.size(1)
        )
        return logits / temperature

    def set_temperature(self, val_loader, device="cuda"):
        self.temperature.data[0] = 1.5
        nll_criterion = nn.CrossEntropyLoss().to(device)

        ece_criterion = _ECELoss().to(device)

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input_ids, mask, choice_mask, labels in val_loader:
                input_ids = input_ids.to(device)
                mask = mask.to(device)
                choice_mask = choice_mask.to(device)
                logits = self.get_logits(input_ids, mask, choice_mask)
                logits_list.append(logits)
                labels_list.append(labels)
            logits = torch.cat(logits_list).to(device)
            labels = torch.cat(labels_list).to(device)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print(
            "Before temperature - NLL: %.3f, ECE: %.3f"
            % (before_temperature_nll, before_temperature_ece)
        )

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(
            self.temperature_scale(logits), labels
        ).item()
        after_temperature_ece = ece_criterion(
            self.temperature_scale(logits), labels
        ).item()
        print("Optimal temperature: %.3f" % self.temperature.item())
        print(
            "After temperature - NLL: %.3f, ECE: %.3f"
            % (after_temperature_nll, after_temperature_ece)
        )
