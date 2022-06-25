import pdb

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class SentencePairDataset(Dataset):
    def __init__(
        self,
        sentence1s,
        sentence2s,
        labels,
        max_length,
        tokenizer_variant="bert-base-multilingual-uncased",
    ):

        self.sentence1s = sentence1s
        self.sentence2s = sentence2s
        self.labels = labels
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_variant)

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        sentence1 = self.sentence1s[idx]
        sentence2 = self.sentence2s[idx]
        label = int(self.labels[idx])

        sentence_concat = f"{sentence1} {self.tokenizer.sep_token} {sentence2}"

        tokenizer_output = self.tokenizer(
            sentence_concat,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokenizer_output["input_ids"]
        mask = tokenizer_output["attention_mask"]

        return input_ids.squeeze(0), mask.squeeze(0), label


class SentencePairMultilingualDataset(SentencePairDataset):
    def __init__(
        self,
        lang2sentence1s,
        lang2sentence2s,
        lang2labels,
        max_length,
        tokenizer_variant="bert-base-multilingual-uncased",
    ):

        self.langs = sorted(list(lang2sentence1s.keys()))
        self.lang2id = {lang: idx for idx, lang in enumerate(self.langs)}
        sentence1s, sentence2s, labels, lang_ids = [], [], [], []
        for lang in self.langs:
            sentence1s += lang2sentence1s[lang].tolist()
            sentence2s += lang2sentence2s[lang].tolist()
            labels += lang2labels[lang].tolist()
            lang_ids += [self.lang2id[lang] for _ in lang2sentence1s[lang]]

        super().__init__(sentence1s, sentence2s, labels, max_length, tokenizer_variant)

        self.lang_ids = lang_ids

    def __getitem__(self, idx):
        input_ids, mask, label = super().__getitem__(idx)
        return input_ids, mask, label, self.lang_ids[idx]


class MultipleChoiceDataset(Dataset):
    def __init__(
        self,
        contexts,
        questions,
        choices,
        labels,
        max_length,
        max_num_choices=3,
        tokenizer_variant="bert-base-multilingual-uncased",
    ):

        self.contexts = contexts
        self.questions = questions
        self.choices = choices
        self.labels = labels
        self.max_length = max_length
        self.max_num_choices = max_num_choices
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_variant)

    def __getitem__(self, idx):
        def concat_nd_tokenize_text(context, question, choice):
            concat_text = f"{context} {self.tokenizer.sep_token} {question} {self.tokenizer.sep_token} {choice}"
            tokenizer_output = self.tokenizer(
                concat_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tokenizer_output["input_ids"]
            mask = tokenizer_output["attention_mask"]
            return input_ids, mask

        context, question, choices = (
            self.contexts[idx],
            self.questions[idx],
            list(self.choices[idx]),
        )
        assert self.max_num_choices >= len(choices)
        padded_choices = choices + [
            self.tokenizer.pad_token for _ in range(self.max_num_choices - len(choices))
        ]
        choice_mask = [1 for _ in range(len(choices))] + [
            0 for _ in range(len(padded_choices) - len(choices))
        ]
        input_ids_all, masks_all = [], []
        for choice in padded_choices:
            input_ids, mask = concat_nd_tokenize_text(context, question, choice)
            input_ids_all.append(input_ids)
            masks_all.append(mask)

        return (
            torch.cat(input_ids_all),
            torch.cat(masks_all),
            torch.FloatTensor(choice_mask),
            int(self.labels[idx]),
        )

    def __len__(self):
        return len(self.labels)

def init_sentence_pair_dataset(
    df, max_length, tokenizer_variant
):
    sentence1s, sentence2s = df["sentence1"].values.tolist(), df["sentence2"].values.tolist()
    labels = df["label"].values.tolist()

    return SentencePairDataset(
        sentence1s,
        sentence2s,
        labels,
        max_length,
        tokenizer_variant=tokenizer_variant
    )
    

def init_mcq_task_dataset(
    df, dataset, max_length, tokenizer_variant, max_num_choices=3,
):
    contexts = (
        df["context"].values.tolist()
        if dataset == "siqa"
        else df["premise"].values.tolist()
    )
    questions = df["question"].values.tolist()
    labels = df["label"]
    if dataset == "siqa":
        choices = list(
            zip(
                df["answerA"].values.tolist(),
                df["answerB"].values.tolist(),
                df["answerC"].values.tolist(),
            )
        )
    else:
        choices = list(
            zip(df["choice1"].values.tolist(), df["choice2"].values.tolist())
        )

    return MultipleChoiceDataset(
        contexts=contexts,
        questions=questions,
        choices=choices,
        labels=labels,
        max_length=max_length,
        tokenizer_variant=tokenizer_variant,
        max_num_choices=max_num_choices,
    )


def init_mixed_mcq_dataset(siqa_df, copa_df, max_length, tokenizer_variant):
    siqa_contexts = siqa_df["context"].values.tolist()
    copa_contexts = copa_df["premise"].values.tolist()
    siqa_questions = siqa_df["question"].values.tolist()
    copa_questions = copa_df["question"].values.tolist()

    siqa_choices = list(
        zip(
            siqa_df["answerA"].values.tolist(),
            siqa_df["answerB"].values.tolist(),
            siqa_df["answerC"].values.tolist(),
        )
    )
    copa_choices = list(
        zip(copa_df["choice1"].values.tolist(), copa_df["choice2"].values.tolist())
    )

    siqa_labels = siqa_df["label"].values.tolist()
    copa_labels = copa_df["label"].values.tolist()
    return MultipleChoiceDataset(
        contexts=siqa_contexts + copa_contexts,
        questions=siqa_questions + copa_questions,
        choices=siqa_choices + copa_choices,
        labels=siqa_labels + copa_labels,
        max_length=max_length,
        tokenizer_variant=tokenizer_variant,
    )
