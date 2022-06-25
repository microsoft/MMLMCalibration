import os
import pandas as pd
from datasets import load_dataset
import pdb

dataset2langs = {
    "pawsx": ["de", "en", "es", "fr", "ja", "ko", "zh"],
    "xnli": [
        "ar",
        "bg",
        "de",
        "el",
        "en",
        "es",
        "fr",
        "hi",
        "ru",
        "sw",
        "th",
        "tr",
        "ur",
        "vi",
        "zh",
    ],
    "xcopa": ["en", "et", "id", "sw", "ta", "th", "tr", "vi", "zh"],
    "marc" : ["en", "es", "de", "fr", "ja", "zh"]
}


def label_indexer(label, dataset="pawsx"):
    def pawsx_label_indexer(label):
        return label

    def xnli_label_indexer(label):
        xnli_label_mapping = {
            "contradiction": 0,
            "entailment": 1,
            "neutral": 2,
            "contradictory": 0,
        }
        return xnli_label_mapping[label]

    def marc_label_indexer(label):
        marc_label_mapping = lambda x : int(x) - 1
        return marc_label_mapping(label)

    if dataset == "pawsx":
        return pawsx_label_indexer(label)
    elif dataset == "xnli":
        return xnli_label_indexer(label)
    elif dataset == "marc":
        return marc_label_indexer(label)
    else:
        raise NotImplementedError

def load_marc_dataset(
    lang,
    max_samples = -1,
    data_dir="data/MARC",
    split="train",
    debug=False, **kwargs
):
    filename = os.path.join(data_dir, split, f"{split}-{lang}.tsv")
    sentence1s = []
    sentence2s = []
    labels = []

    with open(filename) as f:
        for i, line in enumerate(f):
            if max_samples != -1 and i > max_samples:
                break
            row = line.split("\t")
            sentence1 = row[0]
            sentence2 = row[1]
            label = row[-1]

            sentence1s.append(sentence1)
            sentence2s.append(sentence2)
            labels.append(label_indexer(label, "marc"))
        
    return pd.DataFrame(
        {"sentence1": sentence1s, "sentence2": sentence2s, "label": labels}
    )

def load_sentence_pair_dataset(
    lang,
    max_samples=-1,
    dataset="pawsx",
    data_dir="data/pawsx/",
    split="train",
    debug=False,
    hide_non_english_labels=False,
):
    filename = os.path.join(data_dir, split, f"{split}-{lang}.tsv")
    sentence1s = []
    sentence2s = []
    labels = []

    with open(filename) as f:
        for i, line in enumerate(f):
            if max_samples != -1 and i > max_samples:
                break
            row = line.split("\t")
            if len(row) == 5:
                row = row[2:]
            sentence1 = row[0]
            sentence2 = row[1]
            label = row[2].split("\n")[0]

            sentence1s.append(sentence1)
            sentence2s.append(sentence2)
            try:
                if lang != "en" and hide_non_english_labels:
                    labels.append(-1)
                else:
                    labels.append(label_indexer(label, dataset))
            except:
                pdb.set_trace()
    return pd.DataFrame(
        {"sentence1": sentence1s, "sentence2": sentence2s, "label": labels}
    )

def load_siqa_dataset(
    max_train_samples=-1,
    debug = False
):
    dataset = load_dataset("social_i_qa")
    max_samples = max_train_samples if max_train_samples != -1 else len(dataset["train"]["label"])
    if debug:
        max_samples = 100
    train_df = pd.DataFrame(dataset["train"]).iloc[:max_samples]
    dev_df = pd.DataFrame(dataset["validation"])

    train_df["label"] = train_df["label"].astype(int) - 1
    dev_df["label"] = dev_df["label"].astype(int) - 1

    return train_df, dev_df

def load_copa_dataset( 
    max_train_samples=-1,
    debug = False,
    use_dev_for_train_for_non_en = True    
    ):


    en_dataset = load_dataset("super_glue", "copa")
    max_samples = max_train_samples if max_train_samples != -1 else len(en_dataset["train"]["label"])
    if debug:
        max_samples = 100

    train_en_df = pd.DataFrame(en_dataset["train"]).iloc[:max_samples]
    dev_en_df = pd.DataFrame(en_dataset["validation"])
    test_en_df = pd.DataFrame(en_dataset["validation"]) # No labels found for original data

    lang2train_dfs = {"en" : train_en_df}
    lang2dev_dfs = {"en" : dev_en_df}
    lang2test_dfs = {"en": test_en_df}
    for lang in dataset2langs["xcopa"]:
        if lang == "en":
            continue
        xcopa_dataset = load_dataset("xcopa", lang)
        lang2dev_dfs[lang] = pd.DataFrame(xcopa_dataset["validation"])
        lang2test_dfs[lang] = pd.DataFrame(xcopa_dataset["test"])

    if use_dev_for_train_for_non_en:
        for lang in dataset2langs["xcopa"]:
            if lang == "en":
                continue
            lang2train_dfs[lang] = lang2dev_dfs[lang]
    
    return train_en_df, dev_en_df, lang2train_dfs, lang2dev_dfs, lang2test_dfs
        


def load_data(
    dataset="pawsx",
    data_dir="data/",
    max_train_samples=-1,
    debug=False,
    load_train_en_only=True,
    train_langs=None,
    hide_non_english_labels=False,
    use_dev_for_train_for_non_en = False
):
    full_data_dir = f"{data_dir}/{dataset}"

    data_load_fn = load_sentence_pair_dataset if dataset in ["pawsx", "xnli"] else load_marc_dataset
    if dataset in ["pawsx", "xnli", "marc"]:
        if load_train_en_only:
            train_df = data_load_fn(
                "en",
                dataset=dataset,
                max_samples=max_train_samples if not debug else 100,
                data_dir=full_data_dir,
                split="train",
                debug=debug,
            )
            dev_df = data_load_fn(
                "en",
                dataset=dataset,
                max_samples=-1 if not debug else 100,
                data_dir=full_data_dir,
                split="dev",
                debug=debug,
            )
        else:
            if train_langs is None:
                train_langs = dataset2langs.get(dataset, ["en"])

            train_df = {
                lang: data_load_fn(
                    lang,
                    dataset=dataset,
                    max_samples=max_train_samples if not debug else 100,
                    data_dir=full_data_dir,
                    split="train" if lang == "en" or not use_dev_for_train_for_non_en else "dev",
                    debug=debug,
                    hide_non_english_labels=hide_non_english_labels,
                )
                for lang in train_langs
            }

            dev_df = {
                lang: data_load_fn(
                    lang,
                    dataset=dataset,
                    max_samples=-1 if not debug else 100,
                    data_dir=full_data_dir,
                    split="dev",
                    debug=debug,
                )
                for lang in train_langs
            }

        devlang2df = {
            lang: data_load_fn(
                lang,
                dataset=dataset,
                max_samples=-1 if not debug else 100,
                data_dir=full_data_dir,
                split="dev",
                debug=debug,
            )
            for lang in dataset2langs.get(dataset, ["en"])
        }

        testlang2df = {
            lang: data_load_fn(
                lang,
                dataset=dataset,
                max_samples=-1 if not debug else 100,
                data_dir=full_data_dir,
                split="test",
                debug=debug,
            )
            for lang in dataset2langs.get(dataset, ["en"])
        }

        return train_df, dev_df, testlang2df, devlang2df

    else:
        raise NotImplementedError()

