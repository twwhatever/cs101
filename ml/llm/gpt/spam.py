import urllib.request
import zipfile
import os
from pathlib import Path

import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

CACHE_DIR = ".cache/"


def _download_and_unzip_spam_data(
    url, zip_path, extracted_path, data_file_path
):
    def maybe_download_and_extract_data():
        if data_file_path.exists():
            print(
                f"{data_file_path} already exists. Skipping download and extraction"
            )
            return

        with urllib.request.urlopen(url) as response:
            with open(zip_path, "wb") as out_file:
                out_file.write(response.read())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extracted_path)

        original_file_path = Path(extracted_path) / "SMSSpamCollection"
        os.rename(original_file_path, data_file_path)
        print(f"File downloaded and saved as {data_file_path}")

    maybe_download_and_extract_data()
    return pd.read_csv(
        data_file_path, sep="\t", header=None, names=["Label", "Text"]
    )


def _create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state=123
    )
    return pd.concat(
        [ham_subset, df[df["Label"] == "spam"]]
    )


def _random_split(df, train_frac, val_frac):
    # Shuffle the data.
    df = df.sample(
        frac=1, random_state=123
    ).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    val_end = train_end + int(len(df) * val_frac)
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    return train_df, val_df, test_df


class SpamDataset(Dataset):
    def __init__(
        self,
        csv_file,
        tokenizer,
        max_length=None,
        pad_token_id=50256
    ):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

            # Truncate if necessary.
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        self.encoded_texts = [
            text + [pad_token_id] * (self.max_length - len(text))
            for text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for text in self.encoded_texts:
            max_length = max(max_length, len(text))
        return max_length


def prepare_spam_data(tokenizer, batch_size=8):
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = f"{CACHE_DIR}sms_spam_collection.zip"
    extracted_path = f"{CACHE_DIR}sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    spam_df = _download_and_unzip_spam_data(
        url, zip_path, extracted_path, data_file_path)

    # For simplicity we'll use a balanced dataset.
    balanced_df = _create_balanced_dataset(spam_df)

    # Convert to numeric labels.
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    split_labels = ["train", "validation", "test"]
    splits = _random_split(balanced_df, 0.7, 0.1)

    loaders = {}
    max_length = None
    for (label, split) in zip(split_labels, splits):
        filename = f"{CACHE_DIR}{label}.csv"
        split.to_csv(filename, index=None)
        dataset = SpamDataset(
            csv_file=filename,
            max_length=max_length,
            tokenizer=tokenizer,
        )

        # Infer max_length from the training data.
        if not max_length:
            max_length = dataset.max_length

        loaders[label] = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=label == "train",
        )

    return loaders
