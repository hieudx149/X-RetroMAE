import argparse
import random
from pathlib import Path

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, PreTrainedTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--short_seq_prob", type=float, default=0.1)

    return parser.parse_args()


def create_wiki_data(file_path: str, tokenizer: PreTrainedTokenizer):

    def psg_tokenize_function(examples):
        return tokenizer(examples["text"], add_special_tokens=False, truncation=True,
                         return_attention_mask=False, eturn_token_type_ids=False)

    psgs_dataset = load_dataset("text", data_files=file_path)
    tokenized_dataset = psgs_dataset.map(psg_tokenize_function, num_proc=8, remove_columns=["text"], batched=True)
    processed_dataset = tokenized_dataset.rename_column("input_ids", "token_ids")
    return processed_dataset  


def create_news_data(file_path: str, tokenizer: PreTrainedTokenizer, max_seq_length: int, short_seq_prob: float = 0.0):
    target_length = max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)

    def sent_tokenize_function(examples):
        return tokenizer(examples["text"], add_special_tokens=False, truncation=True,
                         return_attention_mask=False, return_token_type_ids=False)

    def sent_pad_each_line(examples):
        blocks = []
        curr_block = []
        curr_tgt_len = target_length if random.random() > short_seq_prob else random.randint(10, target_length)
        for sent in examples["input_ids"]:
            if len(curr_block) + len(sent) > curr_tgt_len:
                if len(curr_block) > 0:
                    blocks.append(curr_block)
                    curr_tgt_len = target_length if random.random() > short_seq_prob else random.randint(10, target_length)
                curr_block = sent
            else:
                curr_block.extend(sent)
        if len(curr_block) > 0:
            blocks.append(curr_block)
        return {"token_ids": blocks}

    sents_dataset = load_dataset("text", data_files=file_path, split="train")
    tokenized_dataset = sents_dataset.map(sent_tokenize_function, num_proc=8, remove_columns=["text"], batched=True)
    processed_dataset = tokenized_dataset.map(sent_pad_each_line, num_proc=8, batched=True, remove_columns=["input_ids"])
    return processed_dataset      


if __name__ == "__main__":
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    wiki = create_wiki_data("wiki_passages_tk.txt",tokenizer)
    news = create_news_data("news-sents.txt", tokenizer, args.max_seq_length, args.short_seq_prob)
    dataset = concatenate_datasets([wiki, news])
    dataset.save_to_disk(args.output_dir)
