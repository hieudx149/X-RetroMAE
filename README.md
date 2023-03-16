# X-RetroMAE
[RetroMAE: Pre-Training Retrieval-oriented Language Models Via Masked Auto-Encoder](https://arxiv.org/pdf/2205.12035.pdf) is a powerful model for dense retrieval task, it is pretrained on unlabeled data, which is very useful for languages that don't have a lot of labeled data.

**X-RetroMAE** tries to modify [RetroMAE](https://github.com/staoxiao/RetroMAE) to be compatible with RoBERTa and XLM-RoBERTa, hope this project will help anyone who wants to apply RetroMAE to their own language rather than English.

## Modification
Here I list all the changes of X-RetroMAE compared to RetroMAE
* Change all Bert* to Roberta* in src/pretrain/enhancedDecoder.py, src/pretrain/modeling.py, src/pretrain/run.py 
* In pretrain/modeling.py:
  * self.lm.bert -> self.lm.roberta
  * self.lm.cls -> self.lm.lm_head
* In pretrain/data.py: I create a DataCollatorForWholeWordMask Class for my own tokenizer
* In src/examples/pretra/preprocess.py: I changed a bit to fit my data, but most of it is still based on the original code

## Setup
```
pip install --upgrade pip
pip install -r requirements.txt
```

## Run pretraining
First make sure that you have preprocessed your own data first by running the preprocessing.py in examples/pretrain, then run:
```
sh src/run_pretrain.sh
```

## Citation
```
@inproceedings{RetroMAE,
  title={RetroMAE: Pre-Training Retrieval-oriented Language Models Via Masked Auto-Encoder},
  author={Shitao Xiao, Zheng Liu, Yingxia Shao, Zhao Cao},
  url={https://arxiv.org/abs/2205.12035},
  booktitle ={EMNLP},
  year={2022},
}
```
