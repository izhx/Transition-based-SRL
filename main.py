"""
"""

import os
import time
import random
import argparse

import numpy as np
import torch

from transformers import BertTokenizer

import nmnlp
from nmnlp.common.config import load_yaml
from nmnlp.common.util import output
from nmnlp.core import Trainer, Vocabulary
from nmnlp.core.optim import build_optimizer

from dataset import UniversalPropositions
from model import TransitionModel

_ARG_PARSER = argparse.ArgumentParser(description="我的实验，需要指定配置文件")
_ARG_PARSER.add_argument('--yaml', '-y',
                         type=str,
                         default='en',
                         help='configuration file path.')
_ARG_PARSER.add_argument('--cuda', '-c',
                         type=str,
                         default='0',
                         help='gpu ids, like: 1,2,3')

_ARGS = _ARG_PARSER.parse_args()

nmnlp.core.trainer.EARLY_STOP_THRESHOLD = 5


def set_seed(seed: int = 123):
    output(f"Process id: {os.getpid()}, cuda: {_ARGS.cuda}, set seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_once(cfg, dataset, vocab, device):
    model = TransitionModel(vocab=vocab, **cfg.model)
    para_num = sum([np.prod(list(p.size())) for p in model.parameters()])
    output(f'param num: {para_num}, {para_num / 1000000:4f}M')
    model.to(device=device)

    optimizer = build_optimizer(model, **cfg.optim)
    scheduler = None
    writer = None
    trainer = Trainer(vars(cfg), dataset, vocab, model, optimizer, None, scheduler,
                      writer, device, **cfg.trainer)

    # 训练过程
    trainer.train()

    return model.metric


def main():
    device = torch.device(f"cuda:{_ARGS.cuda}")
    cfg = argparse.Namespace(**load_yaml(f"./dev/config/{_ARGS.yaml}.yml"))
    data_kwargs, vocab_kwargs = dict(cfg.data), dict(cfg.vocab)
    use_bert = 'bert' in cfg.model['word_embedding']['name_or_path']

    # 如果用了BERT，要加载tokenizer
    if use_bert:
        tokenizer = BertTokenizer.from_pretrained(
            cfg.model['word_embedding']['name_or_path'], do_lower_case=False)
        print("I'm batman!  ", tokenizer.tokenize("I'm batman!"))  # [CLS] [SEP]
        data_kwargs['tokenizer'] = tokenizer
        vocab_kwargs['oov_token'] = tokenizer.unk_token
        vocab_kwargs['padding_token'] = tokenizer.pad_token
    else:
        tokenizer = None

    dataset = argparse.Namespace(**{
        k: UniversalPropositions.build(kind=k, **data_kwargs)
        for k in ('train', 'dev', 'test')
    })
    vocab = Vocabulary.from_data(dataset, **vocab_kwargs)

    # 将upostag词表人为的填充完整，数据里可能不全
    pos = [
        'X', 'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN',
        'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB',
        # "CONJ"  # de中非upos标准 CONJ
    ]
    vocab.set_field(pos, 'upostag')

    labels = [
            'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'AA', 'AM-COM', 'AM-LOC', 'AM-DIR', 'AM-GOL',
            'AM-MNR', 'AM-TMP', 'AM-EXT', 'AM-REC', 'AM-PRD', 'AM-PRP', 'AM-CAU',
            'AM-DIS', 'AM-MOD', 'AM-NEG', 'AM-DSP', 'AM-ADV', 'AM-ADJ', 'AM-LVB',
            'AM-CXN', 'AM-PRR', 'A1-DSP', 'V']  # AM-PRR A1-DSP 新的
    labels = labels + ['R-' + i for i in labels] + ['C-' + i for i in labels]
    labels = ['[PAD]'] + labels + ['O_`$', '_']
    vocab.set_field(labels, 'labels')

    if use_bert:
        # 若用BERT，则把words词表替换为BERT的
        vocab.token_to_index['words'] = tokenizer.vocab
        vocab.index_to_token['words'] = tokenizer.ids_to_tokens

    for part in vars(dataset).values():
        for ins in part:
            for i in ins['relations']:
                for j in ins['relations'][i]:
                    label = ins['relations'][i][j]
                    ins['relations'][i][j] = vocab.index_of(label, 'labels')

    run_once(cfg, dataset, vocab, device)

    loop(device)


def loop(device):
    output("start looping...")
    while True:
        time.sleep(0.05)
        a, b = torch.rand(233, 233, 233).to(device), torch.rand(233, 233, 233).to(device)
        c = a * b
        a = c


if __name__ == "__main__":
    set_seed()
    main()
