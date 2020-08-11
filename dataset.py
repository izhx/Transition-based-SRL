"""
from
"""

import glob
from typing import Set, Any

from nmnlp.core import DataSet
from nmnlp.data.conll import conll_like_file

from transition import ActionHelper


class UniversalPropositions(DataSet):
    index_fields = ("words", "upostag")

    @classmethod
    def build(cls,
              data_dir: str,
              kind: str = 'train',
              lang: str = 'en',
              tokenizer: Any = None,
              pretrained_fields: Set[str] = ()):
        path = glob.glob(f"{data_dir}/UP_*/{lang}*{kind}.conllu")[0]
        dataset = cls(list(), pretrained_fields)

        for sentence in conll_like_file(path):
            if len(sentence[0]) > 10:
                dataset.data.append(
                    cls.text_to_instance(sentence, lang, tokenizer))

        return dataset

    @classmethod
    def text_to_instance(cls, sentence, lang, tokenizer):
        sense_col = 10 if lang == 'en' else 9  # 暂时只考虑英德
        ins = {'words': list(), 'upostag': list(), 'text': list()}
        pieces, predicate_ids = dict(), list()

        padding_len = len(sentence[0]) - 4
        sentence.insert(0, [0, '[CLS]', '[CLS]', 'X'] + ['_'] * padding_len)
        sentence.append([len(sentence), '[SEP]', '[SEP]', 'X'] + ['_'] * padding_len)

        for i, row in enumerate(sentence):
            ins['upostag'].append(row[3])
            if row[sense_col] != '_':
                predicate_ids.append(i)
            word = row[1]
            ins['text'].append(word)
            if tokenizer is not None:
                piece = tokenizer.tokenize(word)
                if len(piece) > 0:
                    ins['words'].append(piece[0])
                    if len(piece) > 1:
                        pieces[i] = [tokenizer.vocab[p] for p in piece]
            else:
                ins['words'].append(word.lower())

        ins['word_pieces'] = pieces

        # for row in sentence:
        #     print(len(row), '\t', '\t'.join(row[sense_col:]))
        # print('\n')

        def label_map(label):
            # if label in ('V', 'C-V', 'R-V'):
            #     return '_'  # 待确定
            if 'ARG' in label:
                return label.replace('ARG', 'A')
            return label

        # _relations = {p: {
        #     i: label_map(line[col])
        #     for i, line in enumerate(sentence) if line[col] != '_' and i != p
        #     } for col, p in enumerate(predicate_ids, start=sense_col + 1)}

        relations = dict()
        for col, p in enumerate(predicate_ids, start=sense_col + 1):
            relations[p] = dict()
            for i, line in enumerate(sentence):
                if line[col] != '_' and i != p:
                    relations[p][i] = label_map(line[col])

        ins['relations'] = relations
        ins['oracle'] = ActionHelper.make_oracle(len(sentence), relations)
        return ins
