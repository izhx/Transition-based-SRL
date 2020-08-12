from typing import Any, Dict, List
from collections import defaultdict

import torch
from torch import Tensor
from torch.nn import Module

from nmnlp.core import Vocabulary
from nmnlp.embedding import build_word_embedding
from nmnlp.modules.dropout import WordDropout
from nmnlp.modules.encoder import build_encoder
from nmnlp.modules.linear import NonLinear

from metric import RelationMetric
from transition import Action, ShiftReduce


class TransitionModel(Module):
    def __init__(self,
                 vocab: Vocabulary,
                 word_embedding: Dict,
                 encoder: Dict,
                 parser: Dict,
                 transform_dim: int = 0,
                 dropout: float = 0.5,
                 input_namespace: str = 'words',
                 label_namespace: str = 'labels'):
        super().__init__()
        self.word_embedding = build_word_embedding(num_embeddings=vocab.size_of(input_namespace),
                                                   vocab=vocab,
                                                   dropout=dropout,
                                                   **word_embedding)
        feat_dim: int = self.word_embedding.output_dim

        if transform_dim > 0:
            self.word_transform = NonLinear(feat_dim, transform_dim)
            feat_dim: int = transform_dim
        else:
            self.word_transform = None

        if encoder is not None:
            self.encoder = build_encoder(feat_dim, dropout=dropout, **encoder)
            feat_dim = self.encoder.output_dim
        else:
            self.encoder = None

        self.parser = ShiftReduce(feat_dim,
                                  vocab.size_of(label_namespace),
                                  vocab.size_of(label_namespace) - 1,
                                  dropout=dropout,
                                  **parser)

        self.word_dropout = WordDropout(dropout)
        self.vocab = vocab
        self.metric = RelationMetric(vocab.index_of('_', label_namespace))

    def forward(self,
                words: Tensor = None,
                mask: Tensor = None,
                lengths: Tensor = None,
                relations: List[Dict] = None,
                oracle_actions: List[Action] = None,
                **kwargs: Any) -> Dict[str, Any]:
        feat: Tensor = self.word_embedding(words, mask=mask, **kwargs)
        if self.word_transform is not None:
            feat = self.word_transform(feat)
        feat = self.word_dropout(feat)

        if self.encoder is not None:
            feat = self.encoder(feat, lengths, **kwargs)
        feat = self.word_dropout(feat)

        output_dict = defaultdict(list)

        _relations = relations if relations else [None] * len(lengths)
        _oracle_actions = oracle_actions if relations else [None] * len(lengths)

        for hidden, relation, actions, length in zip(
                feat, _relations, _oracle_actions, lengths):
            parser_output = self.parser(hidden[:length], actions, relations)
            output_dict['loss_action'].append(parser_output[0].unsqueeze(0))
            output_dict['loss_label'].append(parser_output[1].unsqueeze(0))
            output_dict['prediction'].append(parser_output[2])  # relation to labels
            output_dict['actions'].append(parser_output[3])

        if relations:
            output_dict['loss'] = torch.cat(
                output_dict['loss_action']).sum() + torch.cat(
                    output_dict['loss_label']).sum()
            if not self.training:
                output_dict['metric'] = self.metric(output_dict['prediction'], relations)

        return output_dict

    def after_collate_batch(self, input_dict, batch, kwargs):
        relations, oracle_actions = list(), list()
        for ins in batch:
            relations.append(ins['relations'])
            oracle_actions.append(ins['oracle'])
        input_dict['relations'] = relations
        input_dict['oracle_actions'] = oracle_actions
        return input_dict, batch
