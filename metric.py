"""
"""

from typing import Set, Tuple, Dict, List
from collections import OrderedDict

from nmnlp.core.metrics import TaggingMetric, namespace_add


class RelationMetric(TaggingMetric):
    def __call__(self,
                 predictions: List[Dict],
                 relations: List[Dict]) -> OrderedDict:
        batch = self.counter_factory()

        predict_entities = self.get_entities(predictions)
        gold_entities = self.get_entities(relations)
        correct_entities = predict_entities & gold_entities

        batch.total += len(gold_entities)
        batch.positive += len(predict_entities)
        batch.correct += len(correct_entities)

        self.counter = namespace_add(self.counter, batch)

        return self.get_metric(batch)

    def get_entities(self, relations) -> Set[Tuple[int]]:
        entities = set()
        for i, ins in enumerate(relations):
            for p, arguments in ins.items():
                for a, label in arguments.items():
                    entities.add((i, p, a, label))
        return entities
