"""
Transition-based。
"""

from enum import Enum, unique
from typing import Tuple, List, Dict, Any, Iterable
from collections import defaultdict

import torch
from torch import Tensor
from torch.nn import Module, LSTMCell, Parameter, ParameterList, Dropout, Embedding, Linear
from torch.nn.modules.activation import Softmax, Tanh
from torch.nn.functional import log_softmax

from nmnlp.modules.linear import NonLinear


@unique
class Action(Enum):
    PRED_GEN: int = 2
    NO_PRED: int = 1

    LEFT_ARC: int = 3
    RIGHT_ARC: int = 4
    NO_LEFT_ARC: int = 5
    NO_RIGHT_ARC: int = 6

    SHIFT: int = 0
#    O_DELETE: int = 7  # 暂时没用上


class ActionHelper(Module):
    """
    完成一些Enum缺少的功能。
    """

    valid_actions = {
        # 如果上一步生成了predicate，则下一步判断是否为左右arc
        Action.PRED_GEN: [
            Action.LEFT_ARC, Action.NO_LEFT_ARC, Action.RIGHT_ARC,
            Action.NO_RIGHT_ARC
        ],

        # 如果上一步是shift，则下一步只可判断是否为predicate
        Action.SHIFT: [Action.PRED_GEN, Action.NO_PRED],
    }

    def __init__(self):
        super().__init__()
        directional_actions = [
            Action.LEFT_ARC, Action.NO_LEFT_ARC, Action.RIGHT_ARC,
            Action.NO_RIGHT_ARC
        ]
        directional_and_shift = directional_actions + [Action.SHIFT]
        # 如果上一步是方向的，下一步可以继续判断，或者转移
        for action in directional_actions:
            self.valid_actions[action] = directional_and_shift

        common_actions = [
            Action.PRED_GEN, Action.NO_PRED, Action.NO_LEFT_ARC,
            Action.NO_RIGHT_ARC
        ]
        # 如果上一步是NO-PRED，则按左右栈是否空来给定
        # 0. 左右都有
        self.valid_actions[(False, False)] = common_actions
        # 1. 左空右不，
        self.valid_actions[(True, False)] = [Action.RIGHT_ARC] + common_actions
        # 2. 左不右空
        self.valid_actions[(False, True)] = [Action.LEFT_ARC] + common_actions
        # 3. 左右皆空
        self.valid_actions[(True, True)] = [Action.SHIFT]

        self.masks = ParameterList()
        self.key_to_id = dict()
        for k, v in self.valid_actions.items():
            values = set(a.value for a in v)
            self.key_to_id[k] = len(self.masks)
            self.masks.append(Parameter(torch.tensor(
                [1 if i in values else 0 for i in range(len(Action))]
            ), requires_grad=False))

    def get_valid_actions(self,
                          action: Action = None,
                          empty_left: bool = True,
                          empty_right: bool = True) -> Tuple[List[Action], Tensor]:
        """
        Return valid actions list by previous action.
        """
        if action in self.valid_actions:
            return self.valid_actions[action], self.masks[self.key_to_id[action]]
        else:
            key = (empty_left, empty_right)
            return self.valid_actions[key], self.masks[self.key_to_id[key]]

    @staticmethod
    def make_oracle(length: int, relations: Dict[int, Dict[int, str]]) -> List[Action]:
        actions = [Action.SHIFT]

        for i in range(length):
            if i in relations:
                actions.append(Action.PRED_GEN)
                for j in range(1, max(i, length - i) + 1):
                    left = i - j
                    if left >= 0:
                        if left in relations[i]:
                            actions.append(Action.LEFT_ARC)
                        else:
                            actions.append(Action.NO_LEFT_ARC)
                    right = i + j
                    if right < length:
                        if right in relations[i]:
                            actions.append(Action.RIGHT_ARC)
                        else:
                            actions.append(Action.NO_RIGHT_ARC)
                actions.append(Action.SHIFT)
            else:
                actions.append(Action.NO_PRED)

        return actions


class PrototypeModule(Module):
    """
    基类
    """
    def __init__(self, dim: int):
        super().__init__()
        self.empty_embedding = Parameter(Tensor(1, dim), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.empty_embedding)

    def forward(self):
        pass

    def __len__(self) -> int:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def embedding(self) -> Tensor:
        raise NotImplementedError


class Buffer(PrototypeModule):
    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        self.hidden_states = list()
        self.length = -1
        self.index = -1

    def __len__(self) -> int:
        return self.length - self.index

    def clear(self):
        self.hidden_states = list()
        self.length = -1
        self.index = -1

    def embedding(self) -> Tensor:
        return self.hidden_states[self.index] if self.__len__() else self.empty_embedding

    def write(self, hidden_states: Iterable[Tensor]) -> None:
        self.hidden_states = hidden_states
        self.length = len(hidden_states)
        self.index = 0

    def read(self) -> Tuple[Tensor, int]:
        ccurrent_index, self.index = self.index, self.index + 1
        return self.hidden_states[ccurrent_index], ccurrent_index


class Keeper(PrototypeModule):
    def __init__(self, embedding_dim):
        super().__init__(embedding_dim)
        self.data: Tensor = None
        self.index: int = -1

    def __len__(self) -> int:
        return 0 if self.data is None else 1

    def clear(self) -> None:
        self.data, self.index = None, -1

    def embedding(self) -> Tensor:
        return self.empty_embedding if self.data is None else self.data

    def push(self, embedding: Tensor, index: int) -> None:
        self.data = embedding
        self.index = index

    def pop(self) -> Tuple[Tensor, int]:
        data, index = self.data, self.index
        self.data, self.index = None, -1
        return data, index


class StackLSTM(PrototypeModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 200,
                 bias: bool = True,
                 dropout: float = 0.):
        super().__init__(hidden_size)
        self.output_dim = hidden_size
        self.states: List[Tensor] = list()
        self.items = list()
        self.cell = LSTMCell(input_size, hidden_size, bias)
        self.dropout = Dropout(dropout)  # TODO:每一个instance的mask要一致

    def __len__(self) -> int:
        return len(self.states)

    def clear(self) -> None:
        self.states.clear()
        self.items.clear()

    def embedding(self) -> Tensor:
        return self.states[-1][0] if len(self.states) else self.empty_embedding

    def push(self, input: Tensor, item: Any) -> None:
        state = [self.dropout(t) for t in self.states[-1]] if len(self.states) else None
        state = self.cell(input, state)
        self.states.append(state)
        self.items.append(item)

    def pop(self) -> Tuple[Tensor, Any]:
        hx, _ = self.states.pop()
        return hx, self.items.pop()

    def is_empty(self) -> bool:
        return len(self.states) == 0

    def __getitem__(self, item) -> Tuple[Tensor, Any]:
        hx, _ = self.states[item]
        return hx, self.items[item]


class Attention(Module):
    def __init__(self, hidden_size, num_attention_heads, activation=Tanh()):
        super().__init__()  # TODO 以后可换成self attention
        self.hidden = NonLinear(hidden_size, num_attention_heads, activation)
        self.output = Linear(num_attention_heads, 1)

    def forward(self, input):
        hidden = self.hidden(input)
        scores = self.output(hidden)
        probs = torch.softmax(scores, dim=0)
        return probs


class DistributionAttention(Module):
    def __init__(self, in_features, out_features, num_attention_head, activation=Tanh()):
        super().__init__()
        self.output_dim = in_features + out_features
        self.attention = Attention(in_features + out_features, num_attention_head, activation)
        self.empty_embedding = Parameter(Tensor(1, out_features), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.empty_embedding)

    def forward(self, input, distributions):
        if distributions:  # 等效 len(distributions)
            inputs = input.expand(len(distributions), input.size(1))
            # (len, out_features)
            distributions = torch.cat(distributions, dim=0)
            attention_input = torch.cat([distributions, inputs], dim=-1)
            attention = self.attention(attention_input)
            # (len, out_features).T * (len,) -> (out_features,)
            attention = attention.t().matmul(distributions)
            hidden = torch.cat([attention, input], dim=-1)
        else:
            hidden = torch.cat([self.empty_embedding, input], dim=-1)
        return hidden


class RoleLabeler(Module):

    def __init__(self, input_size, stack_dim, action_stack_dim, role_num,
                 position_embedding_dim=64, num_attention_heads=80, activation=Tanh()):
        super().__init__()
        hidden_input_dim = input_size + stack_dim * 5 + action_stack_dim

        self.position_embedding = Embedding(512, position_embedding_dim)
        self.position_attention = Attention(position_embedding_dim * 2 + input_size, num_attention_heads, activation)
        self.distribution_attention = DistributionAttention(
            hidden_input_dim + input_size, role_num, num_attention_heads, activation)

        self.hidden = NonLinear(self.distribution_attention.output_dim, input_size, activation)

        self.output = Linear(input_size, role_num)

    def forward(self,
                embeddings_without_action,  # 为啥没有action
                distributions_role,
                hidden_states,
                head_id,
                last_id):
        attn_rep = self.position_aware_attn(hidden_states, head_id, last_id)

        state_embed = torch.cat([embeddings_without_action, attn_rep], dim=-1)

        rep = self.distribution_attention(state_embed, distributions_role)

        # rep = self.dropout(rep, 0.25)
        hidden = self.hidden(rep)
        scores = self.output(hidden)
        return scores

    def position_aware_attn(self, hidden_states, start1, start2):
        # 这个还没改
        ent1, end2 = start1, start2
        tri_pos_list = []
        ent_pos_list = []

        def relative_position(ent_start, ent_end, tok_idx, max_position_len=150):
            if ent_start <= tok_idx <= ent_end:
                return 0
            elif tok_idx < ent_start:
                return ent_start - tok_idx
            elif tok_idx > ent_end:
                return tok_idx - ent_end + max_position_len
            return None

        for i in range(len(hidden_states)):
            tri_pos_list.append(relative_position(start1, ent1, i))
            ent_pos_list.append(relative_position(start2, end2, i))

        tri_pos_emb = self.position_embedding(
            torch.tensor(tri_pos_list, device=hidden_states.device))
        ent_pos_emb = self.position_embedding(
            torch.tensor(ent_pos_list, device=hidden_states.device))

        att_input = torch.cat([hidden_states, tri_pos_emb, ent_pos_emb], dim=-1)

        attn_prob = self.position_attention(att_input)

        rep = attn_prob.t().matmul(hidden_states)
        return rep


class ActionGenerator(Module):
    def __init__(self, in_features: int, action_num: int, bias: bool = True,
                 activation=Tanh()):
        super().__init__()
        self.distribution_attention = DistributionAttention(in_features, action_num, 50)
        self.output_forward = NonLinear(in_features + action_num, action_num, bias, activation)

    def forward(self, input, distributions_head):
        hidden = self.distribution_attention(input, distributions_head)
        return self.output_forward(hidden)


class ShiftReduce(Module):
    """
    For ORL.
    """

    def __init__(self,
                 input_size,  # 输入特征纬度，如768
                 label_num,
                 role_num,
                 dropout=0.5,
                 embedding_dim=50,
                 action_stack_dim=60):
        super().__init__()
        stack_dim = input_size
        # 保存 head_forward 输出的当前 PRED 的 representation
        self.current_head = Keeper(stack_dim)

        # 保存本句所有 token 的输入 representation
        self.buffer = Buffer(input_size)

        self.left_candidates = StackLSTM(input_size, stack_dim, dropout)  # sigma left
        self.left_undecided = StackLSTM(input_size, stack_dim, dropout)  # alpha left
        self.right_candidates = StackLSTM(input_size, stack_dim, dropout)
        self.right_undecided = StackLSTM(input_size, stack_dim, dropout)
        self.output_stack = StackLSTM(input_size, action_stack_dim)
        self.action_stack = StackLSTM(embedding_dim, action_stack_dim, dropout)

        self.action_embedding = Embedding(len(Action), embedding_dim)
        self.label_embedding = Embedding(label_num, embedding_dim)

        self.action_helper = ActionHelper()
        self.action_generator = ActionGenerator(input_size, len(Action))

        hidden_input_dim = input_size + stack_dim * 5 + action_stack_dim * 2
        self.labeler = RoleLabeler(input_size, stack_dim, action_stack_dim, role_num)
        # 8个 prototype 的 embedding 作为输入
        self.hidden_forward = NonLinear(hidden_input_dim, input_size, activation=Tanh())

        # PRED 的输入 representation 和 embedding 作为输入
        self.head_forward = NonLinear(input_size + embedding_dim, stack_dim, activation=Tanh())

        self.distributions_head_forward = NonLinear(
            stack_dim * 2 + embedding_dim, len(Action), activation=Softmax(-1))
        self.distributions_role_forward = NonLinear(
            stack_dim * 2 + embedding_dim, role_num, activation=Softmax(-1))

        self.dropout = Dropout(p=dropout)

    def forward(self,
                hidden_states: Tensor,
                oracle_actions: List[Action] = None,
                relations: Dict[int, Dict[int, str]] = None):
        """
        一次一句。hidden_states 要符合序列实际长度，去掉padding。
        """
        loss_action, loss_label, prediction, actions = list(), list(), defaultdict(dict), list()
        distributions_head = []
        distributions_role = []

        self.buffer.write(hidden_states.split(1))

        step = 0
        while self.buffer or self.current_head:
            previous_action = None if self.action_stack.is_empty() else self.action_stack.items[-1]

            # based on parser state, get valid actions.
            # only a very small subset of actions are valid, as below.
            valid_actions, mask = self.action_helper.get_valid_actions(
                previous_action, self.left_candidates.is_empty(),
                self.right_candidates.is_empty()
            )

            # predicting action
            state_embedding = self.dropout(self.embeddings())
            hidden_representation = self.hidden_forward(state_embedding)

            # get action scores by hidden and ...
            scores = self.action_generator(hidden_representation, distributions_head).squeeze()
            scores = scores + (mask - 1) * 128
            log_probabilities = -log_softmax(scores, dim=0)

            if oracle_actions:
                action = oracle_actions[step]
                action_id = action.value
                if self.training and action not in valid_actions:
                    raise RuntimeError(f"Action {action} dose not in valid_actions {valid_actions}")
                loss_action.append(log_probabilities[action_id].unsqueeze(0))
            if not self.training:
                action_id = scores.argmax().item()
                action = Action(action_id)
                actions.append(action)

            # update the parser state according to the action.
            self.upadte_parser_state(**locals())
            self.action_stack.push(self.action_embedding(torch.tensor(
                action_id, device=scores.device)).unsqueeze(0), action)
            step += 1  # bump step

        self.clear()
        loss_label = torch.cat(loss_label).sum() if loss_label else torch.tensor(
            0.0, device=hidden_states.device)
        return torch.cat(loss_action).sum(), loss_label, prediction, actions

    def clear(self) -> None:
        for m in self.modules():
            if isinstance(m, PrototypeModule):
                m.clear()

    @staticmethod
    def upadte_parser_state(self,
                            action: Action,
                            hidden_states: Tensor,
                            prediction: Dict,
                            distributions_head: List,
                            distributions_role: List,
                            loss_label: List,
                            relations=None,
                            **kwargs: Any) -> None:
        if action == Action.NO_PRED:
            # 如果当前元素非head，压入左侧 候选栈，同时压入 输出栈。
            buffer = self.buffer.read()  # buffer 将指向下一元素
            self.left_candidates.push(*buffer)
            self.output_stack.push(*buffer)

            # 若还有，弹出 下一个 右侧候选元素
            if not self.right_candidates.is_empty():
                self.right_candidates.pop()

        elif action == Action.PRED_GEN:
            # 若判断为head，
            hx, index = self.buffer.read()

            # 填满 sigma right，从len到index。即所有还未判断的元素
            for i in range(self.buffer.length - 1, index, -1):
                self.right_candidates.push(self.right_candidates.empty_embedding, i)

            head_embedding = self.label_embedding(
                torch.tensor(1, device=hx.device)).unsqueeze(0)
            head_representation = self.head_forward(
                torch.cat((hx, head_embedding), dim=-1))
            self.current_head.push(head_representation, index)

            prediction[index] = dict()

        elif action == Action.NO_LEFT_ARC:
            # 若 head 左侧元素不是 arc，则弹出并压入 alpha
            if not self.left_candidates.is_empty():
                self.left_undecided.push(*self.left_candidates.pop())

        elif action == Action.NO_RIGHT_ARC:
            # 类比上一个 case
            if not self.right_candidates.is_empty():
                self.right_undecided.push(*self.right_candidates.pop())

        elif action == Action.SHIFT:
            # 当前 head 完成，将 左侧 所有已判断元素 和 本身 弹出并压入 左候选栈
            while not self.left_undecided.is_empty():
                self.left_candidates.push(*self.left_undecided.pop())

            if self.current_head:
                self.left_candidates.push(*self.current_head.pop())

            # 将 右侧 所有已判断元素 弹出并压入 候选栈
            while not self.right_undecided.is_empty():
                self.right_candidates.push(*self.right_undecided.pop())

            # 弹出 下一个 候选元素
            if not self.right_candidates.is_empty():
                self.right_candidates.pop()

        elif action in (Action.LEFT_ARC, Action.RIGHT_ARC):
            if action == Action.RIGHT_ARC:
                sigma_rnn, alpha_rnn = self.right_candidates, self.right_undecided
            else:
                sigma_rnn, alpha_rnn = self.left_candidates, self.left_undecided

            head_index, head_embedding = self.current_head.index, self.current_head.embedding()

            if sigma_rnn.is_empty():
                last_embedding, last_index = head_embedding, head_index
            else:
                last_embedding, last_index = sigma_rnn.pop()  # TODO 万一空的

            label_scores = self.labeler(self.embeddings(action=False),
                                        distributions_role,
                                        hidden_states,
                                        head_index,
                                        last_index).squeeze()

            if relations:
                if head_index in relations:
                    label = relations[head_index].get(last_index, 0)
                else:
                    label = 0  # NULL 标签的id
                loss = -log_softmax(label_scores, dim=0)[label]
                loss_label.append(loss.unsqueeze(0))
            else:
                label = label_scores.max()[1]

            prediction[head_index][last_index] = label

            alpha_rnn.push(last_embedding, last_index)

            # distributions
            role_embedding = self.label_embedding(torch.tensor(
                label, device=head_embedding.device)).unsqueeze(0)
            distributions_head.append(
                self.distributions_head_forward(torch.cat(
                    [last_embedding, head_embedding, role_embedding], dim=1))
            )
            distributions_role.append(
                self.distributions_role_forward(torch.cat(
                    [last_embedding, head_embedding, role_embedding], dim=1))
            )

        else:
            raise RuntimeError(f"Unknown action: {action}")

    def embeddings(self, action=True):
        embeddings = [
            self.current_head.embedding(),
            self.buffer.embedding(),
            self.left_candidates.embedding(),
            self.left_undecided.embedding(),
            self.right_candidates.embedding(),
            self.right_undecided.embedding(),
            self.output_stack.embedding(),
        ]
        if action:
            embeddings.append(self.action_stack.embedding())
        return torch.cat(embeddings, dim=1)
