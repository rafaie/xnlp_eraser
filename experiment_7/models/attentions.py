from turtle import forward
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F

from allennlp.modules.transformer.attention_module import AttentionModule, SelfAttention


class SimpCrossAttention:
    def __init__(
        self
    ):
        self.cross_att = AttentionModule(
            is_cross_attention=True, is_decoder=True)

    def forward(self, query_states, attention_mask,
                source_states, source_attention_mask):
        att = self.cross_att(query_states=query_states,
                             attention_mask=attention_mask,
                             source_states=source_states,
                             source_attention_mask=source_attention_mask)

        return att


class CrossSelfAttention:
    def __init__(
        self, self_att_dropout: int = 0
    ):
        self.cross_att = AttentionModule(
            is_cross_attention=True, is_decoder=True)
        self.self_att = SelfAttention(hidden_size=self.cross_att.hidden_size,
                                      num_attention_heads=self.cross_att.num_attention_heads,
                                      is_cross_attention=False,
                                      dropout=self_att_dropout)

    def forward(self, query_states, attention_mask,
                source_states, source_attention_mask):
        att = self.cross_att(query_states=query_states,
                             attention_mask=attention_mask,
                             source_states=source_states,
                             source_attention_mask=source_attention_mask)
        att = self.self_att(att.hidden_states)

        return att


class CrossModality:
    def __init__(
        self, self_att_dropout: int = 0
    ):
        self.cross_att1 = AttentionModule(
            is_cross_attention=True, is_decoder=True)
        self.cross_att2 = AttentionModule(
            is_cross_attention=True, is_decoder=True)
        self.cross_att3 = AttentionModule(
            is_cross_attention=True, is_decoder=True)
        self.self_att1 = SelfAttention(hidden_size=self.cross_att1.hidden_size,
                                       num_attention_heads=self.cross_att1.num_attention_heads,
                                       is_cross_attention=False,
                                       dropout=self_att_dropout)
        self.self_att2 = SelfAttention(hidden_size=self.cross_att2.hidden_size,
                                       num_attention_heads=self.cross_att2.num_attention_heads,
                                       is_cross_attention=False,
                                       dropout=self_att_dropout)
        self.self_att3 = SelfAttention(hidden_size=self.cross_att3.hidden_size,
                                       num_attention_heads=self.cross_att3.num_attention_heads,
                                       is_cross_attention=False,
                                       dropout=self_att_dropout)

    def forward(self, query_states, attention_mask,
                source_states, source_attention_mask):
        att1 = self.cross_att1(query_states=query_states,
                             attention_mask=attention_mask,
                             source_states=source_states,
                             source_attention_mask=source_attention_mask)
        att1 = self.self_att1(att1.hidden_states)


        att2 = self.cross_att2(query_states=source_states,
                             attention_mask=source_attention_mask,
                             source_states=query_states,
                             source_attention_mask=attention_mask)
        att2 = self.self_att2(att2.hidden_states)

        att3 = self.cross_att3(query_states=att1.hidden_states,
                             source_states=att2.hidden_states)
        att3 = self.self_att3(att3.hidden_states)

        return att3
