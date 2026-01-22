import torch
import torch.nn as nn
from transformers import MistralModel, MistralPreTrainedModel

class MistralForHumorMultiTask(MistralPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = MistralModel(config)
        hidden = config.hidden_size

        self.cls_head = nn.Linear(hidden, 2)
        self.reg_head = nn.Linear(hidden, 1)

        self.dropout = nn.Dropout(getattr(config, "classifier_dropout", 0.1))
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels_cls=None, labels_reg=None):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # pooled = last token hidden state (common for decoder-only classification)
        last_hidden = out.last_hidden_state  # [B, T, H]
        pooled = last_hidden[:, -1, :]
        pooled = self.dropout(pooled)

        logits_cls = self.cls_head(pooled)         # [B,2]
        pred_reg  = self.reg_head(pooled).squeeze(-1)  # [B]

        return {"logits_cls": logits_cls, "pred_reg": pred_reg}
