import torch
from torch import nn
from transformers import AutoModel
from neural_networks.custom_layers.dropout import SpatialDropout
from neural_networks.custom_layers.masking import Camouflage
from configuration import Configuration
class Model(nn.Module):
    def __init__(self,n_classes=4654,dropout_rate=0.5):
        super(Model, self).__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.dropout = SpatialDropout(drop=dropout_rate)
        self.masking = Camouflage(mask_value=0)
        self.Wa = nn.Linear(768, n_classes,bias=False)#[768,4654]
        self.Wo = nn.Linear(768, n_classes,bias=False)

    def label_wise_attention(self,value1,value2):
        def dot_product(x, kernel):
            # 4654,512 [768 512]
            return torch.transpose(torch.squeeze((x @ torch.unsqueeze(kernel, dim=-1)), dim=-1), 0, 1)
        doc_repi, ai = value1, value2  # 应分成[512,4654] [512,4654]
        rg = len(ai)
        ai_tmp = torch.softmax(torch.transpose(ai[0], 0, 1), dim=-1)  # 得注意力分数dim应为求最里面一维的softmax
        tmp = dot_product(ai_tmp, torch.transpose(doc_repi[0], 0, 1)) #768

        tmp = torch.unsqueeze(tmp, dim=0)#1××
        for i in range(1, rg):
            ai_itmp = torch.softmax(torch.transpose(ai[i], 0, 1), dim=0)  # 得注意力分数
            label_aware_doc_rep_tmp = dot_product(ai_itmp, torch.transpose(doc_repi[i], 0, 1))
            label_aware_doc_rep_tmp = torch.unsqueeze(label_aware_doc_rep_tmp, dim=0)
            tmp = torch.cat([tmp, label_aware_doc_rep_tmp])
        return tmp  # torch.Size([32, 768, 4654])

    def forward(self,x_batch):#32 512
        bert_output = self.bert(x_batch)#32 512 768
        inner_out = self.dropout(bert_output[0])
        x = self.masking(inputs=[inner_out, bert_output[0]])#32 512 768
        a = self.Wa(x)##a[32,512,4654]
        label_aware_doc_reprs = self.label_wise_attention(x, a)
        label_aware_doc_reprs = self.Wo(label_aware_doc_reprs)
        label_aware_doc_reprs = torch.sum(label_aware_doc_reprs, dim=-1)
        label_aware_doc_reprs = torch.sigmoid(label_aware_doc_reprs)

        return label_aware_doc_reprs


