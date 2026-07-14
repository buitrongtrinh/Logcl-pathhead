import math

import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class ConvTransR(torch.nn.Module):
    """Decoder tích chập cho dự đoán quan hệ: chấm điểm (s, ?, o) trên mọi quan hệ."""
    def __init__(self, num_relations, embedding_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3, use_bias=True):
        super(ConvTransR, self).__init__()
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        # kernel_size lẻ nên padding = floor(kernel_size / 2) giữ nguyên độ dài chuỗi
        self.conv1 = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
                               padding=int(math.floor(kernel_size / 2)))
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_relations*2)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(embedding_dim)

    def forward(self, embedding, emb_rel, triplets, nodes_id=None, mode="train", negative_rate=0):
        e1_embedded_all = embedding
        batch_size = len(triplets)
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        e2_embedded = e1_embedded_all[triplets[:, 2]].unsqueeze(1)
        stacked_inputs = torch.cat([e1_embedded, e2_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, emb_rel.transpose(1, 0))
        return x


class ConvTransE(torch.nn.Module):
    """Decoder tích chập cho dự đoán thực thể: chấm điểm (s, r, ?) trên mọi thực thể."""

    def __init__(self, num_entities, embedding_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3, use_bias=True):
        super(ConvTransE, self).__init__()
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        # kernel_size lẻ nên padding = floor(kernel_size / 2) giữ nguyên độ dài chuỗi
        self.conv1 = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
                               padding=int(math.floor(kernel_size / 2)))
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(embedding_dim)

    def forward(self, embedding, emb_rel, triplets, his_emb, pre_weight, pre_type, partial_embeding=None):
        batch_size = len(triplets)
        if pre_type =="all":
            # Trộn embedding tiến hoá cục bộ với embedding lịch sử toàn cục theo pre_weight
            e1_embedded_all = F.tanh(embedding)
            embedded_his = F.tanh(his_emb)
            e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
            e1_his_embedded = embedded_his[triplets[:, 0]].unsqueeze(1)
            e1_embed = pre_weight*e1_embedded + (1-pre_weight)*e1_his_embedded
        rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1)
        stacked_inputs = torch.cat([e1_embed, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        cl_x = x
        if partial_embeding is None:
            x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        else:
            x = torch.mm(x, partial_embeding.transpose(1, 0))
        return x, cl_x

    def forward_slow(self, embedding, emb_rel, triplets):
        """Chấm điểm từng triple một (chỉ điểm của thực thể đích, không xét mọi ứng viên)."""
        e1_embedded_all = F.tanh(embedding)
        batch_size = len(triplets)
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        e2_embedded = e1_embedded_all[triplets[:, 2]]
        score = torch.sum(torch.mul(x, e2_embedded), dim=1)
        pred = score
        return pred