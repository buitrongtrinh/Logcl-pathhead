import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer, RGAT, UnionRGCNLayer2, UnionRGATLayer, CompGCNLayer
from src.model import BaseRGCN
from src.decoder import ConvTransE, ConvTransR

class MLPLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPLinear, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.act = nn.LeakyReLU(0.2)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, x):
        x = self.act(F.normalize(self.linear1(x), p=2, dim=1))
        x = self.act(F.normalize(self.linear2(x), p=2, dim=1))

        return x

class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, self_loop=self.self_loop, dropout=self.dropout, skip_connect=sc, rel_emb=self.rel_emb)
        elif self.encoder_name == "kbat":
            return UnionRGATLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, self_loop=self.self_loop, dropout=self.dropout, skip_connect=sc, rel_emb=self.rel_emb)
        elif self.encoder_name == "compgcn":
            return CompGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.opn, self.num_bases,
                            activation=act, self_loop=self.self_loop, dropout=self.dropout, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn" or self.encoder_name == "kbat" or self.encoder_name == "compgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')


class RGCNCell2(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer2(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')


class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0, 
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1,pre_weight=0.7, discount=0, angle=0, use_static=False, pre_type = 'short', 
                 use_cl= False, temperature=0.007, entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu = 0, analysis=False, use_path=False, path_dim=32, path_layers=2,
                 path_batch_size=128, path_level=2):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.pre_weight = pre_weight
        self.discount = discount
        self.use_static = use_static
        self.pre_type = pre_type
        self.use_cl = use_cl
        self.temp =temperature
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.use_path = use_path
        self.emb_rel = None
        self.gpu = gpu

        self.w1 = nn.Linear(self.h_dim*2, self.h_dim)
        
        self.w2 = nn.Linear(self.h_dim, self.h_dim)
        self.w3 = nn.Linear(self.h_dim, self.h_dim)
        self.w4 = nn.Linear(self.h_dim*2, self.h_dim)
        self.w5 = nn.Linear(self.h_dim, self.h_dim)
        self.w6 = nn.Linear(self.h_dim,self.h_dim)
        self.w7 = nn.Linear(self.h_dim, self.h_dim)
        self.w_cl = nn.Linear(self.h_dim*2, self.h_dim)

        self.weight_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))

        self.weight_1 = nn.Linear(self.h_dim*2, self.h_dim)
        self.weight_2 = nn.Linear(self.h_dim*2, self.h_dim)
        self.bias = nn.Parameter(torch.zeros(1))

        self.weight_3 = nn.Linear(self.h_dim, 1)
        self.weight_4 = nn.Linear(self.h_dim, 1)
        self.bias_r = nn.Parameter(torch.zeros(1))

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        # Path head (bật bằng --use-path): chấm điểm ứng viên bằng lan truyền
        # thông điệp 2 bước trên đồ thị lịch sử, rồi cộng vào logits của LogCL
        # với hệ số học được gamma (khởi tạo 0 nên không ảnh hưởng lúc đầu).
        self.path_level = path_level
        if self.use_path:
            if path_dim <= 0 or path_layers <= 0 or path_batch_size <= 0:
                raise ValueError("Path head dimensions, layers, and batch size must be positive")
            if path_level not in (1, 2):
                raise ValueError("path_level must be 1 (shallow) or 2 (deep fusion)")
            self.path_dim = path_dim
            self.path_L = path_layers
            self.path_batch_size = path_batch_size
            self.path_rel = nn.Linear(self.h_dim, self.path_dim)
            self.path_boundary = nn.Linear(self.h_dim, self.path_dim)
            self.path_upd = nn.ModuleList([
                nn.Linear(self.path_dim * 2, self.path_dim)
                for _ in range(self.path_L)
            ])
            self.path_out = nn.Linear(self.path_dim, 1)
            self.path_gamma = nn.Parameter(torch.zeros(1))
            # Mức 2: chấm điểm path theo truy vấn [h_s ; hr_r] đã tiến hoá của LogCL
            if self.path_level >= 2:
                self.path_q = nn.Linear(self.h_dim * 2, self.path_dim)
            print("[path] enabled: dim=%d, layers=%d, level=%d" % (self.path_dim, self.path_L, self.path_level))

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)

        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()

        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)
        
        self.his_rgcn_layer = RGCNCell2(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)
        
        self.rgat_layer = RGAT(self.h_dim, self.h_dim, activation=F.rrelu, dropout=dropout, self_loop=True)
        self.projection_model = MLPLinear(self.h_dim, self.h_dim)

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)   

        self.pre_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.pre_gate_weight, gain=nn.init.calculate_gain('relu'))

        # GRU tiến hoá embedding thực thể/quan hệ qua các snapshot
        self.entity_cell = nn.GRUCell(self.h_dim, self.h_dim)
        self.relation_cell = nn.GRUCell(self.h_dim, self.h_dim)

        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError

    def _merge_path_edges(self, history_graphs, device):
        """Gộp cạnh của mọi đồ thị lịch sử thành một danh sách (src, dst, rel) chung."""
        edge_parts = []
        for graph in history_graphs:
            src, dst = graph.edges()
            edge_parts.append((
                src.to(device=device, dtype=torch.long),
                dst.to(device=device, dtype=torch.long),
                graph.edata['type'].to(device=device, dtype=torch.long),
            ))

        if not edge_parts:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty, empty

        return (
            torch.cat([part[0] for part in edge_parts]),
            torch.cat([part[1] for part in edge_parts]),
            torch.cat([part[2] for part in edge_parts]),
        )

    def path_head_scores(self, queries, history_graphs, relation_embedding=None,
                         merged_edges=None, ent_emb=None):
        """Tính điểm path 2 bước cho mọi thực thể ứng viên của từng truy vấn.

        Khởi tạo biểu diễn tại thực thể nguồn từ embedding quan hệ truy vấn,
        rồi lan truyền path_L vòng trên đồ thị lịch sử đã gộp cạnh.
        - Mức 1: relation_embedding = emb_rel tĩnh, ent_emb = None,
          điểm = path_out(H).
        - Mức 2: truyền quan hệ (hr) và thực thể (h) đã tiến hoá,
          điểm = path_q([h_s ; hr_r]) · H (chấm điểm có điều kiện theo truy vấn).
        """
        relation_embedding = self.emb_rel if relation_embedding is None else relation_embedding
        device = relation_embedding.device
        queries = queries.to(device=device, dtype=torch.long)
        batch_size = queries.size(0)

        if merged_edges is None:
            merged_edges = self._merge_path_edges(history_graphs, device)
        src, dst, rel = merged_edges

        relation = F.normalize(relation_embedding, p=2, dim=1)
        edge_relation = self.path_rel(relation[rel])
        boundary = self.path_boundary(relation[queries[:, 1]])

        hidden = relation.new_zeros(batch_size, self.num_ents, self.path_dim)
        batch_index = torch.arange(batch_size, device=device)
        source = queries[:, 0]
        hidden[batch_index, source] = boundary

        in_degree = torch.bincount(dst, minlength=self.num_ents).clamp_min(1)
        in_degree = in_degree.to(dtype=hidden.dtype).view(1, self.num_ents, 1)
        source_index = batch_index * self.num_ents + source

        for update in self.path_upd:
            message = hidden[:, src, :] * edge_relation.unsqueeze(0)
            aggregate = torch.zeros_like(hidden)
            aggregate.index_add_(1, dst, message)
            aggregate = aggregate / in_degree

            aggregate_weight = update.weight[:, :self.path_dim]
            hidden_weight = update.weight[:, self.path_dim:]
            hidden = F.linear(aggregate, aggregate_weight, update.bias) + \
                     F.linear(hidden, hidden_weight)
            hidden = F.relu(hidden)
            hidden = torch.index_add(
                hidden.view(-1, self.path_dim), 0, source_index, boundary
            ).view(batch_size, self.num_ents, self.path_dim)

        # Mức 2: chấm điểm theo truy vấn [h_s ; hr_r] thay cho path_out của mức 1
        if self.path_level >= 2 and ent_emb is not None:
            q_path = self.path_q(torch.cat([ent_emb[source], relation[queries[:, 1]]], dim=-1))
            return torch.einsum('bp,bnp->bn', q_path, hidden)
        return self.path_out(hidden).squeeze(-1)

    def _fuse_path(self, logcl_score, queries, history_graphs, ent_emb=None, rel_emb=None):
        """Chuẩn hoá điểm path rồi cộng vào logits LogCL với hệ số học được gamma."""
        if not self.use_path or not history_graphs:
            return logcl_score

        merged_edges = self._merge_path_edges(history_graphs, self.emb_rel.device)
        if merged_edges[0].numel() == 0:
            return logcl_score

        # Mức 2 đọc quan hệ/thực thể đã tiến hoá; mức 1 dùng emb_rel tĩnh, không cần ent_emb
        if self.path_level >= 2:
            relation_embedding = rel_emb if rel_emb is not None else self.emb_rel
            entity_embedding = ent_emb
        else:
            relation_embedding = self.emb_rel
            entity_embedding = None

        def score_chunk(chunk_queries, relation_embedding_):
            return self.path_head_scores(
                chunk_queries, history_graphs, relation_embedding_, merged_edges,
                entity_embedding,
            )

        # Chia truy vấn thành từng khối nhỏ + gradient checkpointing để tiết kiệm bộ nhớ GPU
        path_chunks = []
        for query_chunk in queries.split(self.path_batch_size, dim=0):
            if self.training and torch.is_grad_enabled():
                chunk_score = checkpoint(score_chunk, query_chunk, relation_embedding, use_reentrant=False)
            else:
                chunk_score = score_chunk(query_chunk, relation_embedding)
            path_chunks.append(chunk_score)

        path_score = torch.cat(path_chunks, dim=0)
        mean = path_score.mean(dim=1, keepdim=True)
        std = path_score.std(dim=1, keepdim=True, unbiased=False) + 1e-6
        path_score = (path_score - mean) / std
        return logcl_score + self.path_gamma * path_score

    def forward(self,sub_graph,T_idx, query_mask, g_list, static_graph, use_cuda):

        if self.use_static:
            # Ràng buộc tĩnh: embedding thực thể và embedding từ cùng đi qua RGCN trên đồ thị tĩnh
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None

        # Biểu diễn lịch sử toàn cục: chạy GCN trên đồ thị con lịch sử đã lấy mẫu,
        # rồi nhấn mạnh các thực thể liên quan truy vấn bằng attention theo query_mask
        self.his_ent, subg_index = self.all_GCN(self.h, sub_graph,use_cuda)
        his_r_emb = F.normalize(self.emb_rel)
        his_att = F.softmax(self.w5(query_mask+ self.his_ent),dim=1)
        his_emb = his_att*self.his_ent
        his_emb = F.normalize(his_emb)

        history_embs = []
        att_embs = []
        his_temp_embs =[]
        his_rel_embs =[]
        if self.pre_type=="all":
            # Tiến hoá embedding qua từng snapshot lịch sử (cũ -> mới)
            for i, g in enumerate(g_list):
                g = g.to(self.gpu)
                # Mã hoá khoảng cách thời gian tới snapshot đích rồi trộn vào embedding thực thể
                t2 = len(g_list)-i+1
                h_t = torch.cos(self.weight_t2 * t2 + self.bias_t2).repeat(self.num_ents,1)
                self.h =self.w4(torch.concat([self.h,h_t],dim=1))
                # Embedding quan hệ tại snapshot: trung bình embedding các thực thể kề quan hệ đó
                temp_e = self.h[g.r_to_e]
                x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_rels * 2, self.h_dim).float()
                for span, r_idx in zip(g.r_len, g.uniq_r):
                    x = temp_e[span[0]:span[1],:]
                    x_mean = torch.mean(x, dim=0, keepdim=True)
                    x_input[r_idx] = x_mean
                x_input = self.emb_rel + x_input
                current_h = self.rgcn.forward(g, self.h, [self.emb_rel, self.emb_rel])
                current_h = F.normalize(current_h) if self.layer_norm else current_h
                att_e = F.softmax(self.w2(query_mask+current_h),dim=1)

                # GRU cập nhật trạng thái thực thể; trạng thái ra là đầu vào của snapshot kế tiếp
                if i == 0:
                    self.h_0 = self.entity_cell(current_h, self.h)
                    self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
                else:
                    self.h_0 = self.entity_cell(current_h, self.h_0)
                    self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
                # Cổng thời gian trộn embedding quan hệ theo snapshot với embedding quan hệ tĩnh
                time_weight = F.sigmoid(torch.mm(x_input, self.time_gate_weight) + self.time_gate_bias)
                self.hr = time_weight * x_input + (1-time_weight) * self.emb_rel
                self.hr = F.normalize(self.hr) if self.layer_norm else self.hr
                history_embs.append(self.h_0)
                his_rel_embs.append(self.hr)
                his_temp_embs.append(self.h_0)
                self.h = self.h_0
                att_emb = att_e*self.h_0 
                att_embs.append(att_emb.unsqueeze(0))
            att_ent = torch.mean(torch.concat(att_embs,dim=0),dim=0)
            att_ent = F.normalize(att_ent)
            history_emb=  att_ent+history_embs[-1]
            history_emb = F.normalize(history_emb) if self.layer_norm else history_emb
        else:
            self.hr = None
            history_emb = None

        return history_emb, static_emb, self.hr, his_emb, his_r_emb,his_temp_embs,his_rel_embs


    def predict(self,que_pair, sub_graph,T_id, test_graph, num_rels, static_graph, test_triplets, use_cuda):
        with torch.no_grad():
            all_triples = test_triplets

            # Dựng query_mask: mỗi thực thể truy vấn mang biểu diễn [embedding thực thể ;
            # trung bình embedding các quan hệ nó tham gia], các thực thể khác bằng 0
            uniq_e = que_pair[0]
            r_len = que_pair[1]
            r_idx = que_pair[2]
            temp_r = self.emb_rel[r_idx]
            e_input = torch.zeros(self.num_ents, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_ents, self.h_dim).float()
            for span, e_idx in zip(r_len, uniq_e):
                x = temp_r[span[0]:span[1],:]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                e_input[e_idx] = x_mean

            query_mask = torch.zeros((self.num_ents,self.h_dim)).to(self.gpu) if use_cuda else torch.zeros(1)
            e1_emb = self.dynamic_emb[uniq_e]
            rel_emb = e_input[uniq_e]
            query_emb = self.w1(torch.concat([e1_emb,rel_emb],dim=1))
            query_mask[uniq_e] = query_emb

            embedding, _, r_emb, his_emb, his_r_emb,_,_ = self.forward(sub_graph,T_id, query_mask,test_graph, static_graph, use_cuda)

            if self.pre_type == "all":
                scores_ob,_= self.decoder_ob.forward( embedding,r_emb, all_triples,  his_emb, self.pre_weight, self.pre_type)
                # Cộng điểm path vào logits trước khi softmax
                scores_ob = self._fuse_path(scores_ob, all_triples, test_graph, embedding, r_emb)
                score_seq = F.softmax(scores_ob, dim=1)
                score_en =score_seq
            scores_en = torch.log(score_en)
            return all_triples, scores_en


    def get_loss(self,que_pair, sub_graph,T_idx, glist, triples, static_graph, use_cuda):
        """Tính các thành phần loss cho một snapshot: (loss_ent, loss_rel, loss_static, loss_cl)."""
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_cl = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        all_triples = triples

        # Dựng query_mask như trong predict, nhưng embedding thực thể có thêm mã hoá thời gian
        uniq_e = que_pair[0]
        r_len = que_pair[1]
        r_idx = que_pair[2]
        temp_r = self.emb_rel[r_idx]
        e_input = torch.zeros(self.num_ents, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_ents, self.h_dim).float()
        for span, e_idx in zip(r_len, uniq_e):
            x = temp_r[span[0]:span[1],:]
            x_mean = torch.mean(x, dim=0, keepdim=True)
            e_input[e_idx] = x_mean

        query_mask = torch.zeros((self.num_ents,self.h_dim)).to(self.gpu) if use_cuda else torch.zeros(1)
        t1 = torch.tensor(T_idx).cuda().to(self.gpu)
        q_t = torch.cos(self.weight_t2 * 0 + self.bias_t2).repeat(self.num_ents,1)
        qe_emb = self.w4(torch.concat([self.dynamic_emb,q_t],dim=1))
        
        e1_emb = qe_emb[uniq_e]

        rel_emb = e_input[uniq_e] 
        query_emb = self.w1(torch.concat([e1_emb,rel_emb],dim=1)) 
        query_mask[uniq_e] = query_emb

        embedding, static_emb, r_emb, his_emb, his_r_emb, his_temp_embs, his_rel_embs = self.forward(sub_graph, T_idx, query_mask, glist, static_graph, use_cuda)

        if self.pre_type == "all":
            scores_ob, _= self.decoder_ob.forward(embedding, r_emb, all_triples, his_emb,self.pre_weight, self.pre_type)
            # Cộng điểm path vào logits trước khi softmax
            scores_ob = self._fuse_path(scores_ob, all_triples, glist, embedding, r_emb)
            score_seq = F.softmax(scores_ob, dim=1)
            score_en = score_seq

        scores_en = torch.log(score_en)
        loss_ent += F.nll_loss(scores_en, triples[:, 2])

        if self.relation_prediction:
            score_rel = self.rdecoder.forward(embedding,r_emb, all_triples, mode="train").view(-1, 2 * self.num_rels)
            loss_rel += self.loss_r(score_rel, all_triples[:, 1])
        
        # Học tương phản: kéo biểu diễn truy vấn từ lịch sử toàn cục và từ chuỗi
        # tiến hoá cục bộ (từng snapshot) lại gần nhau
        if self.use_cl and self.pre_type=="all":
            for id, evolve_emb in enumerate(his_temp_embs):
                t3 = len(his_temp_embs)-id+1
                query = torch.concat([self.his_ent[all_triples[:, 0]],his_r_emb[all_triples[:, 1]]],dim=1)
                query2 = torch.concat([evolve_emb[all_triples[:, 0]], his_rel_embs[id][all_triples[:, 1]]],dim=1)
                x1 = self.w_cl(query)
                x2 = self.w_cl(query2)
                loss_cl += self.get_loss_conv(x1, x2) 

        return loss_ent, loss_rel, loss_static, loss_cl

    def all_GCN(self,ent_emb, sub_graph, use_cuda):
        """Mã hoá đồ thị con lịch sử toàn cục; trả về (embedding chuẩn hoá, chỉ số node có cạnh vào)."""
        sub_graph = sub_graph.to(self.gpu)
        sub_graph.ndata['h'] = ent_emb 
        his_emb = self.his_rgcn_layer.forward(sub_graph, ent_emb, [self.emb_rel, self.emb_rel])
        subg_index = torch.masked_select(
                torch.arange(0, sub_graph.number_of_nodes(), dtype=torch.long).cuda(),
                (sub_graph.in_degrees(range(sub_graph.number_of_nodes())) > 0))
        return F.normalize(his_emb),subg_index
    
    def get_loss_conv(self, ent1_emb, ent2_emb):
        """Loss tương phản kiểu InfoNCE: cặp cùng chỉ số là dương, khác chỉ số là âm.

        Lấy trung bình trên cả bốn ma trận tương đồng (z1·z2, z2·z1, z1·z1, z2·z2).
        """
        loss_fn = nn.CrossEntropyLoss().to(self.gpu)
        z1 = self.projection_model(ent1_emb)
        z2 = self.projection_model(ent2_emb)
        pred1 = torch.mm(z1, z2.T)
        pred2 = torch.mm(z2, z1.T)
        pred3 = torch.mm(z1, z1.T)
        pred4 = torch.mm(z2, z2.T)
        labels = torch.arange(pred1.shape[0]).to(self.gpu)
        train_cl_loss =(loss_fn(pred1 / self.temp, labels) + loss_fn(pred2 / self.temp, labels)+loss_fn(pred3 / self.temp, labels) + loss_fn(pred4 / self.temp, labels)) / 4
        return train_cl_loss
