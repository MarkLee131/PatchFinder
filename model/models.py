import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence  
from sparse_activations import sparsemax

class LSTMwithLSTM11(nn.Module):
    def __init__(self, desc_dim, desc_embedding_dim, desc_hidden_dim, desc_n_layers, 
                 code_dim, code_embedding_dim, code_hidden_dim, code_n_layers,
                 dropout, hidden_dim, output_dim):
        super(LSTMwithLSTM11, self).__init__()
        self.desc_n_layers = desc_n_layers
        self.desc_hidden_size = desc_hidden_dim
        self.code_n_layers = code_n_layers
        self.code_hidden_size = code_hidden_dim
        self.code_embedding = nn.Embedding(code_dim, code_embedding_dim)
        self.desc_embedding = nn.Embedding(desc_dim, desc_embedding_dim)
        
        #### 26798 is the number of fields in the dataset
        #### [, 32], 32 is the embedding size
        self.field_warning_embedding = nn.Embedding(26798, 32)
        #### then it output:[batch_size, sequence_length, 32]
        
        #### [input_size, hidden_size]
        self.field_warning_fc = nn.Linear(32,128)
        
        self.field_embedding = nn.Embedding(100002, 512)
        self.priority_embedding = nn.Embedding(4, 32)
        
        self.priority_fc = nn.Linear(32,128)
        self.cat_embedding= nn.Embedding(11, 32)
        self.cat_fc = nn.Linear(32,128)
        self.rule_embedding= nn.Embedding(472, 32)
        self.rule_fc = nn.Linear(32,128)
        self.rank_embedding = nn.Embedding(21, 32)
        self.rank_fc = nn.Linear(32,128)
        
        #### 
        self.dropout = nn.Dropout(0.25)
        
        self.att_code =GlobalAttention(1024)
        self.att_desc=GlobalAttention(1024)
        
        self.code_LSTM = nn.LSTM(input_size=code_embedding_dim,
                                hidden_size=code_hidden_dim,
                                num_layers=code_n_layers,
                                batch_first=True,
                                bidirectional=True,
                                dropout=dropout)
        self.field_LSTM = nn.LSTM(input_size=code_embedding_dim,
                                hidden_size=code_hidden_dim,
                                num_layers=code_n_layers,
                                batch_first=True,
                                bidirectional=True,
                                dropout=dropout)
        self.desc_LSTM = nn.LSTM(input_size=desc_embedding_dim,
                                hidden_size=desc_hidden_dim,
                                num_layers=desc_n_layers,
                                batch_first=True,
                                bidirectional=True,
                                dropout=dropout)

        
        #### [input_size, hidden_size]
        self.fc1 = nn.Linear(3712, 1)
        self.init_weights()
        
        
    #### leave it by default ######
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, desc, code,desc_length,code_length,field_code,field_code_length,cat,rule,rank,priority,field):
        self.code_LSTM.flatten_parameters()
        self.desc_LSTM.flatten_parameters()
        self.field_LSTM.flatten_parameters()
        desc_embedding=self.desc_embedding(desc)
        
        h_0 = torch.zeros(self.desc_n_layers*2, desc.size(0), self.desc_hidden_size,device=desc.device)
        c_0 = torch.zeros(self.desc_n_layers*2, desc.size(0), self.desc_hidden_size,device=desc.device)
        
        
        ####  we remove the zero padding in the code
        if desc_length is not None:
            desc_lens_sorted, indices = desc_length.sort(descending=True)
            desc_sorted = desc_embedding.index_select(0, indices)   
            desc = pack_padded_sequence(desc_sorted, desc_lens_sorted.data.tolist(), batch_first=True)
       
        ##### leave it by default #####
        hids_desc, (h_n, c_n) = self.desc_LSTM(desc, (h_0, c_0))
        
        
        #### we remove the padding in the desc
        #### leave it by default #####
        if desc_length is not None:
            _, inv_indices = indices.sort()
            hids_desc, _ = pad_packed_sequence(hids_desc, batch_first=True)
            hids_desc = F.dropout(hids_desc, p=0.25, training=self.training)
            hids_desc = hids_desc.index_select(0, inv_indices)
            h_n = h_n.index_select(1, inv_indices)
        
        ### max pooling
        hids_desc_max,_ = torch.max(hids_desc, dim=1)
        
        
        
        code_embedding = self.code_embedding(code)
        h_0 = torch.zeros(self.code_n_layers*2, code.size(0), self.code_hidden_size,device=code.device)
        c_0 = torch.zeros(self.code_n_layers*2, code.size(0), self.code_hidden_size,device=code.device)
        if code_length is not None:
            code_lens_sorted, indices = code_length.sort(descending=True)
            code_sorted = code_embedding.index_select(0, indices)   
            code = pack_padded_sequence(code_sorted, code_lens_sorted.data.tolist(), batch_first=True)
        hids_code, (h_n, c_n) = self.code_LSTM(code, (h_0, c_0))
        if code_length is not None:
            _, inv_indices = indices.sort()
            hids_code, _ = pad_packed_sequence(hids_code, batch_first=True)
            hids_code = F.dropout(hids_code, p=0.25, training=self.training)
            hids_code = hids_code.index_select(0, inv_indices)
            h_n = h_n.index_select(1, inv_indices)
        hids_code_max,_ = torch.max(hids_code, dim=1)
        
        field_code_embedding = self.field_embedding(field_code)
        h_0 = torch.zeros(self.code_n_layers*2, field_code.size(0), self.code_hidden_size,device=field_code.device)
        c_0 = torch.zeros(self.code_n_layers*2, field_code.size(0), self.code_hidden_size,device=field_code.device)
        if field_code_length is not None:
            field_code_lens_sorted, indices = field_code_length.sort(descending=True)
            field_code_sorted = field_code_embedding.index_select(0, indices)   
            field_code = pack_padded_sequence(field_code_sorted, field_code_lens_sorted.data.tolist(), batch_first=True)
        hids_field, (h_n, c_n) = self.field_LSTM(field_code, (h_0, c_0))
        if field_code_length is not None:
            _, inv_indices = indices.sort()
            hids_field, _ = pad_packed_sequence(hids_field, batch_first=True)
            hids_field = F.dropout(hids_field, p=0.25, training=self.training)
            hids_field = hids_field.index_select(0, inv_indices)
            h_n = h_n.index_select(1, inv_indices)
        hids_field_max,_ = torch.max(hids_field, dim=1)
        desc_out=self.att_code(hids_code_max.unsqueeze(1),hids_desc,src_len=desc_length)[0]
        desc_out=desc_out.squeeze(1)
        desc_out=self.dropout(desc_out)
        code_out=self.att_desc(hids_desc_max.unsqueeze(1),hids_code,src_len=code_length)[0]
        code_out=code_out.squeeze(1)
        code_out=self.dropout(code_out)
        
        cat=self.cat_embedding(cat)
        cat=self.cat_fc(cat)
        
        rule=self.rule_embedding(rule)
        rule=self.rule_fc(rule)
        
        priority=self.priority_embedding(priority)
        priority=self.priority_fc(priority)
        
        rank=self.rank_embedding(rank)
        rank=self.rank_fc(rank)
        
        #### 
        field_warning=self.field_warning_embedding(field)
        
        #### 3d to 2d (64, 5, 3) -> (64, 15)
        field_warning=field_warning.view(field_warning.size(0), -1)
        field_warning=self.field_warning_fc(field_warning)

        #### 3d to 2d (64, 5, 3) -> (64, 15) field_warning=field_warning.view(field_warning.size(0), 5, -1)
        
        ##### change the order of the field:
        ##### rank = rank.permute(0,2,1)
        
        
        all=torch.cat((code_out,desc_out,hids_field_max,field_warning,cat,rule,priority,rank),dim=1)
        all=self.dropout(all)
        output = self.fc1(all)
        return output
def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (
        torch.arange(0, max_len, device=lengths.device)
        .type_as(lengths)
        .repeat(batch_size, 1)
        .lt(lengths.unsqueeze(1))
    )
       
class GlobalAttention(nn.Module):
    r"""
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.


    .. mermaid::

       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G

    All models compute the output as
    :math:`c = \sum_{j=1}^{\text{SeqLength}} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`\text{score}(H_j,q) = H_j^T q`
       * general: :math:`\text{score}(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`\text{score}(H_j, q) = v_a^T \text{tanh}(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]
       attn_func (str): attention function to use, options [softmax,sparsemax]

    """

    def __init__(self, dim, coverage=False, attn_type="dot", attn_func="softmax"):
        super(GlobalAttention, self).__init__()

        self.dim = dim
        assert attn_type in [
            "dot",
            "general",
            "mlp",
        ], "Please select a valid attention type (got {:s}).".format(attn_type)
        self.attn_type = attn_type
        assert attn_func in [
            "softmax",
            "sparsemax",
        ], "Please select a valid attention function."
        self.attn_func = attn_func

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

        if coverage:
            self.linear_cover = nn.Linear(1, dim, bias=False)

    def score(self, h_t, h_s):
        """
        Args:
          h_t (FloatTensor): sequence of queries ``(batch, tgt_len, dim)``
          h_s (FloatTensor): sequence of sources ``(batch, src_len, dim``

        Returns:
          FloatTensor: raw attention scores (unnormalized) for each src index
            ``(batch, tgt_len, src_len)``
        """
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t = self.linear_in(h_t)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            wq = self.linear_query(h_t)
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous())
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = torch.tanh(wq + uh)

            return self.v(wquh).view(tgt_batch, tgt_len, src_len)

    def forward(self, src, enc_out, src_len=None, coverage=None):
        """

        Args:
          src (FloatTensor): query vectors ``(batch, tgt_len, dim)``
          enc_out (FloatTensor): encoder out vectors ``(batch, src_len, dim)``
          src_len (LongTensor): source context lengths ``(batch,)``
          coverage (FloatTensor): None (not supported yet)

        Returns:
          (FloatTensor, FloatTensor):

          * Computed vector ``(batch, tgt_len, dim)``
          * Attention distribtutions for each query
            ``(batch, tgt_len, src_len)``
        """

        # one step input
        if src.dim() == 2:
            one_step = True
            src = src.unsqueeze(1)
        else:
            one_step = False

        batch, src_l, dim = enc_out.size()
        batch_, target_l, dim_ = src.size()
        if coverage is not None:
            batch_, src_l_ = coverage.size()

        if coverage is not None:
            cover = coverage.view(-1).unsqueeze(1)
            enc_out += self.linear_cover(cover).view_as(enc_out)
            enc_out = torch.tanh(enc_out)

        # compute attention scores, as in Luong et al.
        align = self.score(src, enc_out)

        if src_len is not None:
            mask = sequence_mask(src_len, max_len=align.size(-1))
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.masked_fill_(~mask, -float("inf"))

        # Softmax or sparsemax to normalize attention weights
        if self.attn_func == "softmax":
            align_vectors = F.softmax(align.view(batch * target_l, src_l), -1)
        else:
            align_vectors = sparsemax(align.view(batch * target_l, src_l), -1)
        align_vectors = align_vectors.view(batch, target_l, src_l)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, enc_out)

        # concatenate
        concat_c = torch.cat([c, src], 2).view(batch * target_l, dim * 2)
        attn_h = self.linear_out(concat_c).view(batch, target_l, dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = torch.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

        return attn_h, align_vectors
    
