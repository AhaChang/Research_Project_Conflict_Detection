"""RGCN layer implementation"""
import dgl
import dgl.function as fn
import dgl.nn as dglnn

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    use_attention : bool, optional
        True to use attention mechanism. Default: False
    """

    def __init__(
        self,
        in_feat,
        out_feat,
        rel_names,
        num_bases,
        *,
        weight=True,
        bias=True,
        activation=None,
        self_loop=False,
        dropout=0.0,
        use_attention=False
    ):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.use_attention = use_attention

        self.conv = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GraphConv(
                    in_feat, out_feat, norm="right", weight=False, bias=False
                )
                for rel in rel_names
            }
        )

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis(
                    (in_feat, out_feat), num_bases, len(self.rel_names)
                )
            else:
                self.weight = nn.Parameter(
                    th.Tensor(len(self.rel_names), in_feat, out_feat)
                )
                nn.init.xavier_uniform_(
                    self.weight, gain=nn.init.calculate_gain("relu")
                )

        # bias
        if bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(
                self.loop_weight, gain=nn.init.calculate_gain("relu")
            )

        self.dropout = nn.Dropout(dropout)

        # Add attention layer
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.ModuleDict({
                rel: nn.Sequential(
                    nn.Linear(2 * in_feat, 1),
                    nn.LeakyReLU(0.2)
                )
                for rel in rel_names
            })
            # Add transformation matrix
            self.transform = nn.ModuleDict({
                rel: nn.Linear(in_feat, out_feat)
                for rel in rel_names
            })

    def forward(self, g, inputs):
        """Forward computation

        Parameters
        ----------
        g : DGLGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.

        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {
                self.rel_names[i]: {"weight": w.squeeze(0)}
                for i, w in enumerate(th.split(weight, 1, dim=0))
            }
        else:
            wdict = {}

        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        # First create features for each node type
        for ntype in g.ntypes:
            if ntype in inputs:
                g.nodes[ntype].data['h'] = inputs[ntype]

        # Message passing with attention mechanism
        if self.use_attention:
            # Calculate attention for each relation type
            for rel in g.canonical_etypes:
                src_type, rel_type, dst_type = rel
                
                if src_type not in inputs or dst_type not in inputs:
                    continue
                    
                # 计算源节点和目标节点的特征
                src_feat = inputs_src[src_type]
                dst_feat = inputs_dst[dst_type]
                
                # 获取边的源节点和目标节点索引
                src, dst = g.edges(etype=rel_type)
                
                # 计算attention scores
                h_src = src_feat[src]
                h_dst = dst_feat[dst]
                
                # 将源节点和目标节点特征拼接
                edge_feat = th.cat([h_src, h_dst], dim=1)
                attention_scores = self.attention[rel_type](edge_feat)
                attention_weights = F.softmax(attention_scores, dim=0)
                
                # 在应用attention weights之前进行特征转换
                transformed_h = self.transform[rel_type](h_src)
                
                # 应用attention weights到转换后的特征
                g.edges[rel_type].data['h'] = transformed_h
                g.edges[rel_type].data['a'] = attention_weights

            # Message passing for each relation type
            hs = {ntype: [] for ntype in g.ntypes if ntype in inputs}
            
            # Perform message passing for each relation
            for src_type, rel_type, dst_type in g.canonical_etypes:
                if src_type not in inputs or dst_type not in inputs:
                    continue
                
                # First transform features
                src_feat = inputs_src[src_type]
                transformed_feat = self.transform[rel_type](src_feat)
                g.nodes[src_type].data['h_transformed'] = transformed_feat
                
                # Perform message passing with transformed features
                g.multi_update_all(
                    {rel_type: (fn.u_mul_e('h_transformed', 'a', 'm'), fn.sum('m', 'h_temp'))},
                    'sum'
                )
                
                # Collect messages for target nodes
                if 'h_temp' in g.nodes[dst_type].data:
                    hs[dst_type].append(g.nodes[dst_type].data.pop('h_temp'))
                
                # Clean up temporary features
                g.nodes[src_type].data.pop('h_transformed')

            # Merge all messages
            for ntype in hs:
                if len(hs[ntype]) > 0:
                    hs[ntype] = th.stack(hs[ntype]).sum(0)
                else:
                    hs[ntype] = g.nodes[ntype].data['h']
        else:
            # Original message passing method
            hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + th.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}



class SHADE(nn.Module):
    def __init__(
        self,
        g,
        h_dim,
        out_dim,
        num_bases,
        num_hidden_layers=1,
        dropout=0,
        use_self_loop=False,
        use_attention=True,
    ):
        super(SHADE, self).__init__()
        self.g = g
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.rel_names = list(set(g.etypes))
        self.rel_names.sort()
        if num_bases < 0 or num_bases > len(self.rel_names):
            self.num_bases = len(self.rel_names)
        else:
            self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_attention = use_attention
        self.in_dim = 2307
        self.feature_transform = nn.Linear(self.in_dim, self.h_dim)
        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(
            RelGraphConvLayer(
                self.h_dim,
                self.h_dim,
                self.rel_names,
                self.num_bases,
                activation=F.relu,
                self_loop=self.use_self_loop,
                dropout=self.dropout,
                weight=False,
                use_attention=self.use_attention
            )
        )
        # h2h
        for i in range(self.num_hidden_layers):
            self.layers.append(
                RelGraphConvLayer(
                    self.h_dim,
                    self.h_dim,
                    self.rel_names,
                    self.num_bases,
                    activation=F.relu,
                    self_loop=self.use_self_loop,
                    dropout=self.dropout,
                    use_attention=self.use_attention
                )
            )
        # h2o
        self.output_transform = RelGraphConvLayer(
                self.h_dim,
                self.out_dim,
                self.rel_names,
                self.num_bases,
                activation=None,
                self_loop=self.use_self_loop,
                use_attention=False
            )

    def forward(self, h=None, blocks=None):
        for ntype in h.keys():
            h[ntype] = self.feature_transform(h[ntype])
        if blocks is None:
            # full graph training
            for layer in self.layers:
                h = layer(self.g, h)
            latent_h = {k: v.clone() for k, v in h.items()}
            h = self.output_transform(self.g, h)
        else:
            # minibatch training
            for layer, block in zip(self.layers, blocks):
                h = layer(block, h)
            latent_h = {k: v.clone() for k, v in h.items()}
            h = self.output_transform(self.g, h)
        return h, latent_h
