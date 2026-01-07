import math
import torch.nn.init as init
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DrugMLP(nn.Module):
    def __init__(self,input_dim):
        super(DrugMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.layer2 = nn.Linear(1024, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.output_layer = nn.Linear(256, 4)

        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.output_layer(x)
        x = self.softmax(x)
        return x


# class Drug_Attention(nn.Module):
#     def __init__(self, feature_dim, num_features):
#         super(Drug_Attention, self).__init__()
#         self.attention_weights = nn.Parameter(torch.randn(num_features, 1))
#         nn.init.xavier_uniform_(self.attention_weights)
#
#     def forward(self, features):
#         stacked_features = torch.stack([torch.stack(sample, dim=0) for sample in zip(*features)], dim=0)  # Shape: (batch_size, num_features, feature_dim)
#
#         weights = F.softmax(self.attention_weights, dim=0).squeeze(-1)
#         weighted_features = stacked_features * weights.view(1, -1, 1)
#         combined_features = torch.sum(weighted_features, dim=1)
#
#         return combined_features


class Drug_Attention(nn.Module):
    def __init__(self, feature_dim, num_features, d_k=None, d_v=None):

        super(Drug_Attention, self).__init__()

        d_k = d_k or feature_dim
        d_v = d_v or feature_dim

        self.query = nn.Linear(feature_dim, d_k, bias=False)
        self.key = nn.Linear(feature_dim, d_k, bias=False)
        self.value = nn.Linear(feature_dim, d_v, bias=False)

        self.scale = math.sqrt(d_k)

    def forward(self, features):

        stacked_features = torch.stack([torch.stack(sample, dim=0) for sample in zip(*features)],
                                       dim=0)  # (batch_size, num_features, feature_dim)

        Q = self.query(stacked_features)  # (batch_size, num_features, d_k)
        K = self.key(stacked_features)  # (batch_size, num_features, d_k)
        V = self.value(stacked_features)  # (batch_size, num_features, d_v)

        # scores = Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch_size, num_features, num_features)
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, num_features, num_features)
        combined_features = torch.matmul(attention_weights, V)  # (batch_size, num_features, d_v)
        combined_features = torch.sum(combined_features, dim=1)  # (batch_size, d_v)

        return combined_features

class ProteinAttention(nn.Module):
    def __init__(self, input_dim):
        super(ProteinAttention, self).__init__()
        self.query_layer = nn.Linear(input_dim, input_dim)
        self.key_layer = nn.Linear(input_dim, input_dim)
        self.value_layer = nn.Linear(input_dim, input_dim)

        self.scale_factor = math.sqrt(input_dim)

    def forward(self, protein_features):

        Q = self.query_layer(protein_features)  # (num_sequences, input_dim)
        K = self.key_layer(protein_features)  # (num_sequences, input_dim)
        V = self.value_layer(protein_features)  # (num_sequences, input_dim)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale_factor  # (num_sequences, num_sequences)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (num_sequences, num_sequences)
        weighted_features = torch.matmul(attention_weights, V)  # (num_sequences, input_dim)

        return weighted_features

from torch_geometric.nn import TransformerConv, global_mean_pool


class GnnDrug(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads=2, dropout_rate=0.4):
        super(GnnDrug, self).__init__()
        self.conv1 = TransformerConv(input_dim, hidden_dim, heads=heads, 
                                     dropout=dropout_rate, concat=True, edge_dim=1)
        self.conv2 = TransformerConv(hidden_dim * heads, hidden_dim, heads=heads, 
                                     dropout=dropout_rate, concat=False, edge_dim=1)

    def forward(self, x, edge_index, batch, edge_attr=None, **kwargs):
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)

        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        
        aggregated_features = global_mean_pool(x, batch)
        return aggregated_features


#
# class GnnDrug(nn.Module):
#     def __init__(self, input_dim, hidden_dim, gnn_type='gat', heads=4, dropout_rate=0.4):
#         super(GnnDrug, self).__init__()
#         self.gnn_type = gnn_type.lower()
#
#         if self.gnn_type == 'gcn':
#             self.conv1 = GCNConv(input_dim, hidden_dim)
#             self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         elif self.gnn_type == 'graphsage':
#             self.conv1 = SAGEConv(input_dim, hidden_dim)
#             self.conv2 = SAGEConv(hidden_dim, hidden_dim)
#         elif self.gnn_type == 'gin':
#             nn1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
#             nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
#             self.conv1 = GINConv(nn1)
#             self.conv2 = GINConv(nn2)
#         elif self.gnn_type == 'gat':
#             self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout_rate)
#             self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout_rate)
#         elif self.gnn_type == 'armgcn':
#             self.conv1 = ARMAConv(input_dim, hidden_dim, num_stacks=1, num_layers=2, shared_weights=True, dropout=dropout_rate)
#             self.conv2 = ARMAConv(hidden_dim, hidden_dim, num_stacks=1, num_layers=2, shared_weights=True, dropout=dropout_rate)
#         elif self.gnn_type == 'sgat':

#             self.conv1 = SGConv(input_dim, hidden_dim)
#             self.conv2 = SGConv(hidden_dim, hidden_dim)
#         elif self.gnn_type == 'gin+gat':
#             nn1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
#             self.conv1 = GINConv(nn1)
#             self.conv2 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout_rate)
#         elif self.gnn_type == 'graphsage+armgcn':
#             self.conv1 = SAGEConv(input_dim, hidden_dim)
#             self.conv2 = ARMAConv(hidden_dim, hidden_dim, num_stacks=1, num_layers=2, shared_weights=True, dropout=dropout_rate)
#         elif self.gnn_type == 'gnntransformer+armgcn':
#             self.conv1 = TransformerConv(input_dim, hidden_dim, heads=heads, dropout=dropout_rate)
#             self.conv2 = ARMAConv(hidden_dim * heads, hidden_dim, num_stacks=1, num_layers=2, shared_weights=True, dropout=dropout_rate)
#         else:
#             raise ValueError(f"Unknown GNN type: {self.gnn_type}")
#
#         self.dropout = dropout_rate
#
#     def forward(self, x, edge_index, batch):
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=0.1, training=self.training)
#         x = self.conv2(x, edge_index)

#         aggregated_features = global_mean_pool(x, batch)
#         return aggregated_features


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        self.fc_mu = nn.Linear(latent_dim * 2, latent_dim)
        self.fc_std = nn.Linear(latent_dim * 2, latent_dim)

    def encode(self, x):
        """
        x : [batch_size,784]
        """
        x = self.encoder(x)
        return self.fc_mu(x), F.softplus(self.fc_std(x) - 5, beta=1)

    def reparameterize(self, mu, std):
        """
        mu : [batch_size,z_dim]
        std : [batch_size,z_dim]
        """
        # get epsilon from standard normal
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        return self.decoder(z)


    def forward(self, x):
        mu, std = self.encode(x)
        z = self.reparameterize(mu, std)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, std



    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD


class graph_fusion(nn.Module):

    def __init__(self, in_size, output_dim, hidden=50, dropout=0.5):
        super(graph_fusion, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)

        self.graph_fusion = nn.Sequential(
            nn.Linear(in_size * 2, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, in_size),
            nn.Tanh()
        )

        self.attention = nn.Linear(in_size, 1)
        self.linear_1 = nn.Linear(in_size * 2, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, output_dim)

        self.hidden = hidden
        self.in_size = in_size

    def forward(self, combine_subgraph_features, global_feature):
        ###################### unimodal layer  ##########################
        sa = torch.tanh(self.attention(combine_subgraph_features)).unsqueeze(0)  # shape: [1, in_size]
        sg = torch.tanh(self.attention(global_feature)).unsqueeze(0)  # shape: [1, in_size]

        unimodal_s = sa.expand(1, self.in_size)  # shape: [1, in_size]
        unimodal_g = sg.expand(1, self.in_size)  # shape: [1, in_size]

        unimodal = (unimodal_s * combine_subgraph_features + unimodal_g * global_feature) / 2  # shape: [1, in_size]

        ##################### bimodal layer ############################
        s = F.softmax(combine_subgraph_features, dim=0)  # shape: [in_size]
        g = F.softmax(global_feature, dim=0)  # shape: [in_size]

        sag = (1 / (s * g + 0.5) * (sa + sg))  # shape: [1, in_size]

        normalize = F.softmax(sag, dim=1)  # shape: [1, in_size]

        normalize = normalize.squeeze(0)  # shape: [in_size]

        s_g = torch.tanh(
            normalize * self.graph_fusion(torch.cat([combine_subgraph_features, global_feature]))  # shape: [in_size]
        )

        s_g = s_g.unsqueeze(0)  # shape: [1, in_size]

        bimodal = s_g  # shape: [1, in_size]

        fusion = torch.cat([unimodal, bimodal], dim=1)  # shape: [1, in_size * 2]

        y_1 = torch.tanh(self.linear_1(fusion.squeeze(0)))  # shape: [in_size * 2]
        y_1 = torch.tanh(self.linear_2(y_1))  # shape: [hidden]
        y_2 = torch.tanh(self.linear_3(y_1))  # shape: [output_dim]

        return y_2


class tensor(nn.Module):

    def __init__(self, in_size, output_dim, hidden=50, dropout=0.5):
        super(tensor, self).__init__()
        self.post_fusion_dropout = nn.Dropout(p=dropout)
        self.post_fusion_layer_1 = nn.Linear((in_size + 1) * (in_size + 1) * (in_size + 1), hidden)
        self.post_fusion_layer_2 = nn.Linear(hidden, hidden)
        self.post_fusion_layer_3 = nn.Linear(hidden, output_dim)

    def forward(self, drug, glo):
        DTYPE = torch.cuda.FloatTensor
        # a1 = x[:,0,:]; v1 = x[:,1,:]; l1 = x[:,2,:]

        drug = torch.cat((Variable(torch.ones(drug.size(0), 1).type(DTYPE), requires_grad=False), drug), dim=1)
        glo = torch.cat((Variable(torch.ones(glo.size(0), 1).type(DTYPE), requires_grad=False),  glo), dim=1)


        fusion_tensor = torch.bmm(drug.unsqueeze(2), glo.unsqueeze(1))
        fusion_tensor = fusion_tensor.view(-1, (drug.size(1) + 1) * (glo.size(1) + 1), 1)


        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
        post_fusion_y_1 = (self.post_fusion_layer_1(post_fusion_dropped))
        post_fusion_y_2 = torch.tanh(self.post_fusion_layer_2(post_fusion_y_1))
        post_fusion_y_3 = F.relu(self.post_fusion_layer_3(post_fusion_y_2))
        #  output = post_fusion_y_3 * self.output_range + self.output_shift
        y_2 = post_fusion_y_3

        return y_2


