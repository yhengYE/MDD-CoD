
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads, dim_out):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(dim_q, dim_out)
        self.key_proj = nn.Linear(dim_kv, dim_out)
        self.value_proj = nn.Linear(dim_kv, dim_out)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=dim_out, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(dim_out)

    def forward(self, query, key_value):
        q = self.query_proj(query)
        k = self.key_proj(key_value)
        v = self.value_proj(key_value)
        attention_output, _ = self.multihead_attention(q, k, v)
        output = self.layer_norm(attention_output + q)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.BatchNorm1d(input_dim)  
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x  
        out = self.layer(x)  
        out += residual 
        return self.activation(out)


class PrescriptionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(PrescriptionEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        embedding = self.fc2(x)

        if embedding.dim() == 3 and embedding.shape[1] == 1:
            embedding = embedding.squeeze(1)  

        return embedding


class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(ClinicalEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        embedding = self.fc2(x)
        return embedding


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

    def forward(self, img, tab):
        """
        img: torch.Tensor, shape [147, 128]
        tab: torch.Tensor, shape [147, 128]
        """
        ###################### Unimodal Layer ##########################
        # Attention for img and tab
        sa = torch.tanh(self.attention(img))  # shape: [147, 1]
        sg = torch.tanh(self.attention(tab))  # shape: [147, 1]

        # Expand dimensions to match in_size
        unimodal_s = sa.expand(-1, self.in_size)  # shape: [147, 128]
        unimodal_g = sg.expand(-1, self.in_size)  # shape: [147, 128]

        # Unimodal fusion
        unimodal = (unimodal_s * img + unimodal_g * tab) / 2  # shape: [147, 128]

        ##################### Bimodal Layer ############################
        # Softmax normalization
        s = F.softmax(img, dim=0)  # shape: [147, 128]
        g = F.softmax(tab, dim=0)  # shape: [147, 128]

        # Bimodal interaction
        sag = 1 / (s * g + 0.5) * (sa + sg)  # shape: [147, 1]
        normalize = F.softmax(sag, dim=0)  # shape: [147, 1]

        # Apply graph fusion
        fused_features = self.graph_fusion(torch.cat([img, tab], dim=1))  # shape: [147, 128]
        s_g = torch.tanh(normalize * fused_features)  # shape: [147, 128]

        # Bimodal fusion
        bimodal = s_g  # shape: [147, 128]

        ###################### Final Fusion ##########################
        # Concatenate unimodal and bimodal features
        fusion = torch.cat([unimodal, bimodal], dim=1)  # shape: [147, 128 * 2]

        # Fully connected layers
        y_1 = torch.tanh(self.linear_1(fusion))  # shape: [147, hidden]
        y_1 = torch.tanh(self.linear_2(y_1))  # shape: [147, hidden]
        y_2 = torch.tanh(self.linear_3(y_1))  # shape: [147, output_dim]

        return y_2



class tensor(nn.Module):

    def __init__(self, in_size, output_dim, hidden=50, dropout=0.5):
        super(tensor, self).__init__()
        self.post_fusion_dropout = nn.Dropout(p=dropout)
        self.post_fusion_layer_1 = nn.Linear((in_size + 1) * (in_size + 1), hidden)
        self.post_fusion_layer_2 = nn.Linear(hidden, hidden)
        self.post_fusion_layer_3 = nn.Linear(hidden, output_dim)

    def forward(self, img, tab):
        """
        img: torch.Tensor, shape [batch_size, in_size]
        tab: torch.Tensor, shape [batch_size, in_size]
        """
        DTYPE = torch.cuda.FloatTensor


        img = torch.cat((Variable(torch.ones(img.size(0), 1).type(DTYPE), requires_grad=False), img), dim=1)
        tab = torch.cat((Variable(torch.ones(tab.size(0), 1).type(DTYPE), requires_grad=False), tab), dim=1)

        # Tensor fusion
        fusion_tensor = torch.bmm(img.unsqueeze(2), tab.unsqueeze(1))  # shape: [batch_size, in_size+1, in_size+1]
        fusion_tensor = fusion_tensor.view(-1, (img.size(1)) * (tab.size(1)))  # shape: [batch_size, (in_size+1) * (in_size+1)]

        # Post-fusion layers
        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
        post_fusion_y_1 = self.post_fusion_layer_1(post_fusion_dropped)  # shape: [batch_size, hidden]
        post_fusion_y_2 = torch.tanh(self.post_fusion_layer_2(post_fusion_y_1))  # shape: [batch_size, hidden]
        post_fusion_y_3 = F.relu(self.post_fusion_layer_3(post_fusion_y_2))  # shape: [batch_size, output_dim]

        return post_fusion_y_3
