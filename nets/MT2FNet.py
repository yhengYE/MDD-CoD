from ft_trans.tab_models import *
from nets.module import CrossAttention,graph_fusion,tensor

import torch.nn as nn
import torch.nn.functional as F


class MT2FNet(nn.Module):
    def __init__(
            self,
            cats,
            num_cont,
            dim,
            depth,
            heads,
            targets,
            num_classes,
            dim_head=16,
            dim_out=1,
            num_special=2,
            attn_drop=0.3573363626849294,
            ff_drop=0.17793923889049573,
            img_embed_dim=1536,
            num_heads=4,
            hidden_dim=128,
            shared_dim=256
    ):
        super().__init__()
        self.cats = cats
        self.num_classes = num_classes  
        self.cross_attn = CrossAttention(dim_q=hidden_dim, dim_kv=hidden_dim, num_heads=num_heads, dim_out=shared_dim)
        self.tensor =tensor(in_size=128,output_dim=256)
        self.graphfusion = graph_fusion(in_size=128, output_dim=256)

        self.dim_out=dim_out
        self.dim=dim
        self.ft_transformer = FTTransformer(
            categories=cats,
            num_continuous=num_cont,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            dim_out=1,
            num_special_tokens=num_special,
            attn_dropout=attn_drop,
            ff_dropout=ff_drop
        )


        # self.ft_transformer=TabTransformer(
        #     categories=cats,
        #     num_continuous=num_cont,
        #     dim=dim,  
        #     depth=depth,  
        #     heads=heads,  
        #     dim_head=dim_head,  
        #     dim_out=1,  
        #     attn_dropout=attn_drop,
        #     ff_dropout=ff_drop,
        #     continuous_mean_std=None  
        # )


        # self.ft_transformer= TabNetModel(
        #     columns=30,  
        #     num_features=30,  
        #     feature_dims=128,  
        #     output_dim=128,  
        #     num_decision_steps=5,  
        #     relaxation_factor=1.5,  
        #     batch_momentum=0.02,  
        #     virtual_batch_size=128,  
        #     num_classes=128, 
        #     epsilon=1e-5  
        # )

        # self.ft_transformer= MLPTableNet(
        #     categories=cats, 
        #     num_continuous=num_cont,  
        #     hidden_dims=(256, 128, 64),  
        #     dim_out=1 )



        self.reduce_dim = nn.Linear(img_embed_dim, 128)

        self.img_fc = nn.Sequential(
            nn.Linear(128, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU()
        )

        self.tab_fc = nn.ModuleDict({
            's_Label': nn.Sequential(
                nn.Linear(dim, 2*dim),
                nn.BatchNorm1d(2*dim),
                nn.ReLU(),
                nn.Linear(2*dim, num_classes['s_Label'])
            )
        })

        self.img_fc_out = nn.ModuleDict({
            'i_Label': nn.Sequential(
                nn.Linear(dim,  2*dim),
                nn.BatchNorm1d( 2*dim),
                nn.ReLU(),
                nn.Linear( 2*dim, num_classes['i_Label'])
            )
        })

        self.fuse_fc = nn.ModuleDict({
            'all_Label': nn.Sequential(
                nn.Linear(2*dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Linear(dim, num_classes['all_Label'])
            )
        })

        #
        # self.fuse_fc = nn.ModuleDict({
        #     'all_Label': nn.Sequential(
        #         nn.Linear(dim, 2*dim),
        #         nn.BatchNorm1d(2*dim),
        #         nn.ReLU(),
        #         nn.Linear(2*dim, num_classes['all_Label'])
        #     )
        # })


        self.weight_network = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Sigmoid()
        )

        self.weight_network2 = nn.Sequential(
            nn.Linear(2, 16),  
            nn.ReLU(),         
            nn.Linear(16, 1),  
            nn.Softplus()
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, a=-0.1, b=0.1)

    def forward(self, x_cat, x_num, x_img, return_all_feats=False):
        #trans
        tab_logits, tab_embed = self.ft_transformer(x_cat, x_num, return_embedding=True)

        #mlp
        #tab_embed = self.ft_transformer(x_cat, x_num)


        # #tabnet
        # x = torch.cat([x_cat, x_num], dim=1)
        # _,tab_embed = self.ft_transformer(x)


        if x_img.dim() == 3 and x_img.size(1) == 1:
            x_img = x_img.squeeze(1)

        reduced_img_embed = self.reduce_dim(x_img)


        img_embed = self.img_fc(reduced_img_embed)

        fused_feat = self.cross_attn(img_embed, tab_embed)
        #fused_feat = self.MCAT(img_embed, tab_embed)
        #fused_feat = self.Porpoise(img_embed, tab_embed)
        #fused_feat = self.graphfusion(img_embed, tab_embed)  # [1, 147, 128]
        #fused_feat = self.tensor(img_embed, tab_embed)
        #fused_feat = torch.cat((img_embed, tab_embed), dim=1)
        #fused_feat =img_embed * tab_embed
        #fused_feat = img_embed + tab_embed
        #fused_feat =tab_embed
        #fused_feat = img_embed
        # Single-modal predictions
        logits = {}
        for task, tab_layer in self.tab_fc.items():
            logits[task] = tab_layer(tab_embed)

        for task, img_layer in self.img_fc_out.items():
            logits[task] = img_layer(img_embed)

        # Multi-modal predictions
        for task, fuse_layer in self.fuse_fc.items():
            logits[task] = fuse_layer(fused_feat)


        if return_all_feats:
            return logits, fused_feat,img_embed,tab_embed


        return logits


    def get_task_weights(self):
        return F.softmax(self.task_weights, dim=0)

    def get_task_weight_dict(self):
        weights = self.get_task_weights()
        return {'i_Label': weights[0].item(), 's_Label': weights[1].item()}


