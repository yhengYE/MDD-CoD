import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

# Helper functions
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# Residual wrapper
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# Pre-normalization wrapper
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# GEGLU activation
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

# FeedForward layer
def FeedForward(dim, mult=4, dropout=0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

# Attention module
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out), attn

# Transformer module
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout),
                FeedForward(dim, dropout=ff_dropout)
            ])
            for _ in range(depth)
        ])

    def forward(self, x, return_attn=False):
        post_softmax_attns = []

        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = attn_out + x
            x = ff(x) + x

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)

# MLP module
class MLP(nn.Module):
    def __init__(self, dims, act=None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims_pairs) - 1)
            layers.append(nn.Linear(dim_in, dim_out))
            if not is_last:
                layers.append(default(act, nn.ReLU()))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

# Numerical Embedder
class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases

# TabTransformer
class TabTransformer(nn.Module):
    def __init__(
        self, categories, num_continuous, dim, depth, heads, dim_head=16, dim_out=1,
        mlp_hidden_mults=(4, 2), mlp_act=None, num_special_tokens=2, continuous_mean_std=None,
        attn_dropout=0., ff_dropout=0., use_shared_categ_embed=True, shared_categ_dim_divisor=8.
    ):
        super().__init__()
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        shared_embed_dim = 0 if not use_shared_categ_embed else int(dim // shared_categ_dim_divisor)
        self.category_embed = nn.Embedding(total_tokens, dim - shared_embed_dim)

        self.use_shared_categ_embed = use_shared_categ_embed
        if use_shared_categ_embed:
            self.shared_category_embed = nn.Parameter(torch.zeros(self.num_categories, shared_embed_dim))
            nn.init.normal_(self.shared_category_embed, std=0.02)

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

        self.num_continuous = num_continuous
        if num_continuous > 0:
            if exists(continuous_mean_std):
                assert continuous_mean_std.shape == (num_continuous, 2)
            self.register_buffer('continuous_mean_std', continuous_mean_std)
            self.norm = nn.LayerNorm(num_continuous)

        self.transformer = Transformer(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head,
            attn_dropout=attn_dropout, ff_dropout=ff_dropout
        )

        input_size = (dim * self.num_categories) + num_continuous
        hidden_dimensions = [input_size * t for t in mlp_hidden_mults]
        all_dimensions = [input_size, *hidden_dimensions, dim_out]
        self.mlp = MLP(all_dimensions, act=mlp_act)
        self.mlp2 = nn.Sequential(
            nn.Linear(1046, 128),
        )

    def forward(self, x_categ, x_cont, return_attn = False,return_embedding = True):
        xs = []
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset
            categ_embed = self.category_embed(x_categ)
            if self.use_shared_categ_embed:
                shared_categ_embed = repeat(self.shared_category_embed, 'n d -> b n d', b=categ_embed.shape[0])
                categ_embed = torch.cat((categ_embed, shared_categ_embed), dim=-1)
            x, attns = self.transformer(categ_embed, return_attn=True)
            flat_categ = rearrange(x, 'b ... -> b (...)')
            xs.append(flat_categ)

        if self.num_continuous > 0:
            if exists(self.continuous_mean_std):
                mean, std = self.continuous_mean_std.unbind(dim=-1)
                x_cont = (x_cont - mean) / std
            normed_cont = self.norm(x_cont)
            xs.append(normed_cont)

        x = torch.cat(xs, dim=-1)
        out=self.mlp2(x)

        logits = self.mlp(x)

        if return_attn:
            return logits, attns

        if return_embedding:
            return logits, out

        return logits, attns

# FTTransformer
class FTTransformer(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        num_special_tokens = 2,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            # categorical embedding

            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        # continuous

        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)

        # cls token

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # transformer

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # to logits

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x_categ, x_numer, return_attn = False, return_embedding = True):
        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        xs = []
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset

            x_categ = self.categorical_embeds(x_categ)

            xs.append(x_categ)

        # add numerically embedded tokens
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)

            xs.append(x_numer)

        # concat categorical and numerical

        x = torch.cat(xs, dim = 1)

        # append cls tokens
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)

        # attend

        x, attns = self.transformer(x, return_attn = True)

        # get cls token

        x = x[:, 0]

        # out in the paper is linear(relu(ln(cls)))

        logits = self.to_logits(x)

        if return_attn:
            return logits, attns

        if return_embedding:
            return logits, x

        return logits, attns


class MLPTableNet(nn.Module):
    def __init__(
            self,
            categories,
            num_continuous,
            hidden_dims=(256, 128, 128),
            dim_out=128,
            activation=nn.ReLU
    ):
        """
        MLP-based TableNet for tabular data processing.

        Args:
            categories: List of unique category counts for categorical features.
            num_continuous: Number of continuous (numerical) features.
            hidden_dims: Tuple of hidden layer dimensions for the MLP.
            dim_out: Output dimension (e.g., 1 for regression, N for classification).
            activation: Activation function to use (default is ReLU).
        """
        super().__init__()

        self.num_categories = len(categories)
        self.num_continuous = num_continuous

        # Embedding layers for categorical features
        self.embeds = nn.ModuleList([
            nn.Embedding(num_embeddings=cat_count, embedding_dim=16)  # 16-dim embeddings for each category
            for cat_count in categories
        ])

        # Input dimension calculation
        input_dim = (len(categories) * 16) + num_continuous  # Embedding dims + numerical feature dims

        # MLP layers
        mlp_layers = []
        for dim in hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, dim))
            mlp_layers.append(activation())
            input_dim = dim
        mlp_layers.append(nn.Linear(input_dim, 128))  # Output layer

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x_categ, x_cont):
        """
        Forward pass.

        Args:
            x_categ: Tensor of shape [batch_size, num_categories].
            x_cont: Tensor of shape [batch_size, num_continuous].

        Returns:
            logits: Output predictions.
        """
        # Process categorical features using embeddings
        embedded = [embed(x_categ[:, i]) for i, embed in enumerate(self.embeds)]
        x_categ_emb = torch.cat(embedded, dim=-1)  # Concatenate embeddings

        # Concatenate categorical embeddings and continuous features
        x = torch.cat([x_categ_emb, x_cont], dim=-1)

        # Pass through MLP
        logits = self.mlp(x)
        return logits


import torch.nn.functional as F

import torch
import torch.nn as nn


class Sparsemax(nn.Module):
    """
    Sparsemax activation function, used for sparse attention as described in:
    "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification".
    """

    def __init__(self, dim=-1):
        super(Sparsemax, self).__init__()
        self.dim = dim

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Sparsemax output.
        """
        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=self.dim, keepdim=True)[0]
        # Sort input in descending order
        zs = torch.sort(input, dim=self.dim, descending=True)[0]
        range = torch.arange(1, zs.size(self.dim) + 1, device=input.device, dtype=input.dtype).view(-1, *([1] * (input.dim() - 1))).transpose(0, self.dim)
        # Calculate cumulative sums and sparsity threshold
        bound = 1 + range * zs
        cumsum_zs = torch.cumsum(zs, dim=self.dim)
        is_gt = (bound > cumsum_zs).type(input.dtype)
        k = torch.max(is_gt * range, dim=self.dim, keepdim=True)[0]
        taus = (torch.sum(is_gt * zs, dim=self.dim, keepdim=True) - 1) / k
        return torch.max(torch.zeros_like(input), input - taus)




sparsemax = Sparsemax(dim=1)


# GLU
def glu(act, n_units):
    act[:, :n_units] = act[:, :n_units].clone() * torch.nn.Sigmoid()(act[:, n_units:].clone())

    return act

class TabNetModel(nn.Module):
    def __init__(
            self,
            columns=3,
            num_features=3,
            feature_dims=128,
            output_dim=64,
            num_decision_steps=6,
            relaxation_factor=0.5,
            batch_momentum=0.001,
            virtual_batch_size=2,
            num_classes=2,
            epsilon=1e-5
    ):
        super().__init__()

        self.columns = columns
        self.num_features = num_features
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.num_classes = num_classes
        self.epsilon = epsilon

        # Feature transformation layers
        self.feature_transform_linear1 = nn.Linear(num_features, feature_dims * 2, bias=False)
        self.BN = nn.BatchNorm1d(num_features, momentum=batch_momentum)
        self.BN1 = nn.BatchNorm1d(feature_dims * 2, momentum=batch_momentum)

        self.feature_transform_linear2 = nn.Linear(feature_dims * 2, feature_dims * 2, bias=False)
        self.feature_transform_linear3 = nn.Linear(feature_dims * 2, feature_dims * 2, bias=False)
        self.feature_transform_linear4 = nn.Linear(feature_dims * 2, feature_dims * 2, bias=False)

        # Mask layer
        self.mask_linear_layer = nn.Linear(feature_dims * 2 - output_dim, num_features, bias=False)
        self.BN2 = nn.BatchNorm1d(num_features, momentum=batch_momentum)

        # Final classifier layer
        self.final_classifier_layer = nn.Linear(output_dim, num_classes, bias=False)

    def encoder(self, data):
        """
        Encoding logic for TabNet.
        """
        batch_size = data.shape[0]
        device = data.device  # Dynamically get the device of input data

        # Ensure all created tensors are on the same device as input data
        features = self.BN(data)
        output_aggregated = torch.zeros([batch_size, self.output_dim], device=device)

        masked_features = features
        mask_values = torch.zeros([batch_size, self.num_features], device=device)
        aggregated_mask_values = torch.zeros([batch_size, self.num_features], device=device)
        complementary_aggregated_mask_values = torch.ones([batch_size, self.num_features], device=device)

        total_entropy = 0

        for ni in range(self.num_decision_steps):
            if ni == 0:
                transform_f1 = self.feature_transform_linear1(masked_features)
                norm_transform_f1 = self.BN1(transform_f1)

                transform_f2 = self.feature_transform_linear2(norm_transform_f1)
                norm_transform_f2 = self.BN1(transform_f2)

            else:
                transform_f1 = self.feature_transform_linear1(masked_features)
                norm_transform_f1 = self.BN1(transform_f1)

                transform_f2 = self.feature_transform_linear2(norm_transform_f1)
                norm_transform_f2 = self.BN1(transform_f2)

                # GLU
                transform_f2 = (glu(norm_transform_f2, self.feature_dims) + transform_f1) * np.sqrt(0.5)

                transform_f3 = self.feature_transform_linear3(transform_f2)
                norm_transform_f3 = self.BN1(transform_f3)

                transform_f4 = self.feature_transform_linear4(norm_transform_f3)
                norm_transform_f4 = self.BN1(transform_f4)

                # GLU
                transform_f4 = (glu(norm_transform_f4, self.feature_dims) + transform_f3) * np.sqrt(0.5)

                decision_out = torch.relu(transform_f4[:, :self.output_dim])

                # Decision aggregation
                output_aggregated = torch.add(decision_out, output_aggregated)
                scale_agg = torch.sum(decision_out, dim=1, keepdim=True) / (self.num_decision_steps - 1)
                aggregated_mask_values = torch.add(aggregated_mask_values, mask_values * scale_agg)

                features_for_coef = transform_f4[:, self.output_dim:]

                if ni < (self.num_decision_steps - 1):
                    mask_linear_layer = self.mask_linear_layer(features_for_coef)
                    mask_linear_norm = self.BN2(mask_linear_layer)
                    mask_linear_norm = torch.mul(mask_linear_norm, complementary_aggregated_mask_values)
                    mask_values = sparsemax(mask_linear_norm)

                    complementary_aggregated_mask_values = torch.mul(
                        complementary_aggregated_mask_values,
                        self.relaxation_factor - mask_values
                    )
                    total_entropy += torch.mean(
                        torch.sum(-mask_values * torch.log(mask_values + self.epsilon), dim=1)
                    ) / (self.num_decision_steps - 1)
                    masked_features = torch.mul(mask_values, features)

        return output_aggregated, total_entropy

    def classify(self, output_logits):
        """
        Classification logic.
        """
        logits = self.final_classifier_layer(output_logits)
        predictions = torch.softmax(logits, dim=1)
        return logits, predictions

    def forward(self, x):
        """
        Forward pass for TabNetModel.
        Args:
            x: Input data of shape [batch_size, num_features].

        Returns:
            logits: Unnormalized scores for each class.
            predictions: Probability distribution over classes.
        """
        # Pass the input data through the encoder
        output_aggregated, total_entropy = self.encoder(x)

        # Pass the encoded features through the classification layer
        logits, predictions = self.classify(output_aggregated)

        return logits, output_aggregated