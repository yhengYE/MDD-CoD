import pandas as pd
import torch.nn as nn
import torch
import random
import numpy as np
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
from typing import Tuple, List
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(reduction='none', ignore_index=self.ignore_index)

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BinaryRemoveFNLoss(torch.nn.Module):
    """
    Loss function for multimodal contrastive learning based off of the CLIP paper.

    Embeddings are taken, L2 normalized and dot product between modalities is calculated to generate a cosine
    similarity between all combinations of subjects in a cross-modal fashion. Tempered by temperature.
    Loss is calculated attempting to match each subject's embeddings between the modalities i.e. the diagonal.
    """

    def __init__(self,
                 temperature: float,
                 lambda_0: float = 0.5) -> None:
        super(BinaryRemoveFNLoss, self).__init__()

        self.temperature = temperature

        if lambda_0 > 1 or lambda_0 < 0:
            raise ValueError('lambda_0 must be a float between 0 and 1.')
        self.lambda_0 = lambda_0
        self.lambda_1 = 1 - lambda_0

    def forward(self, out0: torch.Tensor, out1: torch.Tensor, y: torch.Tensor) -> Tuple:
        # normalize the embedding onto the unit hypersphere
        if out0.dim() == 1:
            out0 = out0.unsqueeze(0)  # Add batch dimension if missing
        if out1.dim() == 1:
            out1 = out1.unsqueeze(0)  # Add batch dimension if missing

        if y.dim() == 0:
            y = y.unsqueeze(0)  # Add a batch dimension if y is scalar (0D)

        out0 = nn.functional.normalize(out0, dim=1)
        out1 = nn.functional.normalize(out1, dim=1)

        # Calc logits
        logits = torch.matmul(out0, out1.T) / self.temperature
        exp_logits = torch.exp(logits)

        # Calc positive pull signal
        logits_mask = torch.eye(len(y), device=out0.device, dtype=torch.bool)
        pull = logits[logits_mask]

        # Calc negative push signal

        y_p = y.unsqueeze(0)
        y_p = y_p + 1e-6


        fn_mask = y_p * y_p.T  # is symmetric
        fn_mask = fn_mask.to(torch.bool)
        fn_mask.fill_diagonal_(0)  # other view is always pushed
        push_0 = torch.log((exp_logits * ~fn_mask).sum(1))
        push_1 = torch.log((exp_logits * ~fn_mask).T.sum(1))

        # compute log_prob
        log_prob_0 = pull - push_0
        log_prob_1 = pull - push_1

        loss = self.lambda_0 * (-log_prob_0).mean() + self.lambda_1 * (-log_prob_1).mean()

        # log_prob_2 = torch.log((exp_logits.T*fn_mask).sum(1))
        # loss = self.lambda_0*(-log_prob).mean() + self.lambda_1*(-log_prob_2).mean()

        return loss, torch.matmul(out0, out1.T), torch.arange(len(out0), device=out0.device)
class ContrastiveLoss(nn.Module):
    def __init__(self, initial_temp=1.0):

        super(ContrastiveLoss, self).__init__()

        self.temp = nn.Parameter(torch.tensor(initial_temp))

    def forward(self, prescription_embeddings, clinical_embeddings):

        prescription_embeddings = prescription_embeddings / prescription_embeddings.norm(dim=1, keepdim=True)
        clinical_embeddings = clinical_embeddings / clinical_embeddings.norm(dim=1, keepdim=True)

        temp = torch.exp(self.temp)

        similarity_matrix = torch.matmul(prescription_embeddings, clinical_embeddings.t()) / temp

        batch_size = prescription_embeddings.size(0)
        assert similarity_matrix.shape == (batch_size, batch_size), \
            f"Similarity matrix shape mismatch: {similarity_matrix.shape}, expected ({batch_size}, {batch_size})"

        labels = torch.arange(batch_size).to(prescription_embeddings.device)

        loss_fn = nn.CrossEntropyLoss()
        loss_prescription = loss_fn(similarity_matrix, labels)
        loss_clinical = loss_fn(similarity_matrix.t(), labels)

        loss = (loss_prescription + loss_clinical) / 2
        return loss




def evaluate_recall_at_k(prescription_embeddings, clinical_embeddings, top_k=(1,2, 5, 10,20)):

    prescription_embeddings = torch.tensor(prescription_embeddings)
    clinical_embeddings = torch.tensor(clinical_embeddings)

    if prescription_embeddings.dim() == 1:
        prescription_embeddings = prescription_embeddings.unsqueeze(0)
    if clinical_embeddings.dim() == 1:
        clinical_embeddings = clinical_embeddings.unsqueeze(0)

    similarity_matrix = torch.nn.functional.cosine_similarity(
        prescription_embeddings.unsqueeze(1),  # [num_prescriptions, 1, embedding_dim]
        clinical_embeddings.unsqueeze(0),  # [1, num_clinicals, embedding_dim]
        dim=-1
    )  


    recall_results = {k: 0 for k in top_k}

    precision_results = {k: 0 for k in top_k}

    hit_results = {k: 0 for k in top_k}

    map_sum = 0
    mrr_sum = 0
    num_prescriptions = similarity_matrix.size(0)

    for i in range(similarity_matrix.size(0)):

        similarity_vector = similarity_matrix[i]
        sorted_indices = torch.argsort(similarity_vector, descending=True)

        for k in top_k:
            if i in sorted_indices[:k]:
                recall_results[k] += 1
                precision_results[k] += 1 / k
                hit_results[k] += 1  

        relevant_found = False
        precision_at_i = 0
        for rank, idx in enumerate(sorted_indices):
            if idx == i:
                precision_at_i = 1 / (rank + 1)  # Precision at the rank where the relevant item was found
                map_sum += precision_at_i
                mrr_sum += 1 / (rank + 1)
                relevant_found = True
                break
        if not relevant_found:
            map_sum += 0  # If relevant item is not found, precision for this query is 0

    for k in top_k:
        recall_results[k] /= num_prescriptions
        precision_results[k] /= num_prescriptions
        hit_results[k] /= num_prescriptions  

        #print(f"Recall@{k}: {recall_results[k]:.4f}")
        #print(f"Precision@{k}: {precision_results[k]:.4f}")
        print(f"Hit@{k}: {hit_results[k]:.4f}")

    #map_value = map_sum / num_prescriptions
    mrr_value = mrr_sum / num_prescriptions
    #print(f"Mean Average Precision (MAP): {map_value:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {mrr_value:.4f}")



def load_single_drug_embeddings(single_drug_embeddings_path, target_dim=512):

    df_single_drug = pd.read_excel(single_drug_embeddings_path, engine='openpyxl')

    grouped = df_single_drug.groupby('drug_name')['embedding']

    single_drug_feature_dict = {}

    for drug_name, embeddings in grouped:

        embeddings = embeddings.dropna().astype(str).tolist()
        processed_embeddings = []

        for embedding_str in embeddings:
            try:

                embedding = np.fromiter((float(x) for x in embedding_str.split(',')), dtype=float)

                if embedding.size == 0:
                    raise ValueError("Empty embedding.")

                if embedding.size < target_dim:
   
                    padding = np.zeros(target_dim - embedding.size, dtype=embedding.dtype)
                    averaged_embedding = np.concatenate([embedding, padding])
                    print(
                        f"Info: Embedding size for drug '{drug_name}' is {embedding.size}, padded to {target_dim}."
                    )
                elif embedding.size > target_dim:
   
                    if len(embedding) % 2 != 0:
                        print(f"Info: Embedding for drug '{drug_name}' has odd length {len(embedding)}. Dropping the last element.")
                        embedding = embedding[:-1]

                    averaged_embedding = embedding.reshape(-1, 2).mean(axis=1)

                    if averaged_embedding.size < target_dim:
 
                        padding = np.zeros(target_dim - averaged_embedding.size, dtype=averaged_embedding.dtype)
                        averaged_embedding = np.concatenate([averaged_embedding, padding])
                        print(
                            f"Info: Averaged embedding size for drug '{drug_name}' is {averaged_embedding.size - len(padding)}, padded to {target_dim}."
                        )
                    elif averaged_embedding.size > target_dim:

                        averaged_embedding = averaged_embedding[:target_dim]
                        print(
                            f"Info: Averaged embedding size for drug '{drug_name}' is greater than {target_dim}, truncated to {target_dim}."
                        )
                else:

                    averaged_embedding = embedding
                    print(
                        f"Info: Embedding size for drug '{drug_name}' is exactly {target_dim}."
                    )

                embedding_tensor = torch.tensor(averaged_embedding, dtype=torch.float)
            except Exception as e:
                print(f"Error parsing embedding for drug '{drug_name}': {e}. Using zeros.")
                embedding_tensor = torch.zeros(target_dim, dtype=torch.float)

            processed_embeddings.append(embedding_tensor)


        if processed_embeddings:
            final_embedding = torch.stack(processed_embeddings).mean(dim=0)
        else:
            final_embedding = torch.zeros(target_dim, dtype=torch.float)

        single_drug_feature_dict[drug_name] = final_embedding

    return single_drug_feature_dict


import torch
import torch.nn as nn




# def GradNorm_loss(model, visual_loss, text_loss):
#
#     visual_grad = torch.autograd.grad(
#         visual_loss,
#         [p for p in model.parameters() if p.requires_grad],  # Ensure only trainable parameters are used
#         retain_graph=True,
#         create_graph=True,
#         allow_unused=True
#     )
#
#     text_grad = torch.autograd.grad(
#         text_loss,
#         [p for p in model.parameters() if p.requires_grad],
#         retain_graph=True,
#         create_graph=True,
#         allow_unused=True
#     )
#
#     visual_grad_norm = torch.cat([g.view(-1) for g in visual_grad if g is not None]).norm(2)
#     text_grad_norm = torch.cat([g.view(-1) for g in text_grad if g is not None]).norm(2)
#
#     grad_norm_losses = torch.stack([visual_grad_norm, text_grad_norm])
#
#     weights = model.weight_network(grad_norm_losses)
#     weights2 = model.weight_network2(weights)
#
#     weights = F.softmax(weights, dim=0)
#
#     all_weights = torch.cat([weights, weights2])  # [weight_visual, weight_text, weight_fusion]
#
#     grad_norm_loss = (
#         all_weights[0] * visual_grad_norm +
#         all_weights[1] * text_grad_norm
#     )
#
#     return grad_norm_loss, all_weights[0], all_weights[1], all_weights[2]


def GradNorm_loss(model, visual_loss, text_loss):

    visual_grad = torch.autograd.grad(
        visual_loss,
        [p for p in model.parameters() if p.requires_grad],  # Ensure only trainable parameters are used
        retain_graph=True,
        create_graph=True,
        allow_unused=True
    )

    text_grad = torch.autograd.grad(
        text_loss,
        [p for p in model.parameters() if p.requires_grad],
        retain_graph=True,
        create_graph=True,
        allow_unused=True
    )

    visual_grad_norm = torch.cat([g.view(-1) for g in visual_grad if g is not None]).norm(2)
    text_grad_norm = torch.cat([g.view(-1) for g in text_grad if g is not None]).norm(2)

    grad_norm_losses = torch.stack([visual_grad_norm, text_grad_norm])

    all_weights = model.weight_network(grad_norm_losses)
    weights2 = model.weight_network2(all_weights)

    weights = F.softmax(all_weights, dim=0)

    weights= torch.cat([weights, weights2])  # [weight_visual, weight_text, weight_fusion]
    #fusion_weight=1
    grad_norm_loss = (
        weights[0] * visual_grad_norm +weights[1] * visual_grad_norm
    )/2
    #return grad_norm_loss, weights[0], weights[1], weights[2]
    return grad_norm_loss, weights[0], weights[1], weights[2],visual_grad_norm,text_grad_norm



def fix_w(fix_w0,fix_w1):
    grad_norm_loss=0
    fix_w3=1
    return grad_norm_loss,fix_w0, fix_w1,fix_w3


