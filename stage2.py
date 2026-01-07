# -*- coding: utf-8 -*-
import torch
from sklearn.metrics import balanced_accuracy_score
import os
import pickle
from itertools import chain
import torch.nn as nn
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import BertTokenizer, BertModel
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_adj
from nets.models_drug import Drug_Attention, GnnDrug, VAE, ProteinAttention, DrugMLP
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, f1_score, roc_auc_score
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from tools.function import *
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

bert_dim = 768
probBert_dim = 1024
single_drug_dim = 512
total_feature_dim = bert_dim + single_drug_dim + probBert_dim
protbert_tokenizer = BertTokenizer.from_pretrained(r"F:/models/probBert", do_lower_case=False)
protbert_model = BertModel.from_pretrained(r"F:/models/probBert").to(device)
tokenizer = BertTokenizer.from_pretrained(r'F:/models/bert-chinese')
model = BertModel.from_pretrained(r'F:/models/bert-chinese').to(device)

df_patients = pd.read_excel('data/drug/kd2用药3.xlsx', usecols=[0, 1, 4, 5],
                            dtype={'序号': str})  # 包含 patient ID, time, drug list


str_columns = df_patients.select_dtypes(include=['object']).columns
df_patients[str_columns] = df_patients[str_columns].fillna('').astype(str)

numeric_columns = df_patients.select_dtypes(include=['float64', 'int']).columns
df_patients[numeric_columns] = df_patients[numeric_columns].fillna(0)


df_patients.columns = df_patients.columns.str.strip().str.lower()
if 'label' not in df_patients.columns:
    raise ValueError("Error: 'label' column not found in df_patients.")

df_edges = pd.read_excel('data/drug\药物共现kd2.xlsx')
df_features_prob = pd.read_excel('data/drug\处方序列kd2.xlsx',
                                 usecols=[3, 4])  # 第4列: 药物名称, 第5列: 蛋白质序列
df_features = pd.read_excel('data/drug/药物说明书2KD2.xlsx')
single_drug_embeddings_path = 'data/kd2_single_drug_embeddings.xlsx'

all_drugs = set(chain.from_iterable(df_edges[['Med1', 'Med2']].values.tolist()))
drug_to_idx = {drug: i for i, drug in enumerate(all_drugs)}
edge_weight_map = {(row['Med1'], row['Med2']): row['Weight'] for _, row in df_edges.iterrows()}

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


def extract_bert_features(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def extract_protbert_features(sequence):
    tokens = protbert_tokenizer(sequence, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        outputs = protbert_model(**tokens)
    sequence_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return sequence_embedding

def load_or_save_protein_features(drug_name, protein_sequence):
    feature_save_dir = 'data/drug/pretrain/proball'
    feature_file = os.path.join(feature_save_dir, f"{drug_name}.pkl")

    if os.path.exists(feature_file):
        with open(feature_file, 'rb') as f:
            protein_features = pickle.load(f)
        return protein_features
    else:
        print(f"Extracting and saving features for drug {drug_name}")
        protein_features = extract_protbert_features(protein_sequence)

        os.makedirs(feature_save_dir, exist_ok=True)
        with open(feature_file, 'wb') as f:
            pickle.dump(protein_features, f)

        return protein_features

def load_or_save_bert_features(column_name, df, extractor_function, output_dir='pretrain/drug_bert_features',
                               bert_dim=768):
    os.makedirs(output_dir, exist_ok=True)
    df[column_name] = df[column_name].fillna('').astype(str)
    feature_file = os.path.join(output_dir, f"{column_name}_features.pt")

    if os.path.exists(feature_file):
        feature_tensor = torch.load(feature_file)
        df[f'{column_name}_Tensor'] = list(feature_tensor)
    else:
        def safe_extraction(text):
            try:
                feature_tensor = extractor_function(text)
                if isinstance(feature_tensor, torch.Tensor):
                    return feature_tensor.detach().cpu().float()
                else:
                    raise TypeError("Feature extraction did not return a tensor.")
            except Exception as e:
                return torch.zeros(bert_dim)

        feature_tensors = df[column_name].apply(safe_extraction).tolist()
        torch.save(feature_tensors, feature_file)
        df[f'{column_name}_Tensor'] = feature_tensors

    return df

def build_edge_index_and_weight(node_indices, local_index_map, edge_weight_map, drugs):
    edge_index = []
    edge_weight = []

    if len(node_indices) == 1:
        edge_index.append([local_index_map[node_indices[0]], local_index_map[node_indices[0]]])
        edge_weight.append(0.7)

    for i in range(len(node_indices)):
        for j in range(i + 1, len(node_indices)):
            pair = (drugs[i], drugs[j])
            if pair in edge_weight_map:
                edge_index.append([local_index_map[node_indices[i]], local_index_map[node_indices[j]]])
                edge_weight.append(edge_weight_map[pair])
            reverse_pair = (drugs[j], drugs[i])
            if reverse_pair in edge_weight_map:
                edge_index.append([local_index_map[node_indices[j]], local_index_map[node_indices[i]]])
                edge_weight.append(edge_weight_map[reverse_pair])

    return edge_index, edge_weight

def build_subgraph(drugs, drug_to_idx, edge_weight_map, feature_dict=None, single_drug_feature_dict=None,
                   protein_sequences=None, device=None, apply_dropout=False):

    valid_drugs = []
    invalid_drugs = []

    for drug in drugs.split('，'):
        stripped_drug = drug.strip()
        if stripped_drug in drug_to_idx:
            valid_drugs.append(stripped_drug)
        else:
            invalid_drugs.append(stripped_drug)

    if not valid_drugs:
        raise ValueError("No valid drugs found in the input list. Please check drug names and drug_to_idx.")

    node_indices = [drug_to_idx[drug] for drug in valid_drugs]
    local_index_map = {idx: i for i, idx in enumerate(node_indices)}

    node_features = []

    if feature_dict:
        for drug in valid_drugs:
            if drug in feature_dict:
                features = torch.tensor(feature_dict[drug]).clone().detach().to(device)
            else:
                features = torch.zeros((bert_dim,), dtype=torch.float).to(device)
            node_features.append(features)

    if single_drug_feature_dict:
        for i, drug in enumerate(valid_drugs):
            if drug in single_drug_feature_dict:
                single_embedding = single_drug_feature_dict[drug].clone().detach().to(device)
            else:

                single_embedding = torch.zeros((single_drug_dim,), dtype=torch.float).to(device)
            node_features[i] = torch.cat((node_features[i], single_embedding), dim=0)


    if protein_sequences:
        for i, drug in enumerate(valid_drugs):
            if drug in protein_sequences:
                prot_features = protein_sequences[drug].clone().detach().to(device)
            else:
                prot_features = torch.zeros((probBert_dim,), device=device)
            node_features[i] = torch.cat((node_features[i], prot_features), dim=0)


    edge_index, edge_weight = build_edge_index_and_weight(node_indices, local_index_map, edge_weight_map, valid_drugs)

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float).to(device)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
        edge_weight = torch.empty((0,), dtype=torch.float).to(device)



    x = torch.stack(node_features) if node_features else torch.empty((0, total_feature_dim), device=device)


    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)


    if apply_dropout:
        data.edge_index, _ = dropout_adj(data.edge_index, p=0.2, force_undirected=True, num_nodes=len(node_indices))

    return data

def process_patient_row(row, feature_dict, single_drug_feature_dict, protein_sequences, drug_to_idx, edge_weight_map,
                        bert_dim, probBert_dim, device):

    all_d = row['pre']


    combined_subgraph = build_subgraph(
        drugs=all_d,
        drug_to_idx=drug_to_idx,
        edge_weight_map=edge_weight_map,
        feature_dict=feature_dict,
        single_drug_feature_dict=single_drug_feature_dict,
        protein_sequences=protein_sequences,
        device=device,
        apply_dropout=False
    )


    real_class = row['label']
    combined_subgraph.y = torch.tensor([real_class], dtype=torch.long)

    return combined_subgraph


class PatientDataset(torch.utils.data.Dataset):
    def __init__(self, df, feature_dict, single_drug_feature_dict, protein_sequences, drug_to_idx, edge_weight_map,
                 bert_dim, probBert_dim,
                 device):
        self.df = df.reset_index(drop=True)
        self.feature_dict = feature_dict
        self.single_drug_feature_dict = single_drug_feature_dict
        self.protein_sequences = protein_sequences
        self.drug_to_idx = drug_to_idx
        self.edge_weight_map = edge_weight_map
        self.bert_dim = bert_dim
        self.probBert_dim = probBert_dim
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        combined_subgraph = process_patient_row(
            row, self.feature_dict, self.single_drug_feature_dict, self.protein_sequences, self.drug_to_idx,
            self.edge_weight_map,
            self.bert_dim, self.probBert_dim, self.device
        )
        return combined_subgraph

def run_epoch(dataset, gnn_model, mlp_model_for_prediction, vae, optimizer, loss_fn, device,
              train=True, batch_size=32):

    if train:
        gnn_model.train()
        mlp_model_for_prediction.train()
        vae.train()
    else:
        gnn_model.eval()
        mlp_model_for_prediction.eval()
        vae.eval()

    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    #iterator = tqdm(dataloader, total=len(dataloader), desc="Training" if train else "Validation")

    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            dual_features = gnn_model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
            reconstructed_data, mu, std = vae(dual_features)
            predicted_scores = mlp_model_for_prediction(reconstructed_data)

            real_class_tensor = batch.y.view(-1)

            loss = loss_fn(predicted_scores, real_class_tensor) + 0.001 * torch.mean(mu.pow(2) + std.pow(2) - 2 * std.log() - 1)

            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * batch.num_graphs


        probabilities = torch.softmax(predicted_scores, dim=1).detach().cpu().numpy()
        predicted_labels = predicted_scores.argmax(dim=1).detach().cpu().numpy()
        all_probs.extend(probabilities)
        all_preds.extend(predicted_labels)
        all_labels.extend(real_class_tensor.cpu().numpy())

    average_loss = total_loss / len(dataset)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    bacc = balanced_accuracy_score(all_labels, all_preds)


    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "auroc": auroc,
        "bacc": bacc
    }

    return average_loss, metrics


def main(df_features, df_patients, num_epochs=30, batch_size=32):

    feature_columns = ['性味', '归经', '功效', '副作用']
    for column in feature_columns:
        df_features = load_or_save_bert_features(column, df_features, extract_bert_features)


    feature_tensors = []
    for feature in feature_columns:
        tensors = df_features[f'{feature}_Tensor'].tolist()
        feature_tensors.append(tensors)

    feature_tensors = [torch.stack([f.to(device) for f in tensors]) for tensors in feature_tensors]
    bert_attention = Drug_Attention(bert_dim, num_features=len(feature_columns)).to(device)
    combined_features = bert_attention(feature_tensors)

    df_features['Combined_Attention_Features'] = [f.detach().cpu().numpy().tolist() for f in combined_features]
    feature_dict = df_features.set_index('Drug')['Combined_Attention_Features'].to_dict()



    single_drug_feature_dict = load_single_drug_embeddings(single_drug_embeddings_path)

  
    protein_sequences = {}
    for _, row in df_features_prob.iterrows():
        drug_name = row.iloc[0]
        protein_sequence = row.iloc[1]

        if pd.notna(protein_sequence):
            if drug_name not in protein_sequences:
                protein_sequences[drug_name] = []

            protein_features = load_or_save_protein_features(drug_name, protein_sequence)
            protein_sequences[drug_name].append(protein_features)

    protein_attention = ProteinAttention(probBert_dim).to(device)
    for drug_name, sequences in protein_sequences.items():
        sequences_tensor = torch.stack(sequences).to(device)
        weighted_sequence = protein_attention(sequences_tensor)
        protein_sequences[drug_name] = weighted_sequence.mean(dim=0)

    kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    fold_num = 1
    for train_index, val_index in kf.split(df_patients, df_patients['label']):
        print(f"Starting Fold {fold_num}")
        df_train, df_val = df_patients.iloc[train_index], df_patients.iloc[val_index]

        combined_feature_dim = total_feature_dim  
        gnn_model = GnnDrug(combined_feature_dim, 256).to(device)
        mlp_model_for_prediction = DrugMLP(256).to(device)
        vae = VAE(input_dim=256, hidden_dim=512, latent_dim=512).to(device)


        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        gnn_model.apply(init_weights)
        mlp_model_for_prediction.apply(init_weights)
        vae.apply(init_weights)

        optimizer = torch.optim.AdamW(
            list(gnn_model.parameters()) +
            list(mlp_model_for_prediction.parameters()) +
            list(vae.parameters()),
            lr=1e-4,
            weight_decay=2e-4
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.85)
        loss_fn = FocalLoss(alpha=0.25, gamma=2, reduction='mean')

  
        train_dataset = PatientDataset(df_train, feature_dict, single_drug_feature_dict, protein_sequences, drug_to_idx,
                                       edge_weight_map, bert_dim, probBert_dim, device)
        val_dataset = PatientDataset(df_val, feature_dict, single_drug_feature_dict, protein_sequences, drug_to_idx,
                                     edge_weight_map, bert_dim, probBert_dim, device)

        for epoch in range(num_epochs):
            train_loss, train_metrics = run_epoch(
                train_dataset, gnn_model, mlp_model_for_prediction, vae, optimizer, loss_fn, device, train=True, batch_size=batch_size)
            val_loss, val_metrics = run_epoch(
                val_dataset, gnn_model, mlp_model_for_prediction, vae, optimizer, loss_fn, device, train=False, batch_size=batch_size)

   
            if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
                tqdm.write(
                    f"Fold {fold_num}, Epoch {epoch + 1}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, "
                    f"Train BACC: {train_metrics['bacc']:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
                    f"Val BACC: {val_metrics['bacc']:.4f}, "
                    f"F1: {val_metrics['f1']:.4f}, Precision: {val_metrics['precision']:.4f}, "
                    f"AUROC: {val_metrics['auroc']:.4f}"
                )
            scheduler.step()

        fold_num += 1



if __name__ == "__main__":
    set_seed(2014)
    main(df_features, df_patients, num_epochs=1000, batch_size=64)

