from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from tools.function import *
from torch.utils.data import Dataset
import os
from nets.module import *
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

df_patients = pd.read_excel('data/drug/kd2用药3.xlsx', usecols=[0, 1, 4, 5],
                            dtype={'序号': str})

bert_dim = 768
probBert_dim = 1024
single_drug_dim = 512

def load_prescription_features(prescription_dir, patient_id):

    features = []
    for filename in os.listdir(prescription_dir):
        if filename.startswith(f"{patient_id}_"):  
            file_path = os.path.join(prescription_dir, filename)
            feature = np.load(file_path, allow_pickle=True)
            features.append(torch.tensor(feature, dtype=torch.float))
    if not features:
        print(f"警告: 未找到患者 {patient_id} 的药方特征文件，返回空列表。")
    return features


class CLIPDataset(Dataset):
    def __init__(self, prescription_dir, clinical_dir, patient_ids):
    
        self.prescription_dir = prescription_dir
        self.clinical_dir = clinical_dir
        self.patient_ids = patient_ids

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):

        patient_id = self.patient_ids[idx]
        clinical_feature_path = os.path.join(self.clinical_dir, f"{patient_id}.npy")
        if not os.path.exists(clinical_feature_path):
            raise FileNotFoundError(f"未找到患者 {patient_id} 的临床特征文件：{clinical_feature_path}")
        clinical_feature = torch.tensor(np.load(clinical_feature_path), dtype=torch.float)
        prescription_features = load_prescription_features(self.prescription_dir, patient_id)

        return prescription_features, clinical_feature

def train_clip_model(train_dataloader, test_dataloader,prescription_encoder, clinical_encoder, contrastive_loss_fn, optimizer, device, num_epochs=30, top_k=(1,2, 5, 10,20)):

    prescription_encoder.train()
    clinical_encoder.train()

    for epoch in range(num_epochs):
        total_loss = 0


        all_prescription_embeddings = []
        all_clinical_embeddings = []
        test_all_prescription_embeddings = []
        test_all_clinical_embeddings = []

        for prescription_features, clinical_feature in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):

            optimizer.zero_grad()

            clinical_feature = clinical_feature.to(device)
            batch_prescription_embeddings = []
            batch_clinical_embeddings = []

            for prescription_feature in prescription_features:
                prescription_feature = prescription_feature.to(device)

                prescription_embeddings = prescription_encoder(prescription_feature)
                clinical_embeddings = clinical_encoder(clinical_feature)

                loss = contrastive_loss_fn(prescription_embeddings, clinical_embeddings)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                batch_prescription_embeddings.append(prescription_embeddings)
                batch_clinical_embeddings.append(clinical_embeddings)

            all_prescription_embeddings.extend(torch.cat(batch_prescription_embeddings).cpu().detach().numpy())
            all_clinical_embeddings.extend(torch.cat(batch_clinical_embeddings).cpu().detach().numpy())

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(train_dataloader):.4f}")

        for prescription_features, clinical_feature in tqdm(test_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            with torch.no_grad():
                clinical_feature = clinical_feature.to(device)
                test_batch_prescription_embeddings = []
                test_batch_clinical_embeddings = []

                for prescription_feature in prescription_features:
                    prescription_feature = prescription_feature.to(device)

                    prescription_embeddings = prescription_encoder(prescription_feature)
                    clinical_embeddings = clinical_encoder(clinical_feature)

  
                    loss = contrastive_loss_fn(prescription_embeddings, clinical_embeddings)
                    total_loss += loss.item()

                    test_batch_prescription_embeddings.append(prescription_embeddings)
                    test_batch_clinical_embeddings.append(clinical_embeddings)

                test_all_prescription_embeddings.extend(torch.cat(test_batch_prescription_embeddings).cpu().detach().numpy())
                test_all_clinical_embeddings.extend(torch.cat(test_batch_clinical_embeddings).cpu().detach().numpy())
            print("--------------test------------------")

            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {total_loss / len(train_dataloader):.4f}")

            evaluate_recall_at_k(test_all_prescription_embeddings, test_all_clinical_embeddings, top_k=top_k)



    embeddings = all_prescription_embeddings + all_clinical_embeddings
    labels = [0] * len(all_prescription_embeddings) + [1] * len(all_clinical_embeddings)


    embeddings_reduced = PCA(n_components=3).fit_transform(embeddings)

    # Plotting the 3D embeddings
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot prescription (color = red) and clinical (color = blue)
    ax.scatter(embeddings_reduced[:len(all_prescription_embeddings), 0], embeddings_reduced[:len(all_prescription_embeddings), 1], embeddings_reduced[:len(all_prescription_embeddings), 2], color='r', label='Prescription')
    ax.scatter(embeddings_reduced[len(all_prescription_embeddings):, 0], embeddings_reduced[len(all_prescription_embeddings):, 1], embeddings_reduced[len(all_prescription_embeddings):, 2], color='b', label='Clinical')

    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    ax.legend()

    plt.show()


def stage3():

    patient_ids = df_patients['序号'].tolist()

    train_ids, test_ids = train_test_split(
        patient_ids,
        test_size=0.3,
        random_state=42
    )


    train_dataset = CLIPDataset(
        prescription_dir='data\prescription_features\kd2_prescription_features_bert+prob',
        clinical_dir='data/patient_features/KD2',
        patient_ids=train_ids
    )

    test_dataset = CLIPDataset(
        prescription_dir='data\prescription_features\kd2_prescription_features_bert+prob',
        clinical_dir='data/patient_features/KD2',
        patient_ids=test_ids
    )


    train_dataloader = DataLoader(train_dataset, batch_size=200, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=200, shuffle=False)

    prescription_encoder = PrescriptionEncoder(input_dim=256, hidden_dim=512, embedding_dim=256).to(device)
    clinical_encoder = ClinicalEncoder(input_dim=256, hidden_dim=512, embedding_dim=256).to(device)

    optimizer = torch.optim.Adam(
        list(prescription_encoder.parameters()) + list(clinical_encoder.parameters()),
        lr=1e-4
    )
    contrastive_loss_fn = ContrastiveLoss()

    train_clip_model(
        train_dataloader,
        test_dataloader,
        prescription_encoder,
        clinical_encoder,
        contrastive_loss_fn,
        optimizer,
        device,
        num_epochs=1000
    )

    # torch.save(prescription_encoder.state_dict(), 'model/prescription_encoder_kd2.pth')
    # torch.save(clinical_encoder.state_dict(), 'model/clinical_encoder_kd2.pth')

if __name__ == "__main__":
    stage3()
