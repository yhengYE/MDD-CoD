import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tab_models import FTTransformer, TabTransformer, MLPTableNet,TabNetModel
import argparse
from tools.function import *
from sklearn.metrics import average_precision_score, RocCurveDisplay, precision_recall_curve, auc


set_seed(2014)
class FocalLoss(torch.nn.Module):


    def __init__(self, alpha=0.15, gamma=2.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer model for medical data")
    parser.add_argument('--file_path', type=str, default=r".\data\KOA\骨关节.xlsx", help='Path to the input Excel file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=4.44e-5, help='Initial learning rate')
    parser.add_argument('--step_size', type=int, default=100, help='Step size for learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.85, help='Learning rate decay factor')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for training')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'], help='Optimizer to use for training')
    parser.add_argument('--scheduler', type=str, default='steplr', choices=['steplr', 'coslr'], help='Learning rate scheduler to use')
    parser.add_argument('--model', type=str, default='MLPTableNet', choices=['fttransformer', 'tabtransformer', 'MLPTableNet', 'TabNetModel'], help='Model to use for training')
    parser.add_argument('--tmax', type=int, default=200, help='T_max for CosineAnnealingLR scheduler')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--depth', type=int, default=10, help='Number of transformer layers')
    parser.add_argument('--attn_dropout', type=float, default=0.3573363626849294, help='Attention dropout rate')
    parser.add_argument('--ff_dropout', type=float, default=0.17793923889049573, help='Feed-forward dropout rate')
    parser.add_argument('--loss_function', type=str, default='focal', choices=['cross_entropy', 'focal'], help='Loss function to use for training')
    parser.add_argument('--eval_metric', type=str, default='auroc', choices=['all','auroc','auprc' 'accuracy', 'f1', 'precision', 'recall'], help='Evaluation metric to use')
    return parser.parse_args()

args = parse_args()

file_path = args.file_path
dfdata = pd.read_excel(file_path)

# for col in dfdata.columns:
#     if dfdata[col].isnull().sum() > 0:
#         dfdata[col] = dfdata[col].apply(lambda x: np.random.choice(dfdata[col].dropna().values) if pd.isnull(x) else x)
#
# assert dfdata.isnull().sum().sum() == 0, "数据仍有缺失值，需检查清理流程！"


cat_cols = ['关节疼痛', '活动受限', '僵硬', '肿胀', '摩擦音']  # 类别特征
num_cols = ['尿酸', '白细胞计数', '嗜酸性粒细胞计数', '中性粒细胞百分比', '红细胞计数', '血红蛋白量', '血小板计数'] 

# cat_cols  = [
#     '尿白细胞', '尿糖', '尿胆原', '尿胆红素', '尿蛋白', '尿酮体', '尿隐血', '尿亚硝酸盐'
# ]
# num_cols = [
#     '24小时尿蛋白定量', 'AST/ALT比值', '中性粒细胞', '红细胞', '单核细胞', '血小板', '肌酐', '白细胞',
#     '尿素氮',
#     '*白蛋白', '估算肾小球滤过率', '葡萄糖', '尿酸', '总胆固醇', '*甘油三酯', '*谷丙转氨酶', '*谷草转氨酶',
#     '谷丙转氨酶', '谷草转氨酶', 'GGT', '血红蛋白', 'PLT分布宽度', '尿液肌酐'
# ]

# cat_cols = []
# #

# num_cols = [
#     '24h尿蛋白总量(Pro,24h)',
#     '尿蛋白/尿肌酐比值',
#     'D二聚体',
#     '纤维蛋白原(FIB)',
#     '凝血酶原国际标准化比值(INR)',
#     '凝血酶原时间(PT)',
#     '凝血酶原活动度(AT)',
#     '凝血酶时间(TT)',
#     '活化部分凝血活酶时间(APTT)',
#     '白细胞计数(WBC)',
#     '中性粒细胞计数',
#     '红细胞计数(RBC)',
#     '血小板计数(PLT)',
#     '血红蛋白测定(Hb)',
#     '尿素(Urea)',
#     '白蛋白(ALB)',
#     '肌酐(Cr)',
#     '肾小球滤过率估算值(eGFR)',
#     '空腹葡萄糖',
#     '尿酸(UA)',
#     '总胆固醇(TC)',
#     '甘油三酯(TG)',
#     '谷丙转氨酶(ALT)',
#     '谷草转氨酶(AST)',
#     'GGT'
# ]
#
# cat_cols = ['性别', '癌栓', '神经血管侵犯', '狭窄率', 'EGFR', 'CK8' ,'CEA','VEFG'] 
# num_cols = ['年龄', '淋巴结转移数量', 'ki-67', 'P53']  

set_seed(2014)
target_col = 'Label' 

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

label_encoders = {col: LabelEncoder() for col in cat_cols}
for col in cat_cols:
    dfdata[col] = label_encoders[col].fit_transform(dfdata[col])

target_encoder = LabelEncoder()
dfdata[target_col] = target_encoder.fit_transform(dfdata[target_col])

scaler = StandardScaler()
dfdata[num_cols] = scaler.fit_transform(dfdata[num_cols])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
metric_scores = []
plt.figure(figsize=(10, 8))

for train_index, test_index in kf.split(dfdata):
    train_df, test_df = dfdata.iloc[train_index], dfdata.iloc[test_index]
    X_train, y_train = train_df[cat_cols + num_cols], train_df[target_col]
    X_test, y_test = test_df[cat_cols + num_cols], test_df[target_col]

    X_train_cat = torch.tensor(X_train[cat_cols].values, dtype=torch.long).to(device)
    X_train_cont = torch.tensor(X_train[num_cols].values, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train.values, dtype=torch.long).to(device)

    X_test_cat = torch.tensor(X_test[cat_cols].values, dtype=torch.long).to(device)
    X_test_cont = torch.tensor(X_test[num_cols].values, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test.values, dtype=torch.long).to(device)

    categories = [dfdata[col].nunique() for col in cat_cols]
    num_continuous = len(num_cols)
    n_classes = len(target_encoder.classes_)

    if args.model == 'fttransformer':
        model = FTTransformer(
            categories=categories,
            num_continuous=num_continuous,
            dim=256,  
            depth=args.depth,  
            heads=args.heads,  
            dim_out=n_classes,  
            attn_dropout=args.attn_dropout,
            ff_dropout=args.ff_dropout
        ).to(device)  

    elif args.model == 'tabtransformer':
        model = TabTransformer(
            categories=categories,
            num_continuous=num_continuous,
            dim=64,  
            depth=args.depth,  
            heads=args.heads,  
            dim_head=16,  
            dim_out=n_classes,  
            attn_dropout=args.attn_dropout,
            ff_dropout=args.ff_dropout,
            continuous_mean_std=None 
        ).to(device) 

    elif args.model == 'MLPTableNet':
        model = MLPTableNet(
            categories=categories,  
            num_continuous=num_continuous,  
            hidden_dims=(256, 128, 64), 
            dim_out=n_classes 
             
        ).to(device)

    elif args.model == 'TabNetModel':

        model = TabNetModel(
            columns=12, 
            num_features=12, 
            feature_dims=128, 
            output_dim=64,  
            num_decision_steps=5,  
            relaxation_factor=1.5,  
            batch_momentum=0.02,  
            virtual_batch_size=128,  
            num_classes=n_classes,  
            epsilon=1e-5 
        ).to(device)


    if args.loss_function == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss_function == 'focal':
        criterion = FocalLoss()

    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=2e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    if args.scheduler == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'coslr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax)

    epochs = args.epochs
    batch_size = args.batch_size
    train_data = DataLoader(list(zip(X_train_cat, X_train_cont, y_train)), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_data:
            x_cat, x_cont, y = batch
            optimizer.zero_grad()
            if args.model == 'fttransformer':  
                logits, x = model(x_cat, x_cont)  # (logits, embedding)

            elif args.model == 'TabNetModel':
                x = torch.cat([x_cat, x_cont], dim=1)  
                x = x.to(device)
                logits, prediction = model(x)


            else:
                logits = model(x_cat, x_cont)  #logits

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        if (epoch + 1) % 100 == 0 or epoch == epochs - 1:
            print(f"Fold {fold}, Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Learning Rate: {current_lr:.6f}")

    model.eval()
    with torch.no_grad():
        if args.model == 'fttransformer':
            logits, x = model(X_test_cat, X_test_cont)  # (logits, embedding)

        elif args.model == 'TabNetModel':
            x = torch.cat([X_test_cat, X_test_cont], dim=1)
            x = x.to(device)
            logits, prediction = model(x)

        else:
            logits = model(X_test_cat, X_test_cont) 



        y_test_np = y_test.cpu().numpy()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        predictions = np.argmax(probs, axis=1)

        if args.eval_metric == 'all':
            # 计算所有指标
            results = {}

            # ROC-AUC
            if n_classes == 2:  # 二分类
                roc_auc = roc_auc_score(y_test_np, probs[:, 1])
                RocCurveDisplay.from_predictions(y_test_np, probs[:, 1], ax=plt.gca(), name=f'Fold {fold}')
                results['ROC-AUC'] = roc_auc
            else:  # 多分类
                roc_auc = roc_auc_score(y_test_np, probs, multi_class='ovr')
                results['ROC-AUC'] = roc_auc
                for i in range(n_classes):
                    RocCurveDisplay.from_predictions(
                        y_test_np == i, probs[:, i], ax=plt.gca(),
                        name=f'Fold {fold} Class {i}', alpha=0.3
                    )

            # Accuracy
            accuracy = accuracy_score(y_test_np, predictions)
            results['Accuracy'] = accuracy

            # F1 Score
            f1 = f1_score(y_test_np, predictions, average='weighted')
            results['F1-Score'] = f1

            # Precision
            precision = precision_score(y_test_np, predictions, average='weighted')
            results['Precision'] = precision

            # Recall
            recall = recall_score(y_test_np, predictions, average='weighted')
            results['Recall'] = recall


            print(f"Fold {fold}, Metrics:")
            for metric, value in results.items():
                print(f"{metric}: {value:.4f}")

            metric_scores.append(results) 

        elif args.eval_metric == 'auroc':
            if n_classes == 2:
                metric_score = roc_auc_score(y_test_np, probs[:, 1])
                RocCurveDisplay.from_predictions(y_test_np, probs[:, 1], ax=plt.gca(), name=f'Fold {fold}')
            else:
                metric_score = roc_auc_score(y_test_np, probs, multi_class='ovr')
                for i in range(n_classes):
                    RocCurveDisplay.from_predictions(y_test_np == i, probs[:, i], ax=plt.gca(),
                                                     name=f'Fold {fold} Class {i}', alpha=0.3)
            print(f"Fold {fold}, AUROC: {metric_score:.4f}")
            metric_scores.append(metric_score)

        elif args.eval_metric == 'auprc':
            if n_classes == 2:
                metric_score = average_precision_score(y_test_np, probs[:, 1])
                precision, recall, _ = precision_recall_curve(y_test_np, probs[:, 1])
                plt.plot(recall, precision, color='b', label=f'Fold {fold} (AUPRC = {metric_score:.4f})')
            else:
                metric_score = average_precision_score(y_test_np, probs, average='macro')
                for i in range(n_classes):
                    precision, recall, _ = precision_recall_curve(y_test_np == i, probs[:, i])
                    plt.plot(recall, precision, alpha=0.3, label=f'Fold {fold} Class {i}')
            print(f"Fold {fold}, AUPRC: {metric_score:.4f}")
            metric_scores.append(metric_score)

        elif args.eval_metric == 'accuracy':
            metric_score = accuracy_score(y_test_np, predictions)
            print(f"Fold {fold}, Accuracy: {metric_score:.4f}")
            metric_scores.append(metric_score)

        elif args.eval_metric == 'f1':
            metric_score = f1_score(y_test_np, predictions, average='weighted')
            print(f"Fold {fold}, F1-Score: {metric_score:.4f}")
            metric_scores.append(metric_score)

        elif args.eval_metric == 'precision':
            metric_score = precision_score(y_test_np, predictions, average='weighted')
            print(f"Fold {fold}, Precision: {metric_score:.4f}")
            metric_scores.append(metric_score)

        elif args.eval_metric == 'recall':
            metric_score = recall_score(y_test_np, predictions, average='weighted')
            print(f"Fold {fold}, Recall: {metric_score:.4f}")
            metric_scores.append(metric_score)

        fold += 1

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('5-Fold Cross-Validation AUROC Curves')
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.grid(True)
plt.legend()
#plt.show()

mean_metric_score = np.mean(metric_scores)
print(f"Mean {args.eval_metric.upper()} across 5 folds: {mean_metric_score:.4f}")

#
# def extract_features_all(model, data, cat_cols, num_cols, device, save_dir):
#     import os
#     import numpy as np
#     import torch
#

#     model.eval()
#     os.makedirs(save_dir, exist_ok=True)

#     data[cat_cols] = data[cat_cols].apply(pd.to_numeric, errors='coerce').fillna(-1).astype(int)

#     feature_storage = {}  # {name: {"embeddings": []}}
#
#     for idx, sample in data.iterrows():
#         name = str(sample.iloc[0]).split(".")[0]

#         sample_cont = torch.tensor(sample[num_cols].values, dtype=torch.float32).unsqueeze(0).to(device)

#         sample_cat = torch.tensor(sample[cat_cols].values, dtype=torch.long).unsqueeze(0).to(device)
#
#         with torch.no_grad():

#             logits, embedding = model(sample_cat, sample_cont)

#             if name not in feature_storage:
#                 feature_storage[name] = {"embeddings": []}
#             feature_storage[name]["embeddings"].append(embedding.cpu().numpy().squeeze())
#

#     for name, features in feature_storage.items():
#         avg_embedding = np.mean(features["embeddings"], axis=0)  
#         embedding_filename = os.path.join(save_dir, f"{name}.npy")
#         np.save(embedding_filename, avg_embedding)
#         print(f"样本 {name} 的平均嵌入特征已保存为 '{embedding_filename}'")
#
#
# def extract_features_all(model, data, cat_cols, num_cols, device, save_dir):

#     model.eval()
#     os.makedirs(save_dir, exist_ok=True)  

#     feature_storage = {}  # {name: {"embeddings": []}}

#     empty_cat_tensor = torch.empty((1, 0), dtype=torch.long).to(device)

#     for idx, sample in data.iterrows():
#         name = str(sample.iloc[0]).split(".")[0]
#

#         try:
#             sample_cont = pd.to_numeric(sample[num_cols], errors='coerce').fillna(0).values
#             sample_cont = torch.tensor(sample_cont, dtype=torch.float32).unsqueeze(0).to(device)
#         except Exception as e:
#             print(f"Error processing sample {name}: {e}")
#             continue
#
#         with torch.no_grad():

#             if args.model == 'fttransformer':
#                 logits, embedding = model(empty_cat_tensor, sample_cont)  
#             else:
#                 logits = model(empty_cat_tensor, sample_cont) 
#                 embedding = logits

#             if name not in feature_storage:
#                 feature_storage[name] = {"embeddings": []}
#             feature_storage[name]["embeddings"].append(embedding.cpu().numpy().squeeze())

#     for name, features in feature_storage.items():
#         avg_embedding = np.mean(features["embeddings"], axis=0)  
#         embedding_filename = os.path.join(save_dir, f"{name}.npy")  
#
#         np.save(embedding_filename, avg_embedding)
#         print(f"样本 {name} 的平均嵌入特征已保存为 '{embedding_filename}'")
#
#
# save_dir = ".\data\KD\structure_embeddings"  
# extract_features_all(model, dfdata, cat_cols, num_cols, device, save_dir=save_dir)
