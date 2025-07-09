from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import StepLR
from tools.plot import *
from tools.function import *
from DataProvider.data_loader import *
from nets.MT2FNet import MT2FNet
from DataProvider.config import TaskConfig
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
def run_epoch(model, dataloader, optimizer=None, scheduler=None, device='cuda', train=True, pretrain_task=None, freeze_ft_transformer=False):

    if train:
        model.train()
    else:
        model.eval()

    if pretrain_task is not None:
        for name, param in model.named_parameters():
            if pretrain_task not in name and "ft_transformer" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    if freeze_ft_transformer:
        for name, param in model.named_parameters():
            if 'ft_transformer' in name:
                param.requires_grad = False

    total_loss = 0.0
    val_aurocs = {'s_Label': [], 'i_Label': [], 'all_Label': []}
    epoch_weights = {'weights': {'s_Label': [], 'i_Label': [], 'all_Label': []}} if train else {}

    for batch_idx, (categorical_input, numeric_input, npy_input, labels, patient_ids) in enumerate(dataloader):

        categorical_input = categorical_input.to(device)
        numeric_input = numeric_input.to(device)
        npy_input = npy_input.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        task_logits = model(categorical_input, numeric_input, npy_input)

        losses = []
        task_losses = {}
        for task_name, logits in task_logits.items():
            task_labels = labels[task_name]
            task_loss = F.cross_entropy(logits, task_labels)
            task_losses[task_name] = task_loss
            losses.append(task_loss)

        if pretrain_task is not None:
            if pretrain_task not in task_losses:
                raise ValueError(f"指定的任务 '{pretrain_task}' 不在模型的输出任务中。")
            total_task_loss = task_losses[pretrain_task]
        else:
            total_task_loss = sum(losses)

        if train:
            if pretrain_task is None:
                visual_loss = losses[0]
                text_loss = losses[1]
                fusion_task_loss = losses[2]
                if config.weight_mode == "grad":
                    grad_norm_loss, dynamic_weight_0, dynamic_weight_1, weights2 = GradNorm_loss(model, visual_loss, text_loss)
                    weighted_visual_loss = dynamic_weight_0 * visual_loss
                    weighted_text_loss = dynamic_weight_1 * text_loss
                    weighted_fusion_task_loss = weights2 * fusion_task_loss
                elif config.weight_mode == "fix":
                    grad_norm_loss,f_0, f_1, f_2 = fix_w(1,1)
                    weighted_visual_loss = f_0 * visual_loss
                    weighted_text_loss = f_1 * text_loss
                    weighted_fusion_task_loss = f_2 * fusion_task_loss
                total_weighted_loss =weighted_fusion_task_loss
                #total_weighted_loss = weighted_visual_loss + weighted_text_loss + weighted_fusion_task_loss
                final_loss = total_weighted_loss + grad_norm_loss

                #final_loss = visual_loss


            else:
                final_loss = total_task_loss

            optimizer.zero_grad()
            final_loss.backward()

            if freeze_ft_transformer:
                for name, param in model.named_parameters():
                    if 'ft_transformer' in name:
                        param.grad = None
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            with torch.no_grad():
                total_loss += final_loss.item()

                if pretrain_task is None:

                    if config.weight_mode == "grad":
                        epoch_weights['weights']['i_Label'].append(float(dynamic_weight_0.item()))
                        epoch_weights['weights']['s_Label'].append(float(dynamic_weight_1.item()))
                        epoch_weights['weights']['all_Label'].append(float(weights2))

                    elif config.weight_mode == "fix":
                        epoch_weights['weights']['i_Label'].append(float(f_0))
                        epoch_weights['weights']['s_Label'].append(float(f_1))
                        epoch_weights['weights']['all_Label'].append(float(f_2))

                    print(f"Batch {batch_idx} - Recorded weights (s_Label):", epoch_weights['weights']['s_Label'])
                    print(f"Batch {batch_idx} - Recorded weights (i_Label):", epoch_weights['weights']['i_Label'])
                    print(f"Batch {batch_idx} - Recorded weights (all_Label):", epoch_weights['weights']['all_Label'])

        else:

            total_loss += total_task_loss.item()

            for task_name, logits in task_logits.items():
                task_labels = labels[task_name]
                probs = F.softmax(logits, dim=1).detach().cpu().numpy()
                true_labels = task_labels.cpu().numpy()

                try:

                    if config.val_metric == "AUPRC":

                        if logits.size(1) == 2:

                            score = average_precision_score(true_labels, probs[:, 1])
                        else:

                            n_classes = probs.shape[1]
                            true_labels_bin = label_binarize(true_labels, classes=range(n_classes))
                            score = average_precision_score(true_labels_bin, probs, average="macro")

                    elif config.val_metric == "AUROC":

                        if logits.size(1) == 2:

                            score = roc_auc_score(true_labels, probs[:, 1])
                        else:

                            score = roc_auc_score(true_labels, probs, multi_class="ovr")

                    else:
                        raise ValueError(f"Invalid metric: {config.val_metric}. Choose 'AUROC' or 'AUPRC'.")


                    val_aurocs[task_name].append(score)

                except ValueError:
                    pass

    for name, param in model.named_parameters():
        param.requires_grad = True

    avg_loss = total_loss / len(dataloader)
    avg_aurocs = {task: (sum(scores) / len(scores) if scores else 0.0)
                  for task, scores in val_aurocs.items()}
    avg_weights = None
    if train and pretrain_task is None:
        avg_weights = {
            task: (sum(float(w) for w in weights if isinstance(w, (int, float))) / len(weights))
            for task, weights in epoch_weights['weights'].items()
        }

    return avg_loss, avg_aurocs, avg_weights


def train(model, train_dataloader, val_dataloader, optimizer, scheduler, num_epochs=200,
          pretrain_epochs=100, patience=5, pretrain_task=None, device='cuda'):

    model.to(device)
    history = {
        'weights': [],
        'all_label_auroc': [],
        'joint_train_losses': [],  
        'joint_val_losses': []     
    }
    best_val_auroc = 0.0
    best_epoch = 0
    stop_counter = 0

    for epoch in range(num_epochs):
        if epoch < pretrain_epochs:
            print(f"Pretraining Epoch {epoch + 1}/{pretrain_epochs}")
            freeze_ft_transformer = False
            task = pretrain_task
        else:
            print(f"Joint Training Epoch {epoch - pretrain_epochs + 1}/{num_epochs - pretrain_epochs}")
            freeze_ft_transformer = True
            task = None

        train_loss, train_aurocs, train_weights = run_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            device,
            train=True,
            pretrain_task=task,
            freeze_ft_transformer=freeze_ft_transformer
        )
        # print(f"Training Loss: {train_loss:.4f}")
        # history['train_losses'].append(train_loss)  # 记录训练损失
        # history['weights'].append(train_weights)

        if epoch >= pretrain_epochs:
            history['joint_train_losses'].append(train_loss)
            print(f"Joint Training Loss: {train_loss:.4f}")

        print(f"Train AUROC: {train_aurocs}")

        val_loss, val_aurocs, _ = run_epoch(
            model,
            val_dataloader,
            optimizer=None,
            scheduler=None,
            device=device,
            train=False,
            pretrain_task=task,
            freeze_ft_transformer=freeze_ft_transformer
        )
        # print(f"Validation Loss: {val_loss:.4f}")
        # history['val_losses'].append(val_loss)  
        # print(f"Validation AUROC: {val_aurocs}")
        # current_val_auroc = val_aurocs.get('all_Label', 0.0)
        # history['all_label_auroc'].append(current_val_auroc)

        if epoch >= pretrain_epochs:
            history['joint_val_losses'].append(val_loss)
            print(f"Joint Validation Loss: {val_loss:.4f}")

        print(f"Validation AUROC: {val_aurocs}")
        current_val_auroc = val_aurocs.get('all_Label', 0.0)
        history['all_label_auroc'].append(current_val_auroc)



        if current_val_auroc > best_val_auroc:
            best_val_auroc = current_val_auroc
            best_epoch = epoch
            stop_counter = 0
            print("Validation AUROC improved, resetting early stopping counter.")
        else:
            stop_counter += 1
            print(f"No improvement in Validation AUROC. Early stopping counter: {stop_counter}/{patience}")

        if stop_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}. Best Validation AUROC: {best_val_auroc:.4f} at epoch {best_epoch + 1}.")
            break

    print(f"训练完成！最佳验证 AUROC: {best_val_auroc:.4f}，发生在第 {best_epoch + 1} 轮。")
    return history

def save_features_as_npy(features, patient_ids, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    for feature, patient_id in zip(features, patient_ids):
        file_path = os.path.join(save_dir, f"{patient_id}.npy")
        np.save(file_path, feature)

def run_epoch_with_feature_extraction(
    model, dataloader, device="cuda", save_features=False, save_dir="./features"
):

    model.eval()
    all_features = []
    all_patient_ids = []

    for batch_idx, (categorical_input, numeric_input, npy_input, labels, patient_ids) in enumerate(dataloader):
        categorical_input = categorical_input.to(device)
        numeric_input = numeric_input.to(device)
        npy_input = npy_input.to(device)

        with torch.no_grad():
            _, fused_feat = model(categorical_input, numeric_input, npy_input, return_fused_feat=True)

        all_features.extend(fused_feat.cpu().numpy())
        all_patient_ids.extend(patient_ids)

        if save_features:
            save_features_as_npy(fused_feat.cpu().numpy(), patient_ids, save_dir)

    return all_features, all_patient_ids

#set_seed(2233)
#set_seed(2015)


config = TaskConfig()
dfdata = config.load_data()
task_name = config.task_name  # "CRC"  "KD" "RA"

if config.task_name in ["RA","KD2"]:
    set_seed(2233)
if config.task_name in ["KD"]:
    set_seed(2015)
if config.task_name in ["CRC"]:
    set_seed(2025)

train_dataset = MultiDataLoader(
    xlsx_path=config.xlsx_path,
    npy_dir=config.img_dir,
    categorical_columns=config.categorical_columns,
    numeric_columns=config.numeric_columns,
    target_columns=config.target_columns,
    key_column=config.key_column,
    train=True,
    test_size=config.test_size,
    random_state=config.random_state,
    batch_size=config.batch_size
)

val_dataset = MultiDataLoader(
    xlsx_path=config.xlsx_path,
    npy_dir=config.img_dir,
    categorical_columns=config.categorical_columns,
    numeric_columns=config.numeric_columns,
    target_columns=config.target_columns,
    key_column=config.key_column,
    train=False,
    test_size=config.test_size,
    random_state=config.random_state,
    batch_size=config.batch_size
)

train_dataloader = train_dataset.get_dataloader(shuffle=True)
val_dataloader = val_dataset.get_dataloader(shuffle=False)

num_classes = config.num_classes


print(f"当前任务: {task_name}")
print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")

model = MT2FNet(
    cats=[config.load_data()[col].nunique() for col in config.categorical_columns],
    num_cont=len(config.numeric_columns),
    dim=128,
    depth=4,
    heads=8,
    targets=config.target_columns,
    num_classes=config.num_classes,
    dim_head=16,
    dim_out=1,
    img_embed_dim=config.dim
).to(config.device)

print(model)

optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=config.lr,
    weight_decay=config.weight_decay
)
scheduler = StepLR(optimizer, step_size=config.step_size, gamma=config.step_gamma)

pretrain_epochs = config.pretrain_epochs
total_epochs = config.epoch

print(f"开始预训练 {pretrain_epochs} 轮...")

history = train(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=total_epochs,
    pretrain_epochs=pretrain_epochs,
    patience=config.patience,
    pretrain_task=config.pretrain_task,
    device=config.device
)

# train_feature_save_dir = "./saved_features/KD2"
# train_features, train_patient_ids = run_epoch_with_feature_extraction(
#     model=model,
#     dataloader=train_dataloader,
#     device=config.device,
#     save_features=True,
#     save_dir=train_feature_save_dir,
# )
# print(f"训练集特征已保存至 {train_feature_save_dir}")

# val_feature_save_dir = "./saved_features/KD2"
# val_features, val_patient_ids = run_epoch_with_feature_extraction(
#     model=model,
#     dataloader=val_dataloader,
#     device=config.device,
#     save_features=True,
#     save_dir=val_feature_save_dir,
# )
# print(f"验证集特征已保存至 {val_feature_save_dir}")

plot_weights_and_auroc(history,start_epoch=pretrain_epochs+1)

plot_loss_curve(
    history
)
