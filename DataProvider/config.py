import pandas as pd
import torch
class TaskConfig:
    def __init__(self):
        self.setup_task()

    def setup_task(self):
        self.weight_mode = "grad" #grad
        self.task_name="CRC"
        self.val_metric="AUPRC"#AUROC AUPRC



        if self.task_name == "CRC":

            self.xlsx_path = "./data/CRC/结直肠癌结构化数据.xlsx"
            self.img_dir = "./kd_multi/data/CRC/visual_embeddings"
            self.categorical_columns = ['性别', '癌栓', '神经血管侵犯', '狭窄率', 'EGFR', 'CK8', 'CEA', 'VEFG']
            self.numeric_columns = ['年龄', '淋巴结转移数量', 'ki-67', 'P53']
            self.target_columns = ['s_Label', 'i_Label', 'all_Label']
            self.num_classes = {'s_Label': 3, 'i_Label': 4, 'all_Label': 2}  #342

            self.dim = 1536
            self.key_column = '病理号'
            self.batch_size = 512
            self.test_size = 0.3  # 都是0.3
            self.random_state = 42
            self.epoch = 300  # kd 700 #crc600 #ra400
            self.patience = 140
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.lr = 1e-3  # crc 1e-3   #ra 1e-4   #kd 4e-5
            self.pretrain_epochs =70   # crc 70   #ra 100  #KD70
            self.pretrain_task = 's_Label'
            self.step_size = 100  # crc
            self.step_gamma = 0.85
            self.weight_decay = 5e-4  # crc 5e-4  #ra 1e-4  #KD 1e-4

        elif self.task_name == "MN":
            self.xlsx_path = "./data/KD/all2.xlsx"
            self.img_dir = "./data/KD/visual_embeddings"
            self.categorical_columns = []
            self.numeric_columns = [
                '尿蛋白/尿肌酐比值', 'D二聚体', '纤维蛋白原(FIB)',
                '凝血酶原国际标准化比值(INR)', '凝血酶原时间(PT)', '凝血酶原活动度(AT)', '凝血酶时间(TT)',
                '活化部分凝血活酶时间(APTT)', '白细胞计数(WBC)', '中性粒细胞计数', '红细胞计数(RBC)',
                '血小板计数(PLT)', '血红蛋白测定(Hb)', '尿素(Urea)', '白蛋白(ALB)',
                '肌酐(Cr)', '肾小球滤过率估算值(eGFR)', '空腹葡萄糖', '尿酸(UA)',
                '总胆固醇(TC)', '甘油三酯(TG)', '谷丙转氨酶(ALT)', '谷草转氨酶(AST)', 'GGT'
            ]
            self.target_columns = ['s_Label', 'i_Label', 'all_Label']
            self.num_classes = {'s_Label': 3, 'i_Label': 3, 'all_Label': 3}
            self.dim = 768
            self.key_column = '病理号'
            self.batch_size = 512
            self.test_size = 0.3  # 都是0.3
            self.random_state = 42
            self.epoch = 1000  # kd 800 #crc600 #ra400
            self.patience = 600
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.lr = 1e-4  #AUPRC 1E-4? ROC 4E-5
            self.pretrain_epochs = 70  # crc 70   #ra 100  #KD70
            self.pretrain_task = 's_Label'
            self.step_size = 1000  # crc
            self.step_gamma = 0.85
            self.weight_decay = 1e-4 # crc 5e-4  #ra 1e-4  #KD 1e-4

        elif self.task_name == "RA":
            
            self.xlsx_path = "./data/RA/风湿结构化数据.xlsx"
            self.img_dir = "./data/RA/RA_text_embeddings"
            self.categorical_columns = [
                "增厚滑膜内血流信号", "滑膜血流", "关节软骨面是否显示", "关节软骨面是否光滑",
                "关节软骨面是否连续", "骨皮质不规则", "骨皮质粗糙",
                "骨皮质破坏", "关节间隙变窄"
            ]

            self.numeric_columns = [
                "CRP", "ESR", "RF", "CR", "AST", "ALT",
                "关节积液深度（mm）", "滑膜增厚数值（mm）"
            ]

            self.target_columns = ['s_Label', 'i_Label', 'all_Label']
            self.num_classes = {'s_Label': 4, 'i_Label': 4, 'all_Label': 4}
            self.dim = 768
            self.key_column = '病理号'
            self.batch_size = 512
            self.test_size = 0.3  # 都是0.3
            self.random_state = 42
            self.epoch = 500 # kd 700 #crc600 #ra400
            self.patience = 600
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.lr = 1e-4 # crc 1e-3   #ra 1e-4   #kd 4e-5
            self.pretrain_epochs = 100  # crc 70   #ra 100  #KD70
            self.pretrain_task = 's_Label'
            self.step_size = 1000  # crc
            self.step_gamma = 0.85
            self.weight_decay = 1e-4   # crc 5e-4  #ra 1e-4  #KD 1e-4


        elif self.task_name == "CKD":

            self.xlsx_path = ".\data\KD\中山肾病临床指标数据4.0.xlsx"
            self.img_dir = "./data/KD/npy2"
            self.categorical_columns = [
                '尿白细胞', '尿糖', '尿胆原', '尿胆红素', '尿蛋白', '尿酮体', '尿隐血', '尿亚硝酸盐'
            ]
            self.numeric_columns = [
                '24小时尿蛋白定量', 'AST/ALT比值', '中性粒细胞', '红细胞', '单核细胞', '血小板', '肌酐', '白细胞',
                '尿素氮',
                '*白蛋白', '葡萄糖', '尿酸', '总胆固醇', '*甘油三酯', '*谷丙转氨酶', '*谷草转氨酶',
                '谷丙转氨酶', '谷草转氨酶', 'GGT', '血红蛋白', 'PLT分布宽度', '尿液肌酐'
            ]

            self.target_columns = ['s_Label', 'i_Label', 'all_Label']
            self.num_classes = {'s_Label': 3, 'i_Label': 3, 'all_Label': 4}
            self.dim = 192
            self.key_column = '病理号'
            self.batch_size = 128
            self.test_size = 0.2 # 都是0.3
            self.random_state = 42
            self.epoch = 218# kd 800 #crc600 #ra400
            self.patience = 600
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.lr = 1e-4  # crc 1e-3   #ra 1e-4   #kd 4e-5
            self.pretrain_epochs = 30  # crc 70   #ra 100  #KD70
            self.pretrain_task = 's_Label'
            self.step_size = 600  # crc
            self.step_gamma = 0.85
            self.weight_decay = 1e-4 # crc 5e-4  #ra 1e-4  #KD 1e-4

    

        else:
            raise ValueError(f"未知任务名称: {self.task_name}")


    def load_data(self):
        df = pd.read_excel(self.xlsx_path)
        return df
