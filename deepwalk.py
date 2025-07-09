from utils.deepwalk_method import *
import pandas as pd
import numpy as np
import networkx as nx



def process_prescription_data(prescription_path):
    df_prescription = pd.read_excel(prescription_path, engine='openpyxl')
    df_prescription = df_prescription.iloc[:, [0, 4]]
    df_prescription.columns = ['patient_id', 'prescriptions']

    def process_prescriptions(prescriptions):
        all_drugs = set()
        prescriptions = str(prescriptions)
        drugs = prescriptions.split('，')  
        all_drugs.update([drug.strip() for drug in drugs if drug.strip()])
        return list(all_drugs)

    df_prescription['unique_drugs'] = df_prescription['prescriptions'].apply(process_prescriptions)
    df_combined = df_prescription.groupby('patient_id')['unique_drugs'].apply(
        lambda x: list(set([drug for sublist in x for drug in sublist]))).reset_index()
    return df_combined

def map_drugs_to_ids(medicine_coo_path, df_prescription):

    df_medicine = pd.read_excel(medicine_coo_path, usecols=[0, 3], engine='openpyxl')
    df_medicine.columns = ['protein_id', 'drug_name']


    drug_to_proteins = {}
    for _, row in df_medicine.iterrows():
        drug_name = row['drug_name']
        protein_id = row['protein_id']
        if pd.notna(protein_id):
            if drug_name not in drug_to_proteins:
                drug_to_proteins[drug_name] = []
            drug_to_proteins[drug_name].append(protein_id)


    missing_drugs = set()

    def map_drugs_to_ids_list(drugs):
        ids = []
        for drug in drugs:
            drug_id = drug_to_proteins.get(drug)
            if drug_id:
                ids.extend(drug_id)
            else:
                missing_drugs.add(drug)  
        return ids

    df_prescription['drug_ids'] = df_prescription['unique_drugs'].apply(map_drugs_to_ids_list)

    if missing_drugs:
        print("以下药物名称未找到编号：")
        for drug in missing_drugs:
            print(drug)

    G = nx.Graph()
    for _, row in df_medicine.iterrows():
        drug_name = row['drug_name']
        protein_id = row['protein_id']
        G.add_node(drug_name, type='drug')
        G.add_node(protein_id, type='protein')
        G.add_edge(drug_name, protein_id)  


    meta_paths = [
        ['drug', 'interaction', 'protein'],  
        ['protein', 'interaction', 'drug'], 
    ]

    return df_prescription, drug_to_proteins, meta_paths

def build_edges(ppi_path):

    df_ppi = pd.read_excel(ppi_path, engine='openpyxl')
    df_ppi = df_ppi.iloc[:, [0, 1]]

    edges = set()
    for _, row in df_ppi.iterrows():
        protein1, protein2 = row[0], row[1]
        if isinstance(protein1, str) and isinstance(protein2, str):
            edges.add((protein1.strip(), protein2.strip()))
    return edges

def build_graph_per_patient(df_prescription, edges_set, drug_to_proteins):

    protein_interactions = {}
    for protein1, protein2 in edges_set:
        protein_interactions.setdefault(protein1, set()).add(protein2)
        protein_interactions.setdefault(protein2, set()).add(protein1)

    patient_graphs = {}

    for _, row in df_prescription.iterrows():
        patient_id = row['patient_id']
        drug_ids = row['drug_ids']
        G = nx.Graph()

        protein_nodes = set(drug_ids)
        for protein1 in protein_nodes:
            G.add_node(protein1)

            for protein2 in protein_nodes:
                if protein2 in protein_interactions.get(protein1, set()):
                    G.add_edge(protein1, protein2)

            for protein2 in protein_interactions.get(protein1, set()):
                if protein2 not in protein_nodes:
                    G.add_node(protein2)
                    G.add_edge(protein1, protein2)

        patient_graphs[patient_id] = G

    return patient_graphs



def get_drug_embedding(drug, drug_to_proteins, df_embeddings):

    proteins = drug_to_proteins.get(drug)
    if not proteins:
        print(f"警告：药物 '{drug}' 未找到相关蛋白质。")
        return None

    unique_proteins = set(proteins)
    embeddings = df_embeddings.loc[df_embeddings['protein_id'].isin(unique_proteins)]

    if embeddings.empty:
        print(f"警告：药物 '{drug}' 相关蛋白质的嵌入特征未找到。")
        return None

    embeddings_array = embeddings.drop('protein_id', axis=1).values  # 形状: (num_proteins, embedding_dim)

    concatenated_embedding = embeddings_array.flatten()  # 形状: (num_proteins * embedding_dim,)

    return concatenated_embedding

def extract_unique_drugs(prescription_series):

    unique_drugs = set()
    for prescriptions in prescription_series:
 
        if prescriptions is None or (isinstance(prescriptions, float) and np.isnan(prescriptions)):
            continue

        if isinstance(prescriptions, list):
            prescriptions = '，'.join(map(str, prescriptions))

        prescriptions = str(prescriptions)
        drugs = prescriptions.split('，')

        for drug in drugs:
            drug = drug.strip()
            if drug:
                unique_drugs.add(drug)
    return unique_drugs


def generate_drug_embeddings(unique_drugs, drug_to_proteins, df_embeddings, output_path):

    print(f"共提取到 {len(unique_drugs)} 个唯一的药物名称。")
    drug_embeddings = []
    for drug in unique_drugs:
        embedding = get_drug_embedding(drug, drug_to_proteins, df_embeddings)
        if embedding is not None:
            drug_embeddings.append({
                'drug_name': drug,
                'embedding': embedding
            })
        else:
            drug_embeddings.append({
                'drug_name': drug,
                'embedding': np.nan  
            })


    df_drug_embeddings = pd.DataFrame(drug_embeddings)

    df_drug_embeddings['embedding'] = df_drug_embeddings['embedding'].apply(
        lambda x: ','.join(map(str, x)) if isinstance(x, np.ndarray) else ''
    )
    df_drug_embeddings.to_excel(output_path, index=False)
    print(f"单个药物的嵌入特征已保存到 {output_path}。")

    print("查看部分单个药物的嵌入特征：")
    print(df_drug_embeddings.head())



def main():

    prescription_path = r'data\drug\kd2用药3.xlsx'
    #medicine_coo_path = r'data/drug\肾病中西药物.xlsx'
    medicine_coo_path = r'data/drug\处方序列kd2.xlsx'
    ppi_path = r'data/drug\总表（中药西药病人基因）.xlsx'
    embedding_output_path = r'data\kd2_drug_protein_embeddings.xlsx'
    drug_embedding_output_path = r'data\kd2_single_drug_embeddings.xlsx'

    print("步骤1：处理处方数据...")
    df_prescription = process_prescription_data(prescription_path)

    print("步骤2：映射药物到蛋白质...")
    df_prescription, drug_to_proteins,meta_paths = map_drugs_to_ids(medicine_coo_path, df_prescription)

    print("步骤3：构建蛋白质相互作用边...")
    edges_set = build_edges(ppi_path)


    print("步骤4：为每个患者构建蛋白质图...")
    patient_graphs = build_graph_per_patient(df_prescription, edges_set, drug_to_proteins)

    print("步骤5：合并所有患者的图为一个大图...")
    combined_graph = nx.Graph()
    for G in patient_graphs.values():
        combined_graph = nx.compose(combined_graph, G)

    print("步骤6：生成蛋白质嵌入...")
    df_embeddings = generate_protein_embeddings(combined_graph, embedding_output_path,drug_to_proteins)
    #df_embeddings = generate_gcn_embeddings(combined_graph, drug_to_proteins, embedding_output_path)
    #df_embeddings =generate_metapath2vec_embeddings(combined_graph, drug_to_proteins, embedding_output_path, meta_paths)
    #df_embeddings = generate_line_embeddings(combined_graph, embedding_dim=128, loss_type='both', num_epochs=200,learning_rate=0.01)

    print("开始生成药物嵌入...")

    prescription_column = df_prescription.columns[1]  # 'unique_drugs'
    unique_drugs = extract_unique_drugs(df_prescription['unique_drugs'])


    generate_drug_embeddings(unique_drugs, drug_to_proteins, df_embeddings, drug_embedding_output_path)

if __name__ == "__main__":
    main()
