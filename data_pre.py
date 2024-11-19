import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 读取完整数据集
data = pd.read_excel("all_SA_final.csv")

# 去除列名中的多余空格
data.columns = data.columns.str.strip()

# 使用之前筛选的 20 个重要特征，再加上 Gender 特征
selected_features = [
    'Diseases of the gastrointestinal system (AF - acute form, CF - chronic form)',
    'Bone density (quality)',
    'Oral hygiene (Silness-Loe Index)',
    'Bone grafting before implantation',
    'Oncological diseases of distant organs',
    'Diabetes mellitus, WHO classification',
    'Maxillofacial oncology, III group',
    'Periodontal diseases',
    'Kidney disease',
    'Bone width (mm)',
    'Smoking',
    'Complications of implant treatment (except dental implant treatment)',
    'Condition of the right maxillary sinus',
    'Age',
    'Condition of the left maxillary sinus',
    'Bone height (mm)',
    'Oral mucosal diseases',
    'Adjacent medial tooth status',
    'CVD related medication',
    'Sinus lift',
    'Gender'
]

# 替换无效值 '#NUM!' 为 NaN，并填补缺失值
data.replace("#NUM!", pd.NA, inplace=True)
data.fillna(0, inplace=True)

# 检查特征是否存在
missing_features = [f for f in selected_features if f not in data.columns]
if missing_features:
    print(f"以下特征缺失：{missing_features}")
else:
    print("所有特征均存在！")

# 保留存在的特征列
selected_features = [f for f in selected_features if f in data.columns]

# 数据预处理：选取特征并进行标准化
data_for_clustering = data[selected_features].fillna(0)
scaler = StandardScaler()
data_for_clustering = scaler.fit_transform(data_for_clustering)

# 第一步：初始聚类，将患者分为 4 个大类
kmeans_initial = KMeans(n_clusters=4, random_state=42)
data["Cluster"] = kmeans_initial.fit_predict(data_for_clustering)

# 第二步：从初始类别中抽样，生成 600 个 bags
np.random.seed(42)
bags = []
for i in range(600):
    # 随机选择一个类别（Cluster 0 到 Cluster 3）
    cluster_id = np.random.choice([0, 1, 2, 3])
    sampled_data = data[data["Cluster"] == cluster_id].sample(n=np.random.randint(1, 11), replace=True)
    sampled_data["Bag"] = i
    bags.append(sampled_data)

# 合并所有抽样的 bags
bagged_data = pd.concat(bags)

# 添加目标变量和分组标签
bagged_data["label"] = bagged_data["Treatments"]
bagged_data["bag_labels"] = bagged_data["Bag"]
bagged_data["bag_names"] = bagged_data["Bag"]

# 计算并保存 pfs_avg.csv
pfs_avg = bagged_data.groupby("Bag")["Time Until the Failure Occur(in days)"].mean().reset_index(name="PFS_Avg")
pfs_avg.to_csv("pfs_avg.csv", index=False)
print("pfs_avg.csv 文件已成功保存。")

# 计算并保存 ma_avg.csv，使用 Bone density (quality) 作为替代指标
ma_avg = bagged_data.groupby("Bag")["Bone density (quality)"].mean().reset_index(name="MA_Avg")
ma_avg.to_csv("ma_avg.csv", index=False)
print("ma_avg.csv 文件已成功保存。")

# 保存最终的 NSCLC1_600.csv 文件
bagged_data.to_csv("NSCLC1_600.csv", index=False)
print("NSCLC1_600.csv 文件已成功保存。")

