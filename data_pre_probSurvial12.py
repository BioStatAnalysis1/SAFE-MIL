import pandas as pd
from lifelines import CoxPHFitter

data = pd.read_excel("all_SA_case2_cox.xlsx")
data['Time (months)'] = data['Time Until the Failure Occur(in days)'] / 30

# 特征列
columns_for_model = [
    'Gender', 'Age',
    'Diseases of the gastrointestinal system (AF - acute form, CF - chronic form)',
    'Oral hygiene (Silness-Loe Index)', 'Bone width (mm)',
    'Bone height (mm)', 'Bone density (quality)', 'Time (months)', 'Treatments'
]
df_model = data[columns_for_model]

# 初始化并拟合 Cox 模型
cph = CoxPHFitter()
cph.fit(df_model, duration_col='Time (months)', event_col='Treatments')

# 预测 12 个月的生存概率
survival_prob_12_months = cph.predict_survival_function(df_model, times=[12]).T

data['Survival Probability (12 months)'] = survival_prob_12_months.values.flatten()

# 保存
data.to_excel("all_SA_case2_cox_with_survival_prob.xlsx", index=False)
