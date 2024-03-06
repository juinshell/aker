import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 列名
columns = ["cp", "cutcp", "fft", "lbm", "mrif", "mriq", "sgemm", "stencil"]

raw_data = np.zeros((8, 8))
# 创建一个空的8x8数据框
data = pd.DataFrame(raw_data, columns=columns, index=columns)

data_dic = {
    "cp_cutcp": -40.35043386,
    "cp_fft": 10.57696599,
    "cp_lbm": -29.97889276,
    "cp_mrif": -45.05903255,
    "cp_mriq": -69.69457717,
    "cp_sgemm": 15,
    "cp_stencil": -10.86129325,
    "cutcp_fft": 31.89527621,
    "cutcp_lbm": -20.20935931,
    "cutcp_mrif": -28.87452205,
    "cutcp_mriq": -132.2306274,
    "cutcp_sgemm": 8.861540783,
    "cutcp_stencil": -30.70709167,
    "fft_lbm": 29.74142713,
    "fft_mrif": -5.688244726,
    "fft_mriq": 22.7299194,
    "fft_sgemm": 32.71188502,
    "fft_stencil": 15.44681487,
    "lbm_mrif": 29.07497319,
    "lbm_mriq": 17.38509584,
    "lbm_sgemm": 13.66099623,
    "lbm_stencil": -45.84286082,
    "mrif_mriq": -13.6822231,
    "mrif_sgemm": 29.66036735,
    "mrif_stencil": 16.55089885,
    "mriq_sgemm": 25.7554491,
    "mriq_stencil": -1.397512494,
    "sgemm_stencil": 5.841708278
}

# 创建一个对称的矩阵
for i in range(8):
    for j in range(i+1, 8):
        kernel1_name = columns[i]
        kernel2_name = columns[j]
        raw_data[i, j] = data_dic[kernel1_name + "_" + kernel2_name] / 100

#以corr的形状生成一个全为0的矩阵 
mask = np.zeros_like(raw_data)
#将mask的对角线及以上设置为True
#这部分就是对应要被遮掉的部分
mask[np.triu_indices_from(mask)] = True
# 使用seaborn绘制热力图

plt.figure(figsize=(10, 8))
sns.heatmap(data, annot=True, cmap='coolwarm', cbar=False, square=True, mask=mask)
plt.title('CD Kernel Pair Heatmap')
plt.show()
