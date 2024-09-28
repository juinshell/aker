import matplotlib.pyplot as plt
import matplotlib
# print(matplotlib.get_cachedir())
# input("Press Enter to continue...")
import matplotlib as mpl
import numpy as np
mpl.rcParams["font.family"] = "Times New Roman"
bes = ["cp_fft", "cutcp_fft", "fft_lbm", "fft_mriq", "fft_sgemm", "fft_stencil", "lbm_mrif", "lbm_mriq", "mrif_sgemm", "mrif_stencil", "hot3d_lava", "nn_path"]
pair_num = len(bes)

max_error = 0.0
avg_error = 0.0
def main(filename):
    ret = [[] for i in range(pair_num)]
    with open(filename) as F:
        x = [[] for i in range(pair_num)]
        y = [[] for i in range(pair_num)]
        line = F.readline()
        while line:
            line = line.split(",")
            for i in range(pair_num):
                x[i].append(float(line[i*2]))
                y[i].append(float(line[i*2 + 1]))
            line = F.readline()
        for i in range(pair_num):
            k = (y[i][-1] - y[i][0]) / (x[i][-1] - x[i][0])
            b = y[i][0] - k * x[i][0]
            error = 0.0
            for j in range(len(x[i])):
                # error += ((k * x[i][j] + b - y[i][j])/y[i][j])**2
                ret[i].append(abs((k * x[i][j] + b - y[i][j])/y[i][j]) * 100)
                # show
                print(f"kernel: {i}, x: {x[i][j]}, y: {y[i][j]}, pred: {k * x[i][j] + b}, error: {ret[i][-1]}")
                global max_error
                global avg_error
                max_error = max(max_error, ret[i][-1])
                avg_error += ret[i][-1]
    print(f"len(x[0]): {len(x[0])}, len(x): {len(x)}")
    print(f"max_error: {max_error}, avg_error: {avg_error/len(x[0])/len(x)}")
    return ret

all_data = main("cd_pair_accuracy.csv")

# fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(9,4))
fig = plt.figure(figsize=(15, 3))
ax = fig.add_subplot(111)

bplot=ax.boxplot(all_data,
                    widths=0.8,
                    vert=True,
                    patch_artist=True)
# 三种颜色交互使用
colors = ["#F26077", "#A9A9A9", "#079B9B"]

# for bplot in (bplot1, bplot2):
for patch, color_idx in zip(bplot['boxes'], range(len(bplot['boxes']))):
    color = colors[color_idx % len(colors)]
    patch.set_facecolor(color)

x_tic = range(1, 11 + 2)
kernel_pair = [("fft", "cp"), ("fft", "cutcp"), ("fft", "lbm"), ("mriq", "fft"), ("sgemm", "fft"), ("stencil", "fft"), ("mrif", "lbm"), ("mriq", "lbm"), ("sgemm", "mrif"), ("mrif", "stencil"), ("hot3d", "lava"), ("nn", "path")]

ax.set_xticks(x_tic)
ax.set_xticklabels(bes, fontsize=11)
ax.set_xlabel("Benchmarks", fontsize=14)
ax.set_ylabel("Prediction Error Rate (%)", fontsize=14)
plt.savefig("mix_pred.pdf", bbox_inches="tight")
plt.show()