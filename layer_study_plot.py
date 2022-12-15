from matplotlib import pyplot as plt
import pandas as pd
plt.rcParams['font.sans-serif'] = ['Songti SC']  #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
sst_12layers = [92.5,93.1,93.2,93.7,93.6,93.6,93.4,93.45,93.2,93.1,93.0,92.6]
mrpc_12layers = [89.6,89.94,90.63,90.88,90.60,89.6,89.5,89.2,89.1,89.05,88.93,88.94]
qnli_12layers = [91.3,91.4,91.5,91.6,91.55,91.52,91.48,91.38,91.20,91.1,91.05,90.9]
Geco_12layers = [93.82,93.84,93.88,93.90,93.88,93.88,93.86,93.86,93.85,93.82,93.80,93.78]
lns1 = ax1.plot(range(1, 13), sst_12layers, color='lightcoral', marker='o', linestyle='dashed', label='SST2')
lns2 = ax1.plot(range(1, 13), mrpc_12layers, color='darkgreen', marker='v', linestyle='dashed', label='MRPC')
lns3 = ax1.plot(range(1, 13), qnli_12layers, color='fuchsia', marker='^', linestyle='dashed', label='QNLI')
#lns4 = ax1.plot(range(1, 13), stsb_12layers, color='steelblue', marker='*', linestyle='dashed', label='STS-B')
lns4 = ax2.plot(range(1, 13), Geco_12layers, color='steelblue', marker='*', linestyle='dashed', label='Geco(EN)')
lns = lns1+lns2+lns3+lns4
labs = [l.get_label() for l in lns]
ax1.set_xlabel('层索引值')
ax1.set_ylabel('语言任务表现', fontsize=13)
ax2.set_ylabel('眼动任务表现', fontsize=13)
ax1.legend(loc='lower left')
ax1.legend(lns, labs, loc='lower left')
#ax1.grid()
ax1.set_title('逐层表现')
fig.tight_layout()
plt.savefig('layer_study_no_Geco.pdf', bbox_inches='tight')
plt.show()
