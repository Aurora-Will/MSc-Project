with open('analysis.txt','r') as f:
  analysis_data = f.readlines()

analysis_data =[x.strip('\n') for x in analysis_data]
import numpy as np
from sklearn import metrics
y= []
scores =[]
for value_str in  analysis_data:
  a,b,label =value_str.split(" ")
  a,b,label = float(a),float(b),int(label)
  y.append(label+1)
  scores.append(b)
y = np.array(y)
scores = np.array(scores)

print(scores)
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
# fpr = [1-x for x in fpr]
# tpr = [1-x for x in tpr]
print(fpr)
print(tpr)

auc = metrics.auc(fpr, tpr)
import matplotlib.pyplot as plt
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

