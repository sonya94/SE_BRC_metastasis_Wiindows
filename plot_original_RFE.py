from matplotlib import pyplot as plt
import numpy as np
y = [
0.48,   # 1
0.51,   # 2
0.66,   # 3
0.54,   # 4
0.57,   # 5

0.64,    # 6
0.59,   # 7
0.66,   # 8
0.52,   # 9
0.66,   # 10

0.70,   # 11
0.60,   # 12
0.60,   # 13
0.58,   # 14
0.55,   # 15

0.60,   # 16
0.64,   # 17
0.59    # 18
] 


y2 = [
0.59,   # elim 0
0.64,   # elim 1
0.60,   # elim 2
0.55,   # elim 3
0.58,   # elim 4

0.60,    # elim 5
0.60,   # elim 6
0.70,   # elim 7
0.66,   # elim 8
0.52,   # elim 9

0.66,   # elim 10
0.59,   # elim 11
0.64,   # elim 12
0.57,   # elim 13
0.54,   # elim 14

0.66,   # elim 15
0.51,   # elim 16
0.48    # elim 17
]


#bars = ('43', '42')

#x = [43, 42, 41, 40, 39, 38, 37]
x = range(len(y), 0)
x2 = range(len(y2), 0, -1)
xBars = range(len(y), 1, -1)
xBars2 = range(len(y), 0, -2)
#plt.bar(x, y, width=0.7, color="blue")
#plt.bar(x, y, width=0.5, color="grey")

plt.plot(x2, y2, color="grey")
plt.plot(x2, y2, marker='o', color="grey")
plt.grid(b=True, which='both', axis='both')
plt.title('AUC after Recursive Feature Elimination', fontsize=18)
plt.ylabel('AUC', fontsize=14)
plt.xlabel('Number of Input Features', fontsize=14)

#plt.xticks(x, xBars)
plt.xticks(np.arange(0, len(y), 5))
plt.xlim(0, 19)
#plt.ylim(0.5,1.0)
plt.ylim(0, 1.0)
#plt.text(7, 0.50, " * 0.67 from 37 features", color="red", size=20)
plt.savefig('plot_rfe_original_metadata.png')
plt.show() 
