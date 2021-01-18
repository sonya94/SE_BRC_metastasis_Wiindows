from matplotlib import pyplot as plt

y = [
0.64,
0.62,
0.63,
0.63,
0.65,

0.61,
0.63,
0.6660,
0.63,
0.63,

0.66,
0.62,
0.61,
0.65,
0.64,

0.62,
0.59,
0.64,
0.64,
0.64,

0.62,
0.65,
0.65,
0.62,
0.63,

0.62,
0.62,
0.61,
0.6657,
0.60,

0.58,
0.60,
0.56,
0.62,
0.61,

0.59,
0.61,
0.54,
0.56,
0.59,

0.58,
0.59,
0.59
] 

 


#bars = ('43', '42')

#x = [43, 42, 41, 40, 39, 38, 37]
x = range(len(y))
x2 = range(len(y), 0, -1)
xBars = range(len(y), 1, -1)
xBars2 = range(len(y), 0, -2)
#plt.bar(x, y, width=0.7, color="blue")
#plt.bar(x, y, width=0.5, color="grey")

plt.plot(x2, y, color="grey")
plt.plot(x2, y, marker='o', color="grey")
plt.grid(b=True, which='both', axis='both')
plt.title('AUC after Recursive Feature Elimination', fontsize=18)
plt.ylabel('AUC', fontsize=14)
plt.xlabel('Number of Input Features', fontsize=14)

#plt.xticks(x, xBars)
plt.xlim(0, 44)
#plt.ylim(0.5,1.0)
plt.ylim(0,1.0)
#plt.text(7, 0.50, " * 0.67 from 37 features", color="red", size=20)
plt.savefig('plot_rfe_byStage.png')
plt.show() 
