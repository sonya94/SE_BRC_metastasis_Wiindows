histObject = 'M00CRPminDiff'


df7_p = df7.loc[(df7['02OPnTh']==1), [histObject]]
df7_n = df7.loc[(df7['02OPnTh']==2), [histObject]]
df7_p = df7.loc[(df7['02OPnTh']==1)&(df7['M00CRPstartPOD']!=0), [histObject]]
df7_n = df7.loc[(df7['02OPnTh']==1)&(df7['M00CRPstartPOD']==0), [histObject]]
df7_p = df7.loc[(df7["15GT"]==1)&(df7['02OPnTh']==1)&(df7['M00CRPstartPOD']!=0), [histObject]]
df7_n = df7.loc[(df7["15GT"]==0)&(df7['02OPnTh']==1)&(df7['M00CRPstartPOD']!=0), [histObject]]
df7_p = df7.loc[(df7["15GT"]==1)&(df7['02OPnTh']==1)&(df7['M00CRPstartPOD']!=0)&(df7['M00CRPminDiff']>10), [histObject]]
df7_n = df7.loc[(df7["15GT"]==0)&(df7['02OPnTh']==1)&(df7['M00CRPstartPOD']!=0)&(df7['M00CRPminDiff']>10), [histObject]]


df7_p = df7.loc[(df7["15GT"]==1)&(df7['02OPnTh']==1)&(df7['M00CRPstartPOD']!=0), [histObject]]
df7_n = df7.loc[(df7["15GT"]==0)&(df7['02OPnTh']==1)&(df7['M00CRPstartPOD']!=0), [histObject]]
fig, ax = plt.subplots()
plt.hist(df7_n[histObject], bins=50, normed=True, range=(0,50), alpha=0.5, label='Negative case')
plt.hist(df7_p[histObject], bins=50, normed=True, range=(0,50), alpha=0.5, label='Positive case')
#plt.yscale('log')
plt.title(histObject)
plt.xlabel('CRP Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.legend(loc='best', frameon=False)
df7_p.mean()
df7_n.mean()
runTtest(df7_p,df7_n)
plt.show() 
