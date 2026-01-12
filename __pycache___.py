import numpy as np

import time
SIZE=100000
L1=range(SIZE)
L2=range(SIZE)
A1=np.arange(SIZE)
A2=np.arange(SIZE)
start=time.time()
result=[(x,y) for x,y in zip(L1,L2)]
print((time.time()-start)*1000)
start=time.time()
result=A1+A2
print((time.time()-start)*1000)

import pandas as pd
import numpy as np
dict = {'First Score':[100,90,np.nan,95],
'Second Score':[30,45,56,np.nan],
'Third Score':[np.nan,40,80,98]}
df = pd.DataFrame(dict)
df.isnull()

import pandas as pd
dict={'name':["Shiv","Ranjani","Shizu","Anii"],
      'degree':["BCA","MCA","MBA","M.Tech"],
      'score':[90,40,80,98]}
df=pd.DataFrame(dict)
for i,j in df.iterrows():
    print(i,j)
    print()