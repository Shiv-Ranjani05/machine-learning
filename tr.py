import numpy as np




from sklearn.preprocessing import PowerTransformer
boxCOX= PowerTransformer(method='box-cox')
data_transformed=boxCOX.fit_transform(data)

from sklearn.preprocessing import PowerTransformer
boxCOX=PowerTransformer()
data_transformed=boxCOX.fit_transform(data)

from sklearn.preprocessing import QuantileTransformer
quantie_trans=QuantileTransformer(o)