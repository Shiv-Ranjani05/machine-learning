# PART 1: Importing Libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns



matplotlib.style.use('ggplot')

# PART 2: Making the data distributions
np.random.seed(42)

x = pd.DataFrame({
    # Distribution with lower outliers
    'x1': np.concatenate([
        np.random.normal(20, 1, 2000),
        np.random.normal(1, 1, 20)
    ]),
    # Distribution with higher outliers
    'x2': np.concatenate([
        np.random.normal(30, 1, 2000),
        np.random.normal(50, 1, 20)
    ]),
})

# PART 3: Scaling the Data
scaler = preprocessing.RobustScaler()
robust_scaled_df = scaler.fit_transform(x)
robust_scaled_df = pd.DataFrame(robust_scaled_df, columns=['x1', 'x2'])

# PART 4: Visualizing the impact of scaling
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

ax1.set_title('Before Scaling')
sns.kdeplot(x['x1'], ax=ax1, label='x1')
sns.kdeplot(x['x2'], ax=ax1, label='x2')
ax1.legend()

ax2.set_title('After Robust Scaling')
sns.kdeplot(robust_scaled_df['x1'], ax=ax2, label='x1')
sns.kdeplot(robust_scaled_df['x2'], ax=ax2, label='x2')
ax2.legend()

plt.tight_layout()
plt.show()