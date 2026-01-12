import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
# Plot ROC curve
plt.figure()
# Plot the ROC curve with a label displaying the ROC AUC score
plt.plot(fpr, tpr, color='darkorange', lw=2,
label='ROC curve (area = %0.2f)' % roc_auc)
# Plot a dashed diagonal line for reference
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# Set the x and y-axis limits
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
# Label the x and y-axes
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Set the title of the plot
plt.title('Receiver Operating Characteristic')
# Add a legend to the plot
plt.legend(loc='lower right')
# Display the ROC curve plot
plt.show()

