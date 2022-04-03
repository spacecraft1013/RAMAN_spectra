import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

results_df = pd.read_csv('results_noweights_reference.csv')

sns.set_theme(context='notebook', style='darkgrid', palette='deep', font_scale=1.3)

plot = sns.catplot(
    data=results_df,
    kind='bar',
    x='Average Size',
    y='Validation Accuracy',
    hue='Mode',
    legend=True,
    legend_out=True
)
plot.fig.subplots_adjust(top=.95)
plot.ax.set_title('Validation Accuracy by Average Size and Averaging Method')

plt.savefig('./figures/accuracy_plots.png', dpi=1000, transparent=True)
plt.show()
