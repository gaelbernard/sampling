import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from scipy.stats import pearsonr
import matplotlib.ticker as ticker

# Loading the dataset
df = pd.read_csv('../results.csv')
print (df.columns)

# Colors
colors = ["#ffa600","#ffa600","#ffa600","#ff6e54","#dd5182","#955196","#003f5c","#003f5c","#003f5c"]
dashes = [[1,0],[1,1],[4,4],[1,0],[1,0],[1,0],[1,0],[1,1],[1,0]]
markers = [',',',',',','^','v','s','.','.','o']

# Preprocessing
df['technique'] += df['expectedOccReduction'].astype('str')
df.rename({'emd':'EMD'}, axis=1, inplace=True)
mapping = {
    'RandomSamplingFalse':'T1. Random',
    'RandomSamplingTrue':'T2. Variants stratified',
    'BiasedSamplingVariantFalse':'T3. Variants biased',
    'SimilarityBasedTrue':'T4. Behavior-based',
    'LogRankTrue':'T5. LogRank',
    'IterativeCentralityWithRedundancyCheckTrue':'T6. Redundancy check',
    'IterativeCminSumFalse':'T7. Iterative c-min',
    'IterativeCminSumTrue':'T8. Iterative c-min (EOR)',
    'IterativeCminSumEuclideanTrue':'T9. Iterative c-min (Eucl.)'
}
df['technique'] = df['technique'].map(mapping)
df['error sampling'] = 1-df['trulysampled']
df['ESB'] = 1-df['trulysampled']
df.loc[df['SUM_time_preprocessing']>600, 'timeout'] = True
# 18emd chart (graph showing results for 18 datasets)
if True:
    sns.set_style(style="whitegrid")
    subset = [1,3,4,8]
    subcolor = [colors[x] for x in subset]
    subdashes = [dashes[x] for x in subset]
    submarker = [markers[x] for x in subset]
    sns.set_palette(sns.color_palette(subcolor))
    tiny = df.copy()
    tiny = tiny[df['technique'].isin(['T2. Variants stratified', 'T4. Behavior-based', 'T5. LogRank', 'T9. Iterative c-min (Eucl.)'])]
    ax = sns.relplot(
        data=tiny,
        x="p", y="EMD",
        hue="technique",
        kind="line",
        style="technique",
        dashes=subdashes,
        markers=submarker,
        height=2,
        aspect=1,
        facet_kws=dict(
            sharex=True,
            sharey=False,
        ),
        col="dataset",
        col_wrap=6,
        ci=None,
    )
    for ax in ax.axes.flat:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, p: str(f'{y:.2f}')[1:]))
    plt.tight_layout()
    plt.savefig('output/18emd.svg')
    plt.close()

# Pearson correlation coeff. table
if True:
    table = []
    for id, d in enumerate(df['dataset'].unique()):
        i = (~df['timeout']) & (df['dataset']==d)
        pearson, pvalue = pearsonr(df.loc[i,'ESB'], df.loc[i,'EMD'])
        table.append({'id':id+1, 'dataset':d, 'pearson':round(pearson,4), 'pvalue':pvalue, 'df':df.loc[i,'ESB'].shape[0]-2})
    table = pd.DataFrame(table)
    table['df'] = table['df'].astype(str)
    table['pearson'] = table['pearson'].round(2).astype(str).str.lstrip('0').str.ljust(3,'0')
    table["Pearson's r (APA style)"] = '$r('+table['df']+')='+table['pearson']+', p<.01$'
    if table.loc[table['pvalue']>=0.01,:].shape[0] != 0:
        raise ValueError('P value is not <0.01')
    table.index = table.index + 1
    table.drop(['df','pearson','pvalue','dataset'], axis=1).to_latex('output/correlation.txt', index=False)
    print (table.to_string())
    exit()
# T8 and T9
if True:
    subset = [6]
    subcolor = [colors[x] for x in subset]
    subdashes = [dashes[x] for x in subset]
    submarker = [markers[x] for x in subset]
    sns.set_palette(sns.color_palette(subcolor))
    s = ['T7. Iterative c-min']
    data = df.loc[df['technique'].isin(s),:].groupby(['p','ds_n_variants','dataset'])['time_sampling'].mean().reset_index()
    print (data)
    ax = sns.scatterplot(
        x='p',
        y='ds_n_variants',
        size='time_sampling',
        data=data,
        hue='time_sampling'
        #markers=submarker,
        #size="size
    )
    #plt.xlim(0,5000)
    plt.tight_layout(pad=1.5)
    plt.show()

# T8 and T9
if True:
    subset = [6,7,8]
    subcolor = [colors[x] for x in subset]
    subdashes = [dashes[x] for x in subset]
    submarker = [markers[x] for x in subset]
    sns.set_palette(sns.color_palette(subcolor))
    s = ['T7. Iterative c-min', 'T8. Iterative c-min (EOR)', 'T9. Iterative c-min (Eucl.)']
    data = df.loc[(~df['timeout'])&df['technique'].isin(s),:].groupby(['p','technique'])['time_sampling'].mean().unstack()
    ax = sns.lineplot(
        data=data,
        color=subcolor,
        markers=submarker,
        #size="size
    )
    #plt.xlim(0,5000)
    plt.tight_layout(pad=1.5)
    plt.show()


# Time graph
df['time_sampling'] = df['SUM_time_preprocessing'] - df['time_load_csv']
sns.set(rc={'figure.figsize':(4.8,4)})
sns.set_style(style="whitegrid")
bins = ['<1s','<5s','<1m', '>1m', 'timeout (>10m)']
df['time_bins'] = pd.cut(df['time_sampling'],bins=[0,1,5,60,1200,np.inf], labels=bins)
df['time_bins'] = df['time_bins'].astype(str)
df.loc[df['timeout'], 'time_bins'] = 'timeout (>10m)'
time = df.groupby('technique')['time_bins'].value_counts(sort=False).unstack().fillna(0)[bins]
time /= time.sum(axis=1).max()/100
ax = time.sort_index(ascending=False).plot.barh(stacked=True, color=['#eeeeee','#adadad','#707070','#383838','#ff0000'])
plt.legend(loc="upper right", ncol=len(df.columns), bbox_to_anchor=(1.02, 1.24), title="execution time")
plt.xlim(0,100)
plt.ylabel("")
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout(rect=[0, 0.0, 1, 0.98])
plt.savefig('output/time_overview1.svg')
plt.close()

sns.set(rc={'figure.figsize':(5,4)})
sns.set_style(style="whitegrid")
flierprops = dict(markerfacecolor='black', markeredgecolor='black')

data = df.loc[~df['timeout'],:].groupby(['dataset','p','technique'])['time_sampling'].mean().unstack()
ax = sns.boxplot(data=data, orient="h", color='white' , flierprops=flierprops)
ax.grid(False)
# iterate over boxes
for i,box in enumerate(ax.artists):
    box.set_edgecolor('black')
    box.set_facecolor('white')
    # iterate over whiskers and median lines
    for j in range(6*i,6*(i+1)):
         ax.lines[j].set_color('black')
h_offset = 0.1
v_offset = 0
medians = data.median().round(2)
for ytick in ax.get_yticks():
    ax.text(medians[ytick] + v_offset, ytick+h_offset, str(medians[ytick])+'s', horizontalalignment='center',size='small',color='b',weight='semibold')
ax.set_xscale('log')
ax.set_xticks([]) # <--- set the ticks first
for v in [1,10,60,300,600]:
    plt.axvline(v, 0, v, color='#ccc', zorder=0)
plt.tight_layout(rect=[0, 0.0, 1, 0.98])
plt.savefig('output/time_overview2.svg')
plt.close()




# Aggregated graph
sns.set(rc={'figure.figsize':(6,4)})
sns.set_style(style="whitegrid")
sns.set_palette(sns.color_palette(colors))
for t in ['EMD', 'ESB']:
    u = df.groupby(['technique','p','dataset'])[t].mean()
    u = ((1-u/u.loc['T1. Random'])*-1).reset_index()

    plt.axvline(0, 0, 2, zorder=1, color='black', linestyle='-', linewidth=2)
    ax = sns.boxplot(x=t, y=u['technique'], data=u*100, orient='h', medianprops={'color':'white', 'linewidth':2.5, 'linestyle':':'}, whis=10)
    plt.xlim(-90.1,90.1)
    plt.tight_layout(rect=[0, 0.0, 1, 0.98])
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.title(t)
    #plt.xlabel('percentage change compared to random'.format(t))
    plt.savefig('output/aggregated_{}.pdf'.format(t))
    plt.close()





# Individual graph
sns.set(rc={'figure.figsize':(7.8 ,4)})
sns.set_style(style="whitegrid")
sns.set_palette(sns.color_palette(colors))
for t in ['EMD', 'ESB']:
    pivot = df.pivot_table(index=['dataset','p'], columns='technique', values=t, aggfunc='mean')
    for d in df['dataset'].unique():
        ldf = pivot.loc[d,:]
        #ldf = df.loc[df['dataset']==d]
        #pivot = ldf.pivot_table(index='p', columns='technique', values=t, aggfunc='mean')

        cols = []
        print (pivot.shape)
        sns.lineplot(
            data=ldf,
            dashes=dashes,
            markers=markers,
            linewidth=2,
            #size="size"
        )
        plt.legend(bbox_to_anchor=(-0.05, 1), loc=3, borderaxespad=0.05, ncol=3, frameon=False)
        plt.tight_layout(pad=1.5)


        #.fig.legend(title='species', handles=handles, labels=labels, loc='upper center', ncol=1
        plt.xlim(df['p'].min(), df['p'].max())
        #plt.ylim(-0.01, 1.01)

        plt.ylabel(t)
        plt.savefig('output/individual_{}_{}.pdf'.format(d.replace(' ','_'), t))
        plt.close()

#print (df)