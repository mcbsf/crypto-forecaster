from . import timeseries, retriever
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import coint


df = retriever.get_data('bitcoin')
price_header = 'open'

start_date = pd.to_datetime('2013-12-01')
end_date = pd.to_datetime('2016-02-01')

mask = (df['date'] > start_date) & (df['date'] <= end_date)

# df[price_header] = stats.zscore(df[price_header])
# df['positive_reply'] = stats.zscore(df['positive_reply'])
# df['very_negative_reply'] = stats.zscore(df['very_negative_reply'])

# df[price_header] = timeseries.standardize_laggedly(df[price_header])
# df['positive_reply'] = timeseries.standardize_laggedly(df['positive_reply'])
# df['very_negative_reply'] = timeseries.standardize_laggedly(df['very_negative_reply'])

df[price_header] = timeseries.first_differentiate(df[price_header])
df['positive_reply'] = timeseries.first_differentiate(df['positive_reply'])
df['very_negative_reply'] = timeseries.first_differentiate(df['very_negative_reply'])

# Drop nan values
df = df.dropna()

# Metrics and plots
timeseries.metrics(df[price_header])
timeseries.plot(df['date'], df[price_header], '', '')

timeseries.metrics(df['positive_reply'])
timeseries.plot(df['date'], df['positive_reply'], '', '')

timeseries.metrics(df['very_negative_reply'])
timeseries.plot(df['date'], df['very_negative_reply'], '', '')

restricted_df = df.loc[mask]

# Standalone Granger causality
stack = np.column_stack((restricted_df[price_header], restricted_df['positive_reply']))
res = grangercausalitytests(stack, 13)

print('\n\n\n\n\n')
stack = np.column_stack((restricted_df[price_header], restricted_df['very_negative_reply']))
res = grangercausalitytests(stack, 13)

print(coint(restricted_df[price_header], restricted_df['positive_reply'], maxlag=2, autolag=None))

# VAR model Granger causality
# mdata = restricted_df[['price', 'positive_reply', 'very_negative_reply']]
# mdata.index = pd.DatetimeIndex(restricted_df['date'])

# model = VAR(mdata)

# results = model.fit(13)

# results.summary()
# results.plot().show()

# results.plot_acorr().show()

# tres = results.test_causality('price', ['positive_reply'], kind='f')
# print('tres.test_statistic:', tres.test_statistic)
# print('tres.crit_value:', tres.crit_value)
# print('tres.pvalue:', tres.pvalue)
# print('tres.df:', tres.df)

# tres = results.test_causality('price', ['very_negative_reply'], kind='f')
# print('tres.test_statistic:', tres.test_statistic)
# print('tres.crit_value:', tres.crit_value)
# print('tres.pvalue:', tres.pvalue)
# print('tres.df:', tres.df)

# tres = results.test_causality('price', ['positive_reply', 'very_negative_reply'], kind='f')
# print('tres.test_statistic:', tres.test_statistic)
# print('tres.crit_value:', tres.crit_value)
# print('tres.pvalue:', tres.pvalue)
# print('tres.df:', tres.df)

# tres = results.test_causality('price', ['positive_reply', 'very_negative_reply'], kind='wald')
# print('tres.test_statistic:', tres.test_statistic)
# print('tres.crit_value:', tres.crit_value)
# print('tres.pvalue:', tres.pvalue)
# print('tres.df:', tres.df)