import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import coint

from .util import context
from .util import retriever

from forecaster import timeseries


df = retriever.get_data('bitcoin')
price_header = 'price'

for header in df.column_header:
    if header is not 'date':
        df[header] = timeseries.standardize_laggedly(df[header])

df['positive_reply'] = timeseries.standardize_laggedly(df['positive_reply'])
df['positive_topic'] = timeseries.standardize_laggedly(df['positive_topic'])

# Drop nan values
df = df.dropna()

restricted_df = df#.loc[mask]

# Metrics and plots
timeseries.metrics(restricted_df[price_header])
timeseries.plot(restricted_df['date'], restricted_df[price_header], '', '')

timeseries.metrics(restricted_df['positive_reply'])
timeseries.plot(restricted_df['date'], restricted_df['positive_reply'], '', '')

timeseries.metrics(restricted_df['positive_topic'])
timeseries.plot(restricted_df['date'], restricted_df['positive_topic'], '', '')

# Standalone Granger causality
stack = np.column_stack((restricted_df[price_header], restricted_df['positive_reply']))
res = grangercausalitytests(stack, 13)

print('\n\n\n\n\n')
stack = np.column_stack((restricted_df[price_header], restricted_df['positive_topic']))
res = grangercausalitytests(stack, 13)

print(coint(restricted_df[price_header], restricted_df['positive_reply'], maxlag=2, autolag=None))

# VAR model Granger causality
# mdata = restricted_df[['price', 'positive_reply', 'positive_topic']]
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

# tres = results.test_causality('price', ['positive_topic'], kind='f')
# print('tres.test_statistic:', tres.test_statistic)
# print('tres.crit_value:', tres.crit_value)
# print('tres.pvalue:', tres.pvalue)
# print('tres.df:', tres.df)

# tres = results.test_causality('price', ['positive_reply', 'positive_topic'], kind='f')
# print('tres.test_statistic:', tres.test_statistic)
# print('tres.crit_value:', tres.crit_value)
# print('tres.pvalue:', tres.pvalue)
# print('tres.df:', tres.df)

# tres = results.test_causality('price', ['positive_reply', 'positive_topic'], kind='wald')
# print('tres.test_statistic:', tres.test_statistic)
# print('tres.crit_value:', tres.crit_value)
# print('tres.pvalue:', tres.pvalue)
# print('tres.df:', tres.df)