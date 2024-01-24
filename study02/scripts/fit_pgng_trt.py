import os, sys
import numpy as np
from os.path import dirname
from pandas import read_csv, concat
from cmdstanpy import CmdStanModel
ROOT_DIR = dirname(dirname(os.path.realpath(__file__)))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Define parameters.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## I/O parameters.
stan_model = sys.argv[1]
pairing = int(sys.argv[2])
subs = int(sys.argv[3])
print(subs)
## Sampling parameters.
iter_warmup   = 6000
iter_sampling = 7000
chains = 4
thin = 1
parallel_chains = 4

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Load and prepare data.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Load data
if pairing == 1:
    a = read_csv(os.path.join(ROOT_DIR, 'data', 's1', 'pgng.csv'))
    b = read_csv(os.path.join(ROOT_DIR, 'data', 's2', 'pgng.csv'))
elif pairing == 2:
    a = read_csv(os.path.join(ROOT_DIR, 'data', 's1', 'pgng.csv'))
    b = read_csv(os.path.join(ROOT_DIR, 'data', 's3', 'pgng.csv'))
elif pairing == 3:
    a = read_csv(os.path.join(ROOT_DIR, 'data', 's2', 'pgng.csv'))
    b = read_csv(os.path.join(ROOT_DIR, 'data', 's3', 'pgng.csv'))

## Merge datasets.
data = concat([a, b])

#unique_subjects = data['subject'].unique()
unique_subjects = data['subject'].unique()[:subs]

data = data[data['subject'].isin(unique_subjects)]

## Restrict to participants with both sessions.
data = data.groupby('subject').filter(lambda x: x.session.nunique() == 2)

reject = read_csv(os.path.join(ROOT_DIR, 'data', 's1', 'reject.csv'))
reject2 = read_csv(os.path.join(ROOT_DIR, 'data', 's2', 'reject.csv'))

data = data[data.subject.isin(reject.query('reject==0').subject)].reset_index(drop=True)
data = data[data.subject.isin(reject2.query('reject==0').subject)].reset_index(drop=True)


## Format data.
data['valence'] = data.valence.replace({'win': 1, 'lose': 0})
data['outcome'] = np.where(data.valence, data.outcome > 5, data.outcome > -5).astype(int)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Assemble data for Stan.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Define metadata.
N = len(data)
J = np.unique(data.subject, return_inverse=True)[-1] + 1
K = np.unique(data.stimulus, return_inverse=True)[-1] + 1
M = np.unique(data.session, return_inverse=True)[-1] + 1

## Define data.
Y = data.choice.values.astype(int)
R = data.outcome.values.astype(int)
V = data.valence.values.astype(int)



grouped_data = data.groupby('subject').first().reset_index()

# Calculate z-score for age in the grouped data
age_zscore = (grouped_data['age'] - grouped_data['age'].mean()) / grouped_data['age'].std()
grouped_data['age_zscore']=age_zscore
grouped_data['age_squared']=grouped_data['age_zscore'] ** 2
age_squared_zscore = (grouped_data['age_squared'] - grouped_data['age_squared'].mean()) / grouped_data['age_squared'].std()
print(np.corrcoef(age_squared_zscore,age_zscore))
NJ=data.subject.nunique()



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Fit Stan Model.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Assemble data.
dd = dict(N=N, J=J, K=K, M=M, Y=Y, R=R, V=V,NJ=NJ, age_squared_zscore=age_squared_zscore,age_zscore=age_zscore)

## Load StanModel
StanModel = CmdStanModel(stan_file=os.path.join(ROOT_DIR, 'stan_models', f'{stan_model}.stan'))

## Fit Stan model.
StanFit = StanModel.sample(data=dd, chains=chains, iter_warmup=iter_warmup, iter_sampling=iter_sampling, thin=thin, parallel_chains=parallel_chains,adapt_delta=.9,seed=0, show_progress=True)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Save samples.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
print('Saving data.')

## Define fout.
fout = os.path.join(ROOT_DIR, 'stan_results', stan_model.replace('sh', f'trt{pairing}'))

summary = StanFit.summary(percentiles=(2.5, 50, 97.5), sig_figs=3)

summary.to_csv(f'{fout}_{subs}_summary.tsv', sep='\t')

## Extract summary and samples.
samples = StanFit.draws_pd()
    
## Define columns to save.
summary = summary.T.filter(regex='theta|sigma').T    # Only untransformed parameters.

## Identify number of divergences.
    
## Identify parameters failing to reach convergence.
rhat = len(summary.query('R_hat > 1.01'))
print('rhat',rhat)
n_eff = len(summary.query('N_Eff < 400'))
print('n_eff',n_eff)

## Extract and save samples.
samples = StanFit.draws_pd()
divergence = samples.divergent__.sum()
print('divergence',divergence)
# Assuming `summary` is the DataFrame containing the Stan summary


## Save.
samples.to_csv(f'{fout}.tsv.gz', sep='\t', index=False, compression='gzip')
#