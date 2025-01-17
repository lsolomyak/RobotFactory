import os, sys
import numpy as np
from os.path import dirname
from pandas import read_csv
from cmdstanpy import CmdStanModel
ROOT_DIR = dirname(dirname(os.path.realpath(__file__)))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Define parameters.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## I/O parameters.
session = sys.argv[2]
stan_models = sys.argv[1].split(',')
subs = int(sys.argv[3])


## Sampling parameters.
iter_warmup   = 5000
iter_sampling = 5250
chains = 4
thin = 1
parallel_chains = 4

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Load and prepare data.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Load data.
data = read_csv(os.path.join(ROOT_DIR, 'data', session, 'pgng.csv'))

## Restrict participants.
#if session == 's1':
    reject = read_csv(os.path.join(ROOT_DIR, 'data', session, 'reject.csv'))
    data = data[data.subject.isin(reject.query('reject==0').subject)].reset_index(drop=True)

## Format data.
unique_subjects = data['subject'].unique()[:subs]


data['valence'] = data.valence.replace({'win': 1, 'lose': 0})
data['outcome'] = np.where(data.valence, data.outcome > 5, data.outcome > -5).astype(int)


grouped_data = data.groupby('subject').first().reset_index()

# Calculate z-score for age in the grouped data
age_zscore = (grouped_data['age'] - grouped_data['age'].mean()) / grouped_data['age'].std()
grouped_data['age_zscore']=age_zscore
grouped_data['age_squared']=grouped_data['age_zscore'] ** 2
age_squared_zscore = (grouped_data['age_squared'] - grouped_data['age_squared'].mean()) / grouped_data['age_squared'].std()
print(np.corrcoef(age_squared_zscore,age_zscore))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Assemble data for Stan.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Define metadata.
N = len(data)
J = np.unique(data.subject, return_inverse=True)[-1] + 1
K = np.unique(data.stimulus, return_inverse=True)[-1] + 1
M = np.unique(data.runsheet, return_inverse=True)[-1] + 1

## Define data.
Y = data.choice.values.astype(int)
R = data.outcome.values.astype(int)
V = data.valence.values.astype(int)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Fit Stan Model.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Assemble data.
dd = dict(N=N, J=J, K=K, M=M, Y=Y, R=R, V=V, age_squared_zscored=age_squared_zscore,age_zscored=age_zscore)

## Load StanModel
for stan_model in stan_models:

    StanModel = CmdStanModel(stan_file=os.path.join(ROOT_DIR, 'stan_models', f'{stan_model}.stan'))

    ## Fit Stan model.
    StanFit = StanModel.sample(data=dd, chains=chains, iter_warmup=iter_warmup, iter_sampling=iter_sampling, thin=thin, parallel_chains=parallel_chains, seed=0, show_progress=True)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Save samples.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
print('Saving data.')

## Define fout.
fout = os.path.join(ROOT_DIR, 'stan_results', session, stan_model)
    
## Extract summary and samples.
summary = StanFit.summary(percentiles=(2.5, 50, 97.5), sig_figs=3)
samples = StanFit.draws_pd()
    
## Define columns to save.
cols = np.concatenate([
    
    ## Diagnostic variables.
    samples.filter(regex='__').columns,
    
    ## Regression effects (population-level).
    samples.filter(regex='[a,b,c][0-9]_mu').columns,
        
    ## Variances (group-level).
    samples.filter(regex='sigma').columns,
    
    ## Regression effects (group-level).
    samples.filter(regex='[a,b,c][0-9]\[').columns,
    
])
        
## Save.
samples[cols].to_csv(f'{fout}.tsv.gz', sep='\t', index=False, compression='gzip')
summary.loc[samples[cols].filter(regex='[^__]$').columns].to_csv(f'{fout}_summary.tsv', sep='\t')