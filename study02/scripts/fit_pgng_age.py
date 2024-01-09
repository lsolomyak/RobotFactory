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
print(subs)

## Sampling parameters.
iter_warmup   = 5000
iter_sampling = 5000
chains = 4
thin = 1
parallel_chains = 4

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Load and prepare data.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Load data.
data = read_csv(os.path.join(ROOT_DIR, 'data', session, 'pgng.csv'))

## Restrict participants.
if session == 's1':
    reject = read_csv(os.path.join(ROOT_DIR, 'data', session, 'reject.csv'))
    data = data[data.subject.isin(reject.query('reject==0').subject)].reset_index(drop=True)




## Format data.
unique_subjects = data['subject'].unique()[:subs]
data = data[data['subject'].isin(unique_subjects)]

subject_to_exclude_61 = unique_subjects[23]



# Exclude data associated with the 129th and 61st unique subjects
data = data[(data['subject'] != subject_to_exclude_61)]

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
NJ=data.subject.nunique()

## Define data.
Y = data.choice.values.astype(int)
R = data.outcome.values.astype(int)
V = data.valence.values.astype(int)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Fit Stan Model.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Assemble data.
dd = dict(N=N, J=J, K=K, M=M, Y=Y, R=R, V=V,NJ=NJ, age_squared_zscore=age_squared_zscore,age_zscore=age_zscore)

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
    summary.to_csv(os.path.join(ROOT_DIR, 'stan_results', session, f'{stan_model}_{subs}_summary.tsv'), sep='\t')

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



    samples.to_csv(os.path.join(ROOT_DIR, 'stan_results', session, f'{stan_model}_{subs}.tsv.gz'), sep='\t', index=False, compression='gzip')