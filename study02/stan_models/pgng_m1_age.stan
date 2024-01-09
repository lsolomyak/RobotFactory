data {

    // Metadata
    int<lower=1>  N;       
    int<lower=1> NJ;                      // Number of total observations
// Number of total observations
    array[N] int<lower=1>  J;               // Subject-indicator per observation
    array[N] int<lower=1>  K;               // Bandit-indicator per observation
    vector[NJ] age_zscore;
    vector[NJ] age_squared_zscore;

    // Data
    array[N] int<lower=0, upper=1>  Y;      // Response (go = 1, no-go = 0)
    array[N] int<lower=0, upper=1>  R;      // Outcome (better = 1, worse = 0)

}
transformed data {

    int  NK = max(K);                       // Number of total bandits

}
parameters {

    // Participant parameters
    vector[2]     theta_mu;                 // Population-level effects
    matrix[2,NJ]  theta_pr;                 // Standardized subject-level effects
    
    // Paramter variances
    vector<lower=0>[2] sigma;               // Subject-level standard deviations
    cholesky_factor_corr[4] L;         // Cholesky factor of correlation matrix

}
transformed parameters {

    vector[NJ]  b1;                         // Inverse temperature
    vector[NJ]  a1;  
    vector<lower=0>[4] sigma_pr;
    matrix[4,NJ] theta_pr2;// Learning rate
    sigma_pr[1] = 1;
    sigma_pr[2] = 1;
    sigma_pr[3:4] = sigma; 
    theta_pr2[1,] = to_row_vector(age_zscore);
    theta_pr2[2,] = to_row_vector(age_squared_zscore);
    theta_pr2[3:4,] = theta_pr;
    // Construction block
    {
    
    // Rotate random effects
    matrix[NJ,4] theta = transpose(diag_pre_multiply(sigma_pr,L) * theta_pr2);

    // Construct random effects
    b1 = (theta_mu[1] + theta[,3]) * 10;
    a1 = Phi_approx(theta_mu[2] + theta[,4]);
    
    }

}
model {

    // Initialize Q-values
    array[NJ, NK, 2] real Q = rep_array(0.5, NJ, NK, 2);

    // Construct linear predictor
    vector[N] mu;
    for (n in 1:N) {
    
        // Assign trial-level parameters
        real beta = b1[J[n]];
        real eta  = a1[J[n]];

        // Compute (scaled) difference in state-action values
        mu[n] = beta * (Q[J[n],K[n],2] - Q[J[n],K[n],1]);
        
        // Compute prediction error
        real delta = R[n] - Q[J[n],K[n],Y[n]+1];
                
        // Update state-action values
        Q[J[n],K[n],Y[n]+1] += eta * delta;
        
    }
    
    // Likelihood
    target += bernoulli_logit_lpmf(Y | mu); 
    
    // Priors
    target += std_normal_lpdf(theta_mu);
    target += std_normal_lpdf(to_vector(theta_pr));
    target += student_t_lpdf(sigma | 3, 0, 1);
    target += lkj_corr_cholesky_lpdf(L|23);

}
generated quantities {

    real  b1_mu = theta_mu[1] * 10;
    real  a1_mu = Phi_approx(theta_mu[2]);
    matrix[4,4] Omega=multiply_lower_tri_self_transpose(L);

}