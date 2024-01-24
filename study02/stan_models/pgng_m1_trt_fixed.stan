data {

    // Metadata
    int<lower=1>  N; 
// Number of total observations
    array[N] int<lower=1>  J;               // Subject-indicator per observation
    array[N] int<lower=1>  K;  
    int  NJ = max(J);                       // Number of total subjects
// Bandit-indicator per observation
    array[N] int<lower=1, upper=2> M;       // Session-indicator per observation
    vector[NJ] age_zscore;
    vector[NJ] age_squared_zscore;

    // Data
    array[N] int<lower=0, upper=1>  Y;      // Response (go = 1, no-go = 0)
    array[N] int<lower=0, upper=1>  R;      // Outcome (better = 1, worse = 0)
    array[N] int<lower=0, upper=1>  V;      // Valence (positive = 1, negative = 0)

}
transformed data {

    int  NK = max(K);                       // Number of total bandits

}
parameters {

    // Participant parameters
    matrix[2,2]   theta_mu;                 // Population-level effects
    matrix[2,NJ]  theta_c_pr;               // Standardized subject-level effects (common)
    matrix[2,NJ]  theta_d_pr;               // Standardized subject-level effects (divergent)
    
    // Paramter variances
    matrix<lower=0>[2,2] sigma;             // Subject-level standard deviations
    cholesky_factor_corr[4] L;         // Cholesky factor of correlation matrix

}
transformed parameters {

    array[2] vector[NJ]  b1;                // Inverse temperature (positive valence)
    array[2] vector[NJ]  a1;                // Learning rate (positive valence)
    matrix[4,NJ]  theta_c_pr2;               // Standardized subject-level effects including age (common)
    theta_c_pr2[1,] = to_row_vector(age_zscore);
    theta_c_pr2[2,] = to_row_vector(age_squared_zscore);
    
    matrix<lower=0>[4,2] sigma_pr; 
    sigma_pr[1:2,1:2] = 1;                   //we dont want age to have any variance
    sigma_pr[3:4, ] = sigma;

    // Construction block
    {
    
    
    
    // Rotate random effects
    matrix[NJ,4] theta_c = transpose(diag_pre_multiply(sigma_pr[,1],L) * theta_c_pr2);
    matrix[NJ,2] theta_d = transpose(diag_pre_multiply(sigma[,2], theta_d_pr));
    
    // Construct random effects
    b1[1] = (theta_mu[1,1] + theta_c[,3] - theta_d[,1]) * 10;
    b1[2] = (theta_mu[1,2] + theta_c[,3] + theta_d[,1]) * 10;
    a1[1] = Phi_approx(theta_mu[2,1] + theta_c[,4] - theta_d[,2]);
    a1[2] = Phi_approx(theta_mu[2,2] + theta_c[,4] + theta_d[,2]);

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

    row_vector[2]  b1_mu = theta_mu[1] * 10;
    row_vector[2]  a1_mu = Phi_approx(theta_mu[2]);
    matrix[4,4] Omega=multiply_lower_tri_self_transpose(L);

}