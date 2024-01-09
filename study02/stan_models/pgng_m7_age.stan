data {

    // Metadata
    int<lower=1>  N; 
    int<lower=1> NJ;                      // Number of total observations
    array[N] int<lower=1>  J;               // Subject-indicator per observation
    array[N] int<lower=1>  K;               // Bandit-indicator per observation
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
    vector[7]     theta_mu;                 // Population-level effects
    matrix[7,NJ]  theta_pr;                 // Standardized subject-level effects
    
    // Paramter variances
    vector<lower=0>[7] sigma;               // Subject-level standard deviations
    cholesky_factor_corr[9] L;         // Cholesky factor of correlation matrix

}
transformed parameters {

    vector[NJ]  b1;                         // Inverse temperature (positive valence)
    vector[NJ]  b2;                         // Inverse temperature (negative valence)
    vector[NJ]  b3;                         // Go bias (positive valence)
    vector[NJ]  b4;                         // Go bias (negative valence)
    vector[NJ]  a1;                         // Learning rate (positive valence)
    vector[NJ]  a2;                         // Learning rate (negative valence)
    vector[NJ]  c1;                         // Lapse rate
    vector<lower=0>[9] sigma_pr;
    matrix<lower=0>[9,NJ] theta_pr2;

    sigma_pr[1] = 1;
    sigma_pr[2] = 1;
    sigma_pr[3:9] = sigma; 
    theta_pr2[1,] = to_row_vector(age_zscore);
    theta_pr2[2,] = to_row_vector(age_squared_zscore);
    theta_pr2[3:9,] = theta_pr;
    // Construction block
    {
    
    // Rotate random effects
    matrix[NJ,9] theta = transpose(diag_pre_multiply(sigma_pr,L) * theta_pr);
    // Construct random effects
    b1 = (theta_mu[1] + theta[,3]) * 10;
    b2 = (theta_mu[2] + theta[,4]) * 10;
    b3 = (theta_mu[3] + theta[,5]) * 5;
    b4 = (theta_mu[4] + theta[,6]) * 5;
    a1 = Phi_approx(theta_mu[7] + theta[,5]);
    a2 = Phi_approx(theta_mu[8] + theta[,6]);
    c1 = Phi_approx(-2.0 + theta_mu[9] + 0.5 * theta[,7]);
    
    }

}
model {

    // Initialize Q-values
    array[NJ, NK, 2] real Q = rep_array(0.5, NJ, NK, 2);

    // Construct linear predictor
    vector[N] mu;
    for (n in 1:N) {
    
        // Assign trial-level parameters
        real beta = (V[n] == 1) ? b1[J[n]] : b2[J[n]];
        real tau  = (V[n] == 1) ? b3[J[n]] : b4[J[n]];
        real eta  = (V[n] == 1) ? a1[J[n]] : a2[J[n]];
        real xi   = c1[J[n]];

        // Compute (scaled) difference in state-action values
        mu[n] = (0.5 * xi) + (1-xi) * inv_logit(beta * (Q[J[n],K[n],2] - Q[J[n],K[n],1]) + tau);
        
        // Compute prediction error
        real delta = R[n] - Q[J[n],K[n],Y[n]+1];
                
        // Update state-action values
        Q[J[n],K[n],Y[n]+1] += eta * delta;
        
    }
    
    // Likelihood
    target += bernoulli_lpmf(Y | mu);
    
    // Priors
    target += std_normal_lpdf(theta_mu);
    target += std_normal_lpdf(to_vector(theta_pr));
    target += student_t_lpdf(sigma | 3, 0, 1);
    target += lkj_corr_cholesky_lpdf(L|23);

}
generated quantities {


    real  b1_mu = theta_mu[1] * 10;
    real  b2_mu = theta_mu[2] * 10;
    real  b3_mu = theta_mu[3] * 5;
    real  b4_mu = theta_mu[4] * 5;
    real  a1_mu = Phi_approx(theta_mu[5]);
    real  a2_mu = Phi_approx(theta_mu[6]);
    real  c1_mu = Phi_approx(-2.0 + theta_mu[7]);

    matrix[9,9] Omega=multiply_lower_tri_self_transpose(L);

}