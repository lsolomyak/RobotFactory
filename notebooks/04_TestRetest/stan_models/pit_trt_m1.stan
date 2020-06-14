// Pavlovian Instrumental Transfer (PIT) Task
// Model variant 1 
//
// Parameters (4): 
//   - beta: inverse temperature
//   - eta:  learning rate
//   - tau:  go bias
//   - nu:   Pavlovian bias
//
// Notes:
//   - Model is vectorized such that it iterates over all participants/arms
//   - Model requires no missing data 
//   - Requires mapping between participants (N) and data (H)
//   - Pavlovian bias constant across trials

data {

    // Metadata
    int  H;                         // Number of participants/blocks/arms
    int  T;                         // Number of trials
    
    // Mappings
    int        sub_ix[H];           // Trial-to-participant mapping
    vector[H]  pav_ix;              // Stimulus-to-valence mapping [Gain = 1, Loss = -1]
    
    // Data
    int       Y[2,T,H];             // Choices (Go = 1, No-Go = 0)
    vector[H] R[2,T];               // Rewards (-1, 0, 1)
    
}
transformed data {

    // Number of participants
    int  N = max(sub_ix);
    
    // Vectorized choices
    vector[H]  y[2,T];
    for (i in 1:2) {
        for (j in 1:T) {
            y[i,j] = to_vector(Y[i,j]);
        }
    }

}
parameters {
    
    // Group-level parameters (pre-transform)
    matrix[4,2]  mu_pr;
    
    // Subject-level parameters (pre-transform)
    matrix[4,N]  theta_c_pr;
    matrix[4,N]  theta_d_pr;
    
    // Parameter covariance
    cholesky_factor_corr[4]  L_c;
    cholesky_factor_corr[4]  L_d;
    vector[4]  sigma_c_pr;
    vector[4]  sigma_d_pr;

}
model {
       
    // Generated quantities
    vector[N]  beta[2];
    vector[N]  eta[2];
    vector[N]  tau[2];
    vector[N]  nu[2];
       
    // Rotate random effects
    matrix[4,N]  theta_c = diag_pre_multiply( exp(sigma_c_pr), L_c ) * theta_c_pr;
    matrix[4,N]  theta_d = diag_pre_multiply( exp(sigma_d_pr), L_d ) * theta_d_pr;

    // Construct individual-level parameters (session 1)
    beta[1] = (mu_pr[1,1] + theta_c[1,:]' - theta_d[1,:]') * 10;
    eta[1] = Phi_approx(mu_pr[2,1] + theta_c[2,:]' - theta_d[2,:]');
    tau[1] = mu_pr[3,1] + theta_c[3,:]' - theta_d[3,:]';
    nu[1]  = mu_pr[4,1] + theta_c[4,:]' - theta_d[4,:]';

    // Construct individual-level parameters (session 2)
    beta[2] = (mu_pr[1,2] + theta_c[1,:]' + theta_d[1,:]') * 10;
    eta[2] = Phi_approx(mu_pr[2,2] + theta_c[2,:]' + theta_d[2,:]');
    tau[2] = mu_pr[3,2] + theta_c[3,:]' + theta_d[3,:]';
    nu[2]  = mu_pr[4,2] + theta_c[4,:]' + theta_d[4,:]';
       
    // Priors
    to_vector(mu_pr) ~ normal(0, 2);
    to_vector(theta_c_pr) ~ normal(0, 1);
    to_vector(theta_d_pr) ~ normal(0, 1);
    
    L_c ~ lkj_corr_cholesky(2.0);
    L_d ~ lkj_corr_cholesky(2.0);
    sigma_c_pr ~ normal(0, 1);
    sigma_d_pr ~ normal(0, 1);
    
    // Main loop
    for (i in 1:2) {
    
        // Generated quantities
        matrix[H,T]  p  = rep_matrix(0, H, T);   // Go probability
        vector[H]    Q1 = pav_ix * 0.5;          // State-action values (go)
        vector[H]    Q2 = pav_ix * 0.5;          // State-action values (no-go)
        vector[H]    V  = pav_ix;                // State values

        // Parameter expansion
        vector[H]  beta_vec = beta[i][sub_ix];
        vector[H]  eta_vec = eta[i][sub_ix];
        vector[H]  tau_vec = tau[i][sub_ix];
        vector[H]  nu_vec  = nu[i][sub_ix];
    
        for (j in 1:T) {

            // Compute likelihood of acting (go)
            p[:,j] = beta_vec .* (Q1 - Q2 + tau_vec + nu_vec .* V );

            // Update action value (go)
            Q1 += y[i,j] .* ( eta_vec .* ( R[i,j] - Q1 ) );

            // Update action value (no-go)
            Q2 += (1-y[i,j]) .* ( eta_vec .* ( R[i,j] - Q2 ) );

        }

        // Likelihood
        to_array_1d(Y[i]) ~ bernoulli_logit( to_vector(p) );
        
    }
        
}
generated quantities {
    
    // Test-retest parameters
    vector[4] sigma_c = exp(sigma_c_pr) .* exp(sigma_c_pr);
    vector[4] sigma_d = exp(sigma_d_pr) .* exp(sigma_d_pr);
    vector[4] rho = (sigma_c - sigma_d) ./ (sigma_c + sigma_d);

}