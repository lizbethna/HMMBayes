/*
A Markov Switching GARCH Model
*/

data {
  int<lower=0> T;    // Length of series (number of observations)
  real y[T];         // Return series
}

parameters {
  // GARCH Parameters
  positive_ordered[2] alpha0; // Ordering prevents label switching

  real<lower=0, upper=1> alpha1[2];
  real<lower=0, upper=1-alpha1[1]> beta1_1;
  real<lower=0, upper=1-alpha1[2]> beta1_2;

  // HMM Transition Probabilities =>
  // Parameterize by probability of staying in state
  real<lower=0, upper=1> p_remain[2];
}

transformed parameters {
  // GARCH Parameters
  real<lower=0> beta1[2];

  // Vector of instantaneous GARCH volatilities
  vector[2] sigma_t[T];

  // HMM Parameters
  vector[2] log_alpha[T]; // Accumulated (unnormalized) state probabilities

  // Transition probabilities
  matrix[2, 2] P;
  P[1, 1] =  p_remain[1];
  P[1, 2] = 1 - p_remain[1];
  P[2, 1] = 1 - p_remain[2];
  P[2, 2] = p_remain[2];

  // GARCH Component
  // ------------------

  // Load beta1 from parameters
  beta1[1] = beta1_1;
  beta1[2] = beta1_2;

  // Initialize at unconditional variances
  sigma_t[1, 1] = alpha0[1] / (1 - alpha1[1] - beta1[1]); // Low-vol
  sigma_t[1, 2] = alpha0[2] / (1 - alpha1[2] - beta1[2]); // High-vol

  // GARCH dynamics rolling forward
  for(t in 2:T){
    for(i in 1:2){
      sigma_t[t, i] = sqrt(alpha0[i] +
                           alpha1[i] * pow(y[t-1], 2) +
                           beta1[i] * pow(sigma_t[t-1, i], 2));
    }
  }

  // HMM Component
  // ------------------

  { // Calculate log p(state at t = j | history up to t) recursively
    // Markov property allows us to do one-step updates

// Inicio Forward algorithm 

    real accumulator[2];

    // Assume initial equal distribution among two states
    // Better model would be to weight by HMM stationary distribution
    log_alpha[1, 1] = log(0.5) + normal_lpdf(y[1] | 0, sigma_t[1, 1]);
    log_alpha[1, 2] = log(0.5) + normal_lpdf(y[1] | 0, sigma_t[1, 2]);

    for(t in 2:T){
      for(j in 1:2) { // Current state
        for(i in 1:2) { // Previous state
          accumulator[i] = log_alpha[t-1, i] + // Probability from previous obs
                           log(P[i, j]) + // Transition probability
                           // (Local) likelihood / evidence for given state
                           normal_lpdf(y[t] | 0, sigma_t[t-1, i]);
        }
        log_alpha[t, j] = log_sum_exp(accumulator);
      }
    }
  } // Termino Forward algorithm
}

model {
  // Priors

  // GARCH components (weakly informative)
  alpha0 ~ normal(0, 0.5); // Baseline vol is ~ 0.05
  alpha1 ~ normal(0, 1);
  beta1 ~ normal(1, 1); // Most volatility persistance from MA term of GARCH models

  // HMM components
  p_remain ~ beta(3, 1); // Weakly informative way to say that states are sort-of sticky

  // Likelihood
  target += log_sum_exp(log_alpha[T]); // Note: update based only on last log_alpha
}

generated quantities{
  vector[2] alpha[T];

  int<lower=1, upper=2> zstar[T];
  real logp_zstar;

// Inicio Forward algortihm
  for(t in 1:T){
    alpha[t] = softmax(log_alpha[t]);
  }
// Termino Forward algortihm

  { // Inicio Viterbi algorithm

    int bpointer[T, 2];             // backpointer to the most likely previous state on the most probable path
    real delta[T, 2];               // max prob for the seq up to t
                                    // with final output from state k for time t

    for (j in 1:2)
      delta[1, 1] = normal_lpdf(y[1] | 0, sigma_t[1, 1]);
      delta[1, 2] = normal_lpdf(y[1] | 0, sigma_t[1, 2]);

    for (t in 2:T) {
      for (j in 1:2) { // j = current (t)
        delta[t, j] = negative_infinity();
        for (i in 1:2) { // i = previous (t-1)
          real logp;
          logp = delta[t-1, i] + log(P[i, j]) + normal_lpdf(y[t] | 0, sigma_t[t-1, i]);
          if (logp > delta[t, j]) {
            bpointer[t, j] = i;
            delta[t, j] = logp;
          }
        }
      }
    }

    logp_zstar = max(delta[T]);

    for (j in 1:2)
      if (delta[T, j] == logp_zstar)
        zstar[T] = j;

    for (t in 1:(T - 1)) {
      zstar[T - t] = bpointer[T - t + 1, zstar[T - t + 1]];
    }
  }




}
