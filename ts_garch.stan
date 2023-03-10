/*
Sériese tiempo: modelo GARCH 
*/

// datos 
data { 
  int<lower=0> N; 
  real rend[N]; 
  real<lower=0> sigma1;
}

// parametros 
parameters {
  real mu;
  real<lower=0> alpha0;
  real<lower=0, upper=1> alpha1; 
  real<lower=0, upper=(1-alpha1)> beta1;
}

transformed parameters {
  real<lower=0> sigma[N]; 
  sigma[1] = sigma1;
  for (t in 2:N) {
    sigma[t] = sqrt(alpha0 + alpha1 * pow(rend[t - 1] - mu, 2) + beta1 * pow(sigma[t - 1], 2));
} }

// Verosimilitud
model {
  rend ~ normal(mu, sigma);
}
