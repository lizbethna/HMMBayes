/*
Series de tiempo: modelo GARCH(1,1)
GARCH(1,1), con distribucion Normal(mu,sigma^2) 
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
    sigma[t] = sqrt(alpha0 + alpha1 * pow(rend[t - 1] - mu, 2) + beta1 * pow(sigma[t - 1], 2));  /// GARCH(1,1) 
} }

// Verosimilitud
model {
// distribuciones iniciales de los parametros 
  sigma1 ~ gamma(0.1, 0.1);
  mu ~ normal(0, 10);
  alpha0 ~ normal(0, 10);
  alpha1 ~ beta(2, 2);  
  beta1 ~ beta(2, 2);
//  funcion de verosimilitud 
  rend ~ normal(mu, sigma);   
}
