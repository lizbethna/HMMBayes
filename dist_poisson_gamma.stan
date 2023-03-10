/*
Estadistica Bayesiana 
Distribucion Poisson-Gamma 
*/

// datos
data {
  int<lower = 0> N; // tamanio de muestra
  int<lower = 0> x[N]; // muestra observada
  real<lower=0> a0; // inicial theta ~ Gamma(a0,b0)
  real<lower=0> b0; 
}

// parametros 
parameters {
  real<lower = 0> theta ; // x ~ Poisson(theta)
}


model {
  theta ~ gamma(a0,b0); // distribucion inicial
  x ~ poisson(theta); // funcion de verosimilitud

}

// distribucion predictiva final 
generated quantities {
  int<lower = 0> x_star = poisson_rng(theta);

}




