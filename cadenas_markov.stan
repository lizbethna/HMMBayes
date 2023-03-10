/*
Cadena de Markov 
*/


data {
  int<lower=1> N;                   // numero de observaciones (longitud)
  int<lower=1> K;                   // numero de estados
  int<lower=1, upper=K> z[N];                        // observaciones
//  simplex[K] di1;                   // probabilidades de los estados inicial 
  vector<lower=0>[K] alpha;
}

parameters {
  simplex[K] theta[K];                  // probabilidades de transicion
                                    // A[i][j] = p(z_t = j | z_{t-1} = i)
}


model {
  for(j in 1:K){
    theta[j] ~ dirichlet(alpha); 
  } 
//  z[1] ~ categorical(di1);
  for(t in 2:N){
    z[t] ~ categorical(theta[z[t-1]]);
  }
}




