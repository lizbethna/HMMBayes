/*
Modelo Markov-Switching GARCH(1,1) 
Usando Modelos Ocultos de Markov (HMM) para cambiar de regimen entre dos GARCH(1,1) 
Usando errores con distribucion Normal 
*/

data {
  int<lower=0> T;    // Longitud de la serie (numero de observaciones)
  real y[T];         // Serie de tiempo a modelar (ej: retornos financieros)
}

parameters {
  // Parametros de los dos modelos GARCH 
  positive_ordered[2] alpha0; // Ordenar los parametros, para prevenir el problema label-switching (cambio de etiqueta)

  real<lower=0, upper=1> alpha1[2];
  real<lower=0, upper=1-alpha1[1]> beta1_1;
  real<lower=0, upper=1-alpha1[2]> beta1_2;

  // HMM probabilidades de transicion =>
  // Parametrizar por probabilidad de pertenencia en el estado del HMM (dos estados posibles)
  real<lower=0, upper=1> p_remain[2];
}

transformed parameters {
  // Parametros del GARCH 
  real<lower=0> beta1[2];

  // Vector de volatilidades GARCH instantaneas 
  vector[2] sigma_t[T];

  // Parametros del HMM 
  vector[2] log_alpha[T]; // Probabilidades de estado acumuladas (no normalizadas) 

  // Probabilidades de Transicion
  matrix[2, 2] P;
  P[1, 1] =  p_remain[1];
  P[1, 2] = 1 - p_remain[1];
  P[2, 1] = 1 - p_remain[2];
  P[2, 2] = p_remain[2];

  // Componentes GARCH 
  // ------------------

  // Valores de los parametros beta1  
  beta1[1] = beta1_1;
  beta1[2] = beta1_2;

  // Inicializar las varianzas no condicionales 
  sigma_t[1, 1] = alpha0[1] / (1 - alpha1[1] - beta1[1]); // Low-vol
  sigma_t[1, 2] = alpha0[2] / (1 - alpha1[2] - beta1[2]); // High-vol

  // Dinamica GARCH actualizando con el algoritmo forward
  for(t in 2:T){
    for(i in 1:2){
      sigma_t[t, i] = sqrt(alpha0[i] +
                           alpha1[i] * pow(y[t-1], 2) +
                           beta1[i] * pow(sigma_t[t-1, i], 2));
    }
  }

  // Componentes HMM 
  // ------------------

  { // Calcular log p(estado en t = j | historia hasta t) recursivamente
    // La propoiedad de Markov property permite hace actualizaciones a un paso 

// Inicio algoritmo Forward 

    real accumulator[2];

    // Suponga un distribucion inicial igual entre dos estados 
    // Un mejor modelo seria ponderar por medio de un HMM con distrbucion estacionaria 
    log_alpha[1, 1] = log(0.5) + normal_lpdf(y[1] | 0, sigma_t[1, 1]);
    log_alpha[1, 2] = log(0.5) + normal_lpdf(y[1] | 0, sigma_t[1, 2]);

    for(t in 2:T){
      for(j in 1:2) { // Estado actual 
        for(i in 1:2) { // Estado previo 
          accumulator[i] = log_alpha[t-1, i] + // Probabilidad de observaciones anteriores 
                           log(P[i, j]) + // Probabilidad de Transicion 
                           // (Local) Verosimilitud // evidencia para un estado dado  
                           normal_lpdf(y[t] | 0, sigma_t[t-1, i]);
        }
        log_alpha[t, j] = log_sum_exp(accumulator);
      }
    }
  } // Termino algoritmo Forward 
}

model {
  // Distributiones iniciales  

  // Componentes GARCH (iniciales debilmente informativas) 
  alpha0 ~ normal(0, 0.5); // Valor de referencia ~ 0.05
  alpha1 ~ normal(0, 1);
  beta1 ~ normal(1, 1); // Mayor presistencia de la volatilidad del termino MA de los modelos GARCH 

  // Componentes HMM 
  p_remain ~ beta(3, 1); // Inicial debilmente informativa 

  // Funcion de Verosimilitud
  target += log_sum_exp(log_alpha[T]); // Nota: la actualizacion se basa en el ultimo log_alpha 
}

generated quantities{
  vector[2] alpha[T];

  int<lower=1, upper=2> zstar[T];
  real logp_zstar;

// Inicio algoritmo Forward 
  for(t in 1:T){
    alpha[t] = softmax(log_alpha[t]);
  }
// Termino algoritmo Forward 

  { // Inicio algoritmo Viterbi 

    int bpointer[T, 2];             // apuntador del retroceso al estado anterior mas probable, en la ruta mas probable 
    real delta[T, 2];               // maxima probabilidad para la secuencia hasta t
                                    // con salida final del estado k para tiempo t

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
  } // Termino algoritmo Viterbi




}
