# ------------------------------------------------------------------------------------------------------------

# Simulação das Distribuições pelo método IWLS

# ------------------------------------------------------------------------------------------------------------

# Carrega Pacotes

# ------------------------------------------------------------------------------------------------------------

# Importa Pacotes
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ------------------------------------------------------------------------------------------------------------

# Define funções

# ------------------------------------------------------------------------------------------------------------

# Declara função calcula W 
def calcula_W(eta, theta):
  
  # Calcula W
  return (1 / theta) * np.exp(2 * eta)

# ------------------------------------------------------------------------------------------------------------

# Declara função Z
def calcula_Z(eta, Y):
  
  # Calcula Z
  return eta + ( Y * np.exp(-eta) ) - 1 

# ------------------------------------------------------------------------------------------------------------

# Calcula theta
def calcula_theta(eta, dist):
  
  # Condição de distribuição
  if dist == "poisson":
    
    # Retorno da função
    return np.exp(eta)
  
  elif dist == "binomial":
    
    # Retorno da função
    return np.exp(eta) / (1 + np.exp(eta))
    
# ------------------------------------------------------------------------------------------------------------

# Define função de multiplicação matricial
def mat(A, B): 
  return np.matmul(A,B)

# Define função de transposição de matriz
def t(A):
  return np.transpose(A)

# Define função de invenrsa de matriz
def inv(A):
  return np.linalg.inv(A)

# ------------------------------------------------------------------------------------------------------------

# Funcao estimativa de betas
def EMV(Y, Xs, beta0, tol, norma, dist):
  
  # Declara betas
  betas = pd.DataFrame({"it" : [1],
                        "beta0" : beta0[0],
                        "beta1" : beta0[1]})
                        
  # Declara norma
  norma = norma
  
  # Declara iteracao
  r = 0
  
  while norma > tol:
    
    # Declara beta da vez
    beta = np.array(betas.iloc[r,[1,2]])
    
    # Calcula eta0
    eta = mat(Xs, beta)

    # Calcula media inicial
    theta = calcula_theta(eta, dist)
    
    # Calcula diagonal da matriz Wi
    W_i = calcula_W(eta, theta)
    
    # Declara matriz W
    W = np.eye(Y.shape[0])
    np.fill_diagonal(W, W_i)

    # Calcula z
    z_i = calcula_Z(eta, Y)
    
    # Calcula beta_0
    b0 = mat( mat( mat( inv( mat( mat( t(Xs), W) , Xs) ), t(Xs) ), W), z_i)[0]
    
    # Calcula beta_1
    b1 = mat( mat( mat( inv( mat( mat( t(Xs), W) , Xs) ), t(Xs) ), W), z_i)[1]
    
    # Gera data frame de novos betas
    df_temp = pd.DataFrame({"it" : [r+1],
                            "beta0" : [b0],
                            "beta1" : [b1]})
                            
    # Incrementa data frame
    betas = betas.append(df_temp, ignore_index=True)
    
    # Calcula norma euclideana
    norma = np.sqrt(sum((abs(betas.iloc[r+1, [1,2]] - betas.iloc[r, [1,2]]))**2))
    
    # Incrementa r
    r = r + 1
  
  # Retorno da função
  return betas 

# ------------------------------------------------------------------------------------------------------------

# Simulação de Betas Iniciais Poisson

# ------------------------------------------------------------------------------------------------------------

# Declara array de beta 0 -10 ate 10
betas0 = np.arange(1,11, 0.1)
betas0.shape

# Declara betas 1 de -10 ate 10
betas1 = np.arange(-5, 5, 0.1)
betas1.shape

# Declara array 1000, 2000, 10 dados para simulação
it_poisson = np.zeros((100, 100))
it_poisson.shape
it_poisson[:, 0].shape
it_poisson[0, :].shape

# ------------------------------------------------------------------------------------------------------------

# Calcula número de iterções EMV IWLS Poisson

# ------------------------------------------------------------------------------------------------------------

# Laço para popular betas e número de iterações de 100 vezes para cada combinação de betas
for i in range(it_poisson.shape[0]):
  for j in range(it_poisson.shape[1]):
      
    # Declara betas reais
    betas_reais = np.array([4,-0.5]) 
    
    # Declara X
    X = np.random.normal(0, 0.5, 50)
    
    # Declara matriz de Xs
    Xs =  np.stack([np.transpose(np.repeat(1, X.shape)), np.transpose(X)], axis = 1)
    
    # Calcula eta0
    eta = np.matmul(Xs, betas_reais)

    # Declara vetor Y
    Y = calcula_theta(eta, "poisson")
    
    # Declara beta0
    beta0 = np.array([betas0[i],betas1[j]])
    
    # Calcula solução
    sol = EMV(Y, Xs, beta0, 10e-5, 1, "poisson")
  
    # Popula numero de interações gastas
    it_poisson[i, j] = sol.shape[0]

# ------------------------------------------------------------------------------------------------------------

# Grafico de Betas Poisson

# ------------------------------------------------------------------------------------------------------------

# Encerra matplotlib
plt.close()

# Indica tamanho da figura 
sns.set(rc={'figure.figsize':(8, 8)}, font_scale=1.3) 

# Define heatmap
ax = sns.heatmap(it_poisson.transpose(),
                 cmap='coolwarm',
                 cbar_kws={"orientation": "horizontal",
                           "shrink": 0.70, 
                           "aspect": 50,
                           "label": "Iterações"})

# Define ticks e cofigurações dos eixos
ax.set_xticks(np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
ax.set_xticklabels(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
ax.set_yticks(np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
ax.set_yticklabels(np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]))
ax.invert_yaxis()

# Altera rótulos dos eixos
ax.set(xlabel= 'Beta 0 Inicial', ylabel= 'Beta 1 Inicial')
    
# Define titulo do gráfico       
ax.set_suptitle("Numéro de Iterações de Beta 0 Inicial x Beta 1 Inicial")
             
# Subtítulo
ax.set_title("Beta 0 Real = 4, Beta 1 Real = -0.5")
             
# Adiciona grids ao grafico e define limites
plt.grid()

# Desenha gráfico
plt.show()

# ------------------------------------------------------------------------------------------------------------

# Deviance de Observações e Tolerância Poisson

# ------------------------------------------------------------------------------------------------------------

# Declara array numero de observações entre 30 e 130
obs = np.arange(20,120, 1)
obs.shape

# Declara tolerâncias de 10**(-1) até 10**(-11) 
tol = np.arange(0.01, 1.01, 0.01)
tol.shape

# Declara array 1000, 2000, 10 dados para simulação
dev_poisson = np.zeros((100, 100))
dev_poisson.shape

# Declara funcao deviance poisson
def dev_pois(y, mu):
  
  # Caslcula deviance
  deviance = 2 * np.sum(y * np.log(y / mu) - (y - mu))
  
  # Retorno da função
  return deviance


# ------------------------------------------------------------------------------------------------------------

# Calcula Deviances Poisson

# ------------------------------------------------------------------------------------------------------------

# Laço para popular deviances com simulações de observações e tolerâncias
for i in range(dev_poisson.shape[0]):
  for j in range(dev_poisson.shape[1]):
      
    # Declara betas reais
    betas_reais = np.array([4,-0.5]) 
    
    # Declara X
    X = np.random.normal(0, 0.5, obs[i])
    
    # Declara matriz de Xs
    Xs =  np.stack([np.transpose(np.repeat(1, X.shape)), np.transpose(X)], axis = 1)
    
    # Calcula eta0
    eta = np.matmul(Xs, betas_reais)

    # Declara vetor Y
    Y = calcula_theta(eta, "poisson")
    
    # Declara beta0
    beta0 = np.array([1,1])
    
    # Teste de erro
    try:
      
      # Calcula solução
        sol = EMV(Y, Xs, beta0, tol[j], 10, "poisson")
        
    # Lança excessão    
    except np.linalg.LinAlgError as err:
      
      # Condição de verificação
      if 'Singular matrix' in str(err):
        
        # Atribui na
        dev_poisson[i, j] = np.nan
        
    # Recupera betas estimados
    betas_estimados = np.array([sol.iloc[-1, 1], sol.iloc[-1, 2]])
    
    # Calcula eta0
    eta_estimado = np.matmul(Xs, betas_estimados)

    # Declara vetor Y
    mu = np.exp(eta_estimado)
  
    # Popula numero de interações gastas
    dev_poisson[i, j] = dev_pois(Y, mu)

# ------------------------------------------------------------------------------------------------------------

# Grafico Deviances Poisson

# ------------------------------------------------------------------------------------------------------------

# Encerra matplotlib
plt.close()

# Indica tamanho da figura 
sns.set(rc={'figure.figsize':(8, 8)}, font_scale=1.3) 

# Define heatmap
ax = sns.heatmap(np.log10(dev_poisson.transpose()),
                 cmap='coolwarm',
                 cbar_kws={"orientation": "horizontal",
                           "shrink": 0.70, 
                           "aspect": 50,
                           "label": "Log 10 Deviance"})

# Define ticks e cofigurações dos eixos
ax.set_xticks(np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
ax.set_xticklabels(np.array([20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]))
ax.set_yticks(np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
ax.set_yticklabels(np.array([0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91, 1.01, 1.11]))
ax.invert_yaxis()

# Altera rótulos dos eixos
ax.set(xlabel= 'Número de Observações', ylabel= 'Tolerância')
    
# Define titulo do gráfico       
ax.set_title("Deviance de Tolerância x Número de Observações")
             
# Desenha gráfico
plt.show()

# ------------------------------------------------------------------------------------------------------------

# Iterações de Observações e Tolerância Poisson

# ------------------------------------------------------------------------------------------------------------

# Declara array 1000, 2000, 10 dados para simulação
it_poisson2 = np.zeros((100, 100))
it_poisson2.shape

# ------------------------------------------------------------------------------------------------------------

# Calcula Iterações Poisson De Observações x Tolerância

# ------------------------------------------------------------------------------------------------------------

# Laço para popular deviances com simulações de observações e tolerâncias
for i in range(it_poisson2.shape[0]):
  for j in range(it_poisson2.shape[1]):
      
    # Declara betas reais
    betas_reais = np.array([4,-0.5]) 
    
    # Declara X
    X = np.random.normal(0, 0.5, obs[i])
    
    # Declara matriz de Xs
    Xs =  np.stack([np.transpose(np.repeat(1, X.shape)), np.transpose(X)], axis = 1)
    
    # Calcula eta0
    eta = np.matmul(Xs, betas_reais)

    # Declara vetor Y
    Y = calcula_theta(eta, "poisson")
    
    # Declara beta0
    beta0 = np.array([1,1])
    
    # Teste de erro
    try:
      
      # Calcula solução
        sol = EMV(Y, Xs, beta0, tol[j], 10, "poisson")
        
    # Lança excessão    
    except np.linalg.LinAlgError as err:
      
      # Condição de verificação
      if 'Singular matrix' in str(err):
        
        # Atribui na
        it_poisson2[i, j] = np.nan
        
    # Recupera betas estimados
    betas_estimados = np.array([sol.iloc[-1, 1], sol.iloc[-1, 2]])
    
    # Calcula eta0
    eta_estimado = np.matmul(Xs, betas_estimados)

    # Declara vetor Y
    mu = np.exp(eta_estimado)
  
    # Popula numero de interações gastas
    it_poisson2[i, j] = sol.shape[0]

# ------------------------------------------------------------------------------------------------------------

# Grafico simulações de iterações Poisson com Observações x Tolerâncias

# ------------------------------------------------------------------------------------------------------------

# Encerra matplotlib
plt.close()

# Indica tamanho da figura 
sns.set(rc={'figure.figsize':(8, 8)}, font_scale=1.3) 

# Define heatmap
ax = sns.heatmap(it_poisson2.transpose(),
                 cmap='coolwarm',
                 cbar_kws={"orientation": "horizontal",
                           "shrink": 0.70, 
                           "aspect": 50,
                           "label": "Iterações"})

# Define ticks e cofigurações dos eixos
ax.set_xticks(np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
ax.set_xticklabels(np.array([20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]))
ax.set_yticks(np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]))
ax.set_yticklabels(np.array([0.1, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91, 1.01, 1.1]))
ax.invert_yaxis()

# Altera rótulos dos eixos
ax.set(xlabel= 'Amostra', ylabel= 'Tolerância')
    
# Define titulo do gráfico       
ax.set_title("Iterações de Tolerância x Número de Observações")

# Limites do grafico
ax.set_ylim((0,99))

# Desenha gráfico
plt.show()

# ------------------------------------------------------------------------------------------------------------




















# ------------------------------------------------------------------------------------------------------------

# Simulação de Betas Iniciais Binomial

# ------------------------------------------------------------------------------------------------------------

# Declara array de beta 0 -10 ate 10
betas0 = np.arange(1,11, 0.1)
betas0.shape

# Declara betas 1 de -10 ate 10
betas1 = np.arange(-5, 5, 0.1)
betas1.shape

# Declara array 1000, 2000, 10 dados para simulação
it_binomial = np.zeros((100, 100))
it_binomial.shape
it_binomial[:, 0].shape
it_binomial[0, :].shape

# ------------------------------------------------------------------------------------------------------------

# Calcula número de iterções EMV IWLS Binomial

# ------------------------------------------------------------------------------------------------------------

# Laço para popular betas e número de iterações de 100 vezes para cada combinação de betas
for i in range(it_binomial.shape[0]):
  for j in range(it_binomial.shape[1]):
      
    # Declara betas reais
    betas_reais = np.array([4,-0.5]) 
    
    # Declara X
    X = np.random.normal(0, 0.5, 50)
    
    # Declara matriz de Xs
    Xs =  np.stack([np.transpose(np.repeat(1, X.shape)), np.transpose(X)], axis = 1)
    
    # Calcula eta0
    eta = np.matmul(Xs, betas_reais)

    # Declara vetor Y
    Y = calcula_theta(eta, "binomial")
    
    # Declara beta0
    beta0 = np.array([betas0[i],betas1[j]])
    
    # Calcula solução
    sol = EMV(Y, Xs, beta0, 10e-5, 1, "binomial")
  
    # Popula numero de interações gastas
    it_binomial[i, j] = sol.shape[0]

# ------------------------------------------------------------------------------------------------------------

# Grafico de Betas Binomial

# ------------------------------------------------------------------------------------------------------------

# Encerra matplotlib
plt.close()

# Indica tamanho da figura 
sns.set(rc={'figure.figsize':(8, 8)}, font_scale=1.3) 

# Define heatmap
ax = sns.heatmap(it_binomial.transpose(),
                 cmap='coolwarm',
                 cbar_kws={"orientation": "horizontal",
                           "shrink": 0.70, 
                           "aspect": 50,
                           "label": "Iterações"})

# Define ticks e cofigurações dos eixos
ax.set_xticks(np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
ax.set_xticklabels(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
ax.set_yticks(np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
ax.set_yticklabels(np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]))
ax.invert_yaxis()

# Altera rótulos dos eixos
ax.set(xlabel= 'Beta 0 Inicial', ylabel= 'Beta 1 Inicial')
    
# Define titulo do gráfico       
ax.set_suptitle("Numéro de Iterações de Beta 0 Inicial x Beta 1 Inicial")
             
# Subtítulo
ax.set_title("Beta 0 Real = 4, Beta 1 Real = -0.5")
             
# Adiciona grids ao grafico e define limites
plt.grid()

# Desenha gráfico
plt.show()

# ------------------------------------------------------------------------------------------------------------

# Simulação de Observações e Tolerância Binomial

# ------------------------------------------------------------------------------------------------------------

# Declara array numero de observações entre 30 e 130
obs = np.arange(20, 120, 1)
obs.shape

# Declara tolerâncias de 10**(-1) até 10**(-11) 
tol = np.arange(0.11, 1.11, 0.01)
tol.shape

# Declara array 1000, 2000, 10 dados para simulação
dev_binomial = np.zeros((100, 100))
dev_binomial.shape

# Declara funcao deviance poisson
def dev_binom(m, y, mu):
  
  # Caslcula deviance
  deviance = 2 * np.sum(y * np.log(y / mu) + (m - y) * np.log((m - y) / (m - mu)))
  
  # Retorno da função
  return deviance

# ------------------------------------------------------------------------------------------------------------

# Calcula Deviances Poisson

# ------------------------------------------------------------------------------------------------------------

# Laço para popular deviances com simulações de observações e tolerâncias
for i in range(dev_binomial.shape[0]):
  for j in range(dev_binomial.shape[1]):
      
    # Declara betas reais
    betas_reais = np.array([4,-0.5]) 
    
    # Declara X
    X = np.random.normal(0, 0.5, obs[i])
    
    # Declara matriz de Xs
    Xs =  np.stack([np.transpose(np.repeat(1, X.shape)), np.transpose(X)], axis = 1)

    # Declara vetor Y
    Y = np.random.binomial(20, 0.5, obs[i])
    
    # Declara beta0
    beta0 = np.array([1,1])
    
    # Teste de erro
    try:
      
      # Calcula solução
        sol = EMV(Y, Xs, beta0, tol[j], 10, "poisson")
        
    # Lança excessão    
    except np.linalg.LinAlgError as err:
      
      # Condição de verificação
      if 'Singular matrix' in str(err):
        
        # Atribui na
        dev_binomial[i, j] = np.nan
        
    # Recupera betas estimados
    betas_estimados = np.array([sol.iloc[-1, 1], sol.iloc[-1, 2]])
    
    # Calcula eta0
    eta_estimado = np.matmul(Xs, betas_estimados)

    # Declara vetor Y
    mu = calcula_theta(eta_estimado, "binomial")
  
    # Popula numero de interações gastas
    dev_binomial[i, j] = dev_binom(20, Y, mu)

# ------------------------------------------------------------------------------------------------------------

# Grafico simulações de Deviances Poisson

# ------------------------------------------------------------------------------------------------------------

# Encerra matplotlib
plt.close()

# Indica tamanho da figura 
sns.set(rc={'figure.figsize':(8, 8)}, font_scale=1.3) 

# Define heatmap
ax = sns.heatmap(np.log10(dev_binomial.transpose()),
                 cmap='coolwarm',
                 cbar_kws={"orientation": "horizontal",
                           "shrink": 0.70, 
                           "aspect": 50,
                           "label": "Log 10 Deviance"})

# Define ticks e cofigurações dos eixos
ax.set_xticks(np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
ax.set_xticklabels(np.array([20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]))
ax.set_yticks(np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
ax.set_yticklabels(np.array([0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91, 1.01, 1.11]))
ax.invert_yaxis()

# Altera rótulos dos eixos
ax.set(xlabel= 'Número de Observações', ylabel= 'Tolerância')
    
# Define titulo do gráfico       
ax.set_title("Deviance de Tolerância x Número de Observações")
             
# Desenha gráfico
plt.show()

# ------------------------------------------------------------------------------------------------------------

# Iterações de Observações e Tolerância Poisson

# ------------------------------------------------------------------------------------------------------------

# Declara array 1000, 2000, 10 dados para simulação
it_binom2 = np.zeros((100, 100))
it_binom2.shape

# ------------------------------------------------------------------------------------------------------------

# Calcula Iterações Poisson De Observações x Tolerância

# ------------------------------------------------------------------------------------------------------------

# Laço para popular deviances com simulações de observações e tolerâncias
for i in range(it_binom2.shape[0]):
  for j in range(it_binom2.shape[1]):
      
    # Declara betas reais
    betas_reais = np.array([4,-0.5]) 
    
    # Declara X
    X = np.random.normal(0, 0.5, obs[i])
    
    # Declara matriz de Xs
    Xs =  np.stack([np.transpose(np.repeat(1, X.shape)), np.transpose(X)], axis = 1)
    
    # Calcula eta0
    eta = np.matmul(Xs, betas_reais)

    # Declara vetor Y
    Y = calcula_theta(eta, "poisson")
    
    # Declara beta0
    beta0 = np.array([1,1])
    
    # Teste de erro
    try:
      
      # Calcula solução
        sol = EMV(Y, Xs, beta0, tol[j], 10, "poisson")
        
    # Lança excessão    
    except np.linalg.LinAlgError as err:
      
      # Condição de verificação
      if 'Singular matrix' in str(err):
        
        # Atribui na
        it_binom2[i, j] = np.nan
        
    # Recupera betas estimados
    betas_estimados = np.array([sol.iloc[-1, 1], sol.iloc[-1, 2]])
    
    # Calcula eta0
    eta_estimado = np.matmul(Xs, betas_estimados)

    # Declara vetor Y
    mu = np.exp(eta_estimado)
  
    # Popula numero de interações gastas
    it_binom2[i, j] = sol.shape[0]

# ------------------------------------------------------------------------------------------------------------

# Grafico simulações de iterações Poisson com Observações x Tolerâncias

# ------------------------------------------------------------------------------------------------------------

# Encerra matplotlib
plt.close()

# Indica tamanho da figura 
sns.set(rc={'figure.figsize':(8, 8)}, font_scale=1.3) 

# Define heatmap
ax = sns.heatmap(it_binom2.transpose(),
                 cmap='coolwarm',
                 cbar_kws={"orientation": "horizontal",
                           "shrink": 0.70, 
                           "aspect": 50,
                           "label": "Log 10 Deviance"})

# Define ticks e cofigurações dos eixos
ax.set_xticks(np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
ax.set_xticklabels(np.array([20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]))
ax.set_yticks(np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
ax.set_yticklabels(np.array([0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91, 1.01, 1.11]))
ax.invert_yaxis()

# Altera rótulos dos eixos
ax.set(xlabel= 'Número de Observações', ylabel= 'Tolerância')
    
# Define titulo do gráfico       
ax.set_title("Deviance de Tolerância x Número de Observações")
             

# Desenha gráfico
plt.show()

# ------------------------------------------------------------------------------------------------------------


