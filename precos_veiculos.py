import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ler dataset carros

df_carros = pd.read_csv('carprices.csv')
print(df_carros)

# printar as descrições

print(df_carros.describe())

# Fazendo uma análise de correlação
num_carros = df_carros.select_dtypes(include=['float64', 'int64'])
print(num_carros)

## Calculando uma matriz de correlação

correlacao_carros = num_carros.corr()

# Plotar a matriz
#plt.figure(figsize=(10, 8))
#sns.heatmap(correlacao_carros, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
#plt.title('Matriz de Correlação Carros')
#plt.show()

# Análise de gráficos de dispersão

# Pega cada coluna numérica e constrói um gráfico de dispersão em relação a preço

#for column in num_carros.columns:
#    plt.figure(figsize=(10,8))
#    sns.scatterplot(data=num_carros, x=column, y='price')
#    plt.title(f'Gráfico de dispersão {column} vs Preço')
#    plt.xlabel(column)
#    plt.ylabel("preço")
#    plt.show()

# A partir disso vamos pegar a dispersão de preço e enginesize e aplicar regressão linear simples

# importando a biblioteca statsmodels.api

import statsmodels.api as sm

# definindo uma variável explicativa e adicionando uma constante

X = sm.add_constant(num_carros['enginesize'])
print(X)

# definindo a variável de interesse, no caso preço

Y = num_carros['price']

# Fit do modelo: aqui o python irá calcular os valores de intercepto e coeficientes angular estimados

model = sm.OLS(Y, X).fit()
print(model.summary())

# Dessa forma a equação do modelo: preço_do_carro(Y) = -8005.44 + 167.69*(enginesize)

# Gerando o gráfico da reta estimada

#plt.figure(figsize=(10,8))
#sns.scatterplot(data=num_carros, x='enginesize', y='price', color='green', alpha=0.5)

# a função regplot gera a regressão e o plot da reta 

#sns.regplot(data=num_carros, x='enginesize', y='price', scatter=False, color='red')
#plt.title("Regressão do Preço do Veículo vs Tamanho")
#plt.xlabel('Tamanho')
#plt.ylabel('Preço')
#plt.show()


# Análise dos resíduos

# salvando dado do resíduo
residuals = model.resid

# calcula o valor predito
predicted_values = model.fittedvalues
#print(predicted_values)


#plt.figure(figsize=(10,8))
#plt.scatter(predicted_values, residuals, color='blue', alpha=0.5)

# adiciona linha

#plt.axhline(y=0, color='red', linestyle='--')

# titulos
#plt.title("Resíduo vs Valor Predito")
#plt.xlabel("Valor Predito")
#plt.ylabel("Resíduo")
#plt.show()


# Regressão Multipla

# Inicialmente selecionamos variáveis com correlação elevada
high_corr_variables = correlacao_carros[(correlacao_carros['price'] > 0.7) | (correlacao_carros['price'] < -0.7)].index.tolist()
high_corr_variables.remove('price')

# variáveis selecionadas

selected_var = ['price'] + high_corr_variables
selected_df = num_carros[selected_var]


# Fit do modelo 

# definindo uma variável explicativa e adicionando uma constante

X = sm.add_constant(selected_df.drop(columns=['price']))


# definindo a variável de interesse, no caso preço

Y = selected_df['price']

# Fit do modelo: aqui o python irá calcular os valores de intercepto e coeficientes angular estimados

model = sm.OLS(Y, X).fit()

print(model.summary())

# Analisando os p-valores, uma das variáveis ficou acima de 0.05, nesse caso podemos remover curbweight

high_corr_variables.remove('curbweight')
#print(high_corr_variables)

#re-selecionar variáveis
selected_var = ['price'] + high_corr_variables
selected_df = num_carros[selected_var]

# definindo uma variável explicativa e adicionando uma constante

X = sm.add_constant(selected_df.drop(columns=['price']))


# definindo a variável de interesse, no caso preço

Y = selected_df['price']

# Fit do modelo: aqui o python irá calcular os valores de intercepto e coeficientes angular estimados

model = sm.OLS(Y, X).fit()

print(model.summary())

# O modelo estimado foi: Preçodocarro(Y) = -6.021e+04 + 848.6984*(carwidth) + 94.9419*(enginesize) + 52.8026*(horsepower)

# Análise dos resíduos

# salvando dado do resíduo
residuals = model.resid

# calcula o valor predito
predicted_values = model.fittedvalues
#print(predicted_values)


plt.figure(figsize=(10,8))
plt.scatter(predicted_values, residuals, color='blue', alpha=0.5)

# adiciona linha

plt.axhline(y=0, color='red', linestyle='--')

# titulos
plt.title("Resíduo vs Valor Predito")
plt.xlabel("Valor Predito")
plt.ylabel("Resíduo")
plt.show()


