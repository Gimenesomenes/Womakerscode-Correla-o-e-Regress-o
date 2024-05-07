import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pydataset import data 


df = data('Journals')

print(df.head())

# Variáveis categóricas e correlações

# variáveis numéricas
print(df.describe())

# selecionar variáveis numéricas:

df_num = df.select_dtypes(include=['float64', 'int64'])

# calcula matriz de correlação

matriz_corr = df_num.corr()

# Plotar a matriz
#plt.figure(figsize=(10, 8))
#sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
#plt.title('Matriz de Correlação')
#plt.show()

# Gráfico de dispersão entre as variáveis numéricas

#sns.pairplot(df_num, diag_kind='hist')
#plt.show()

# Análise de variáveis categóricas
# Análise de correspondência

# Verifica cardinalidade das editoras
print(df['pub'].nunique())

# Verifica cardinalidade de campos de atuação
print(df['field'].nunique())

print(df['society'].nunique())


## Como vimos que pub e field tem elevada cardinalidade, vamos recategorizar os dados nas top categorias
# Analisa variável editora (pub)

contagens = df['pub'].value_counts()
#print(contagens)

# seleciona as top 5 categorias

top_categorias = contagens.head(5).index.tolist()

#print(top_categorias)

# muda o nome das categorias diferentes das top para outras
df['pub'] = df['pub'].apply(lambda x: x if x in top_categorias else 'Outra')
print(df['pub'].value_counts())

contagens = df['field'].value_counts()
#print(contagens)

# seleciona as top 5 categorias

top_categorias = contagens.head(5).index.tolist()

#print(top_categorias)

# muda o nome das categorias diferentes das top para outras
df['field'] = df['field'].apply(lambda x: x if x in top_categorias else 'Outra')
print(df['field'].value_counts())

# Campo editora

palette = sns.color_palette('viridis', len(df['pub'].unique()))

plt.figure(figsize=(10,8))
sns.countplot(df, x='pub', palette=palette).set_title("Contagem de editoras")
plt.show()

# Campo de atuação

palette = sns.color_palette('viridis', len(df['field'].unique()))

plt.figure(figsize=(10,8))
sns.countplot(df, x='field', palette=palette).set_title("Contagem de Campo")
plt.show()

# Criar tabela de contingencia 

from scipy.stats import chi2_contingency

tab_contingencia = pd.crosstab(df['pub'], df['field'])

print("Tabela de contingência")
print(tab_contingencia)

# Analisa frequencia de linhas
perc_linha = tab_contingencia.div(tab_contingencia.sum(axis=1), axis=0) * 100

print("\nFrequências: ")
print(perc_linha)

# Realiza o teste qui quadrado

chi2, p, dof, expected = chi2_contingency(tab_contingencia)
print(chi2)
print(p)
print(dof)
print(expected)


# Interagindo numéricas e categoricas


#print(df.columns)
#
#
#plt.figure(figsize=(10, 6))
#
#
#sns.boxplot(df, x='citestot', hue='pub', showfliers=False)
#
#plt.title('Preço da publicação por editoras')
#plt.show()


# Modelos de regressão e categóricas

df.drop('title', axis=1, inplace=True)

print(df.dtypes)

# transformar variáveis categoricas em dummies (variáveis binárias)

df_dummies = pd.get_dummies(df).astype(int)

print(df_dummies.head())

# OLS
import statsmodels.api as sm

# definindo a variável explicativa e adiciona a constante

X = sm.add_constant(df_dummies['pages'])

# define a variável de interesse

y = df_dummies['citestot']

# fit modelo

model = sm.OLS(y,X).fit()

# gera tabela de regressão

print(model.summary())

# R-squared baixo: relação não parece completamente linear/explicativa

# Regressão multipla com eliminação recursiva

def OLS_RFE(X, y, threshold=0.05):
    while True:
        model = sm.OLS(y,X).fit()
        p_values = model.pvalues
        max_p_value = p_values.max()
        if max_p_value > threshold:
            remove_feature = p_values.idxmax()
            X = X.drop(remove_feature, axis=1)
        else:
            break
    return model, X.columns

# aplica a função e print da tabela de regressão

X = df_dummies.drop(['citestot', 'society_yes'], axis=1)
y = df_dummies['citestot']

final_model, selected_features = OLS_RFE(X, y)
print(f"Variáveis selecionadas: {selected_features}")
print("Tabela de Regressão: ")
print(final_model.summary())

# salva residuos
residuals = model.resid

# calcula o valor predito

predicted_values = model.fittedvalues

plt.figure(figsize=(8,6))
plt.scatter(predicted_values, residuals, color='blue', alpha=0.5)

# adiciona linha
plt.axhline(y=0, color='red', linestyle='--')

# titulos

plt.title("Resíduo vs Valor Predito")
plt.xlabel("Valor Predito")
plt.ylabel("Resíduo")
plt.show()