#/usr/bin/env python3


from dvc import api
import numpy as np
import pandas as pd
from io import StringIO
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.preprocessing import StandardScaler
import shutil
from colorama import Fore,  Style



columns = shutil.get_terminal_size().columns

logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        datefmt='%H: %M: %S',
        stream=sys.stderr
        )
logger = logging.getLogger(__name__)

logging.info('Fetching data...')
peru_path = api.read('dataset/dataset2.csv', remote='dataset_track')
peru = pd.read_csv(StringIO(peru_path))

print(Fore.CYAN, end="")
print(' Changing DTYPES to columns according to the DICTIONARY OF FIELDS'.center(columns))
peru['Número de estacionamiento'] = peru['Número de estacionamiento'].astype('float64')
peru['Área Terreno'] = peru['Área Terreno'].replace(',','', regex=True)
peru['Área Terreno'] = peru['Área Terreno'].astype('float64')
peru['Área Construcción'] = peru['Área Construcción'].replace(',','', regex=True)
peru['Área Construcción'] = peru['Área Construcción'].astype('float64')
peru['valor comercial (USD)'] = peru['Valor comercial (USD)'].astype('float64')

peru = peru.rename(columns={peru.columns[19]: 'new'})

peru = peru.apply(lambda x: x.str.lower() if x.dtype == "object" else x)  

print('Drop columns with the most NAN values and unnecessary columns'.center(columns))

peru = peru.drop(["Piso", "Elevador", "Posición", "Número de frentes"], axis=1)

peru = peru.drop(peru.columns[15], axis=1)

peru = peru.drop('Fecha entrega del Informe', axis=1)

print('Fill NaN values'.center(columns))

peru['Tipo de vía'].fillna(peru['Tipo de vía'].mode().iloc[0], inplace=True)
peru['Número de estacionamiento'].fillna(peru['Número de estacionamiento'].mode().iloc[0], inplace=True)
peru['Depósitos'].fillna(peru['Depósitos'].mode().iloc[0], inplace=True)
peru['Latitud (Decimal)'].fillna(peru['Latitud (Decimal)'].mode().iloc[0], inplace=True)
peru['Longitud (Decimal)'].fillna(peru['Longitud (Decimal)'].mode().iloc[0], inplace=True)
peru['Categoría del bien'].fillna(peru['Categoría del bien'].mode().iloc[0], inplace=True)
peru['Edad'].fillna(peru['Edad'].mode().iloc[0], inplace=True)
peru['Estado de conservación'].fillna(peru['Estado de conservación'].mode().iloc[0], inplace=True)
peru['Método Representado'].fillna(peru['Método Representado'].mode().iloc[0], inplace=True)
peru['Área Terreno'].fillna(peru['Área Terreno'].mode().iloc[0], inplace=True)
peru['Área Construcción'].fillna(peru['Área Construcción'].mode().iloc[0], inplace=True)

print("Visualizing the correlations between numerical variables".center(columns))

plt.figure(figsize=(10, 8))
sns.heatmap(peru.select_dtypes(include=np.number).corr(), cmap="RdBu", annot=True)
plt.title("Correlations between variables", size=15)
plt.show()

print(Fore.CYAN + "Saving prepared data".center(columns))
peru.to_csv('./dataset/prepared_data.csv', index=False)
print(Style.RESET_ALL, end="")


logger.info('Data fetched and prepared')
