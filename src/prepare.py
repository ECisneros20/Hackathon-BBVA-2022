#/usr/bin/env python3


from dvc import api
import numpy as np
import pandas as pd
from io import StringIO
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import requests
from sklearn.preprocessing import StandardScaler
import shutil
from colorama import Fore,  Style
# from coor_val import location



columns = shutil.get_terminal_size().columns

logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        datefmt='%H: %M: %S',
        stream=sys.stderr
        )
logger = logging.getLogger(__name__)

def prepareData(data=None):
    logging.info('Fetching data...')
    if data is None:
        peru_path = api.read('dataset/dataset2.csv', remote='dataset_track')
        peru = pd.read_csv(StringIO(peru_path))
    else:
        peru = data
        peru.rename(columns = {
            'Latitud':'Latitud_(Decimal)',
            'Longitud': 'Longitud_(Decimal)',
            }, inplace = True)
    peru.columns = peru.columns.str.replace(" ", "_")

    # print('Check and delete invalid locations'.center(columns))
    # bad_indexes = location()
    # print("Shape before > ", peru.shape)
    # peru = peru.drop(bad_indexes).reset_index()
    # print("Shape after > ", peru.shape)


    print('Drop columns with the most NAN values and unnecessary columns'.center(columns))

    peru = peru.drop(["Piso", "Elevador", "Posición", "Número_de_frentes"], axis=1)


    peru = peru.drop('Fecha_entrega_del_Informe', axis=1)


    print(Fore.CYAN, end="")
    print(' Changing DTYPES to columns according to the DICTIONARY OF FIELDS'.center(columns))
    peru['Número_de_estacionamiento'] = peru['Número_de_estacionamiento'].astype('float64')



    peru['Área_Terreno'] = peru['Área_Terreno'].replace(',','', regex=True)
    peru['Área_Terreno'] = peru['Área_Terreno'].astype('float64')
    peru['Área_Construcción'] = peru['Área_Construcción'].replace(',','', regex=True)
    peru['Área_Construcción'] = peru['Área_Construcción'].astype('float64')
    peru['Latitud_(Decimal)'] = peru['Latitud_(Decimal)'].replace(',','', regex=True)
    peru['Latitud_(Decimal)'] = peru['Latitud_(Decimal)'].astype('float64')
    peru['Longitud_(Decimal)'] = peru['Longitud_(Decimal)'].replace(',','', regex=True)
    peru['Longitud_(Decimal)'] = peru['Longitud_(Decimal)'].astype('float64')

    peru = peru.apply(lambda x: x.str.lower() if x.dtype == "object" else x)  


    print('Fill NaN values'.center(columns))

    peru['Tipo_de_vía'].fillna(peru['Tipo_de_vía'].mode().iloc[0], inplace=True)
    peru['Número_de_estacionamiento'].fillna(peru['Número_de_estacionamiento'].mode().iloc[0], inplace=True)
    peru['Depósitos'].fillna(peru['Depósitos'].mode().iloc[0], inplace=True)
    peru['Latitud_(Decimal)'].fillna(peru['Latitud_(Decimal)'].mode().iloc[0], inplace=True)
    peru['Longitud_(Decimal)'].fillna(peru['Longitud_(Decimal)'].mode().iloc[0], inplace=True)
    peru['Categoría_del_bien'].fillna(peru['Categoría_del_bien'].mode().iloc[0], inplace=True)
    peru['Edad'].fillna(peru['Edad'].mode().iloc[0], inplace=True)
    peru['Estado_de_conservación'].fillna(peru['Estado_de_conservación'].mode().iloc[0], inplace=True)
    peru['Método_Representado'].fillna(peru['Método_Representado'].mode().iloc[0], inplace=True)
    peru['Área_Terreno'].fillna(peru['Área_Terreno'].mode().iloc[0], inplace=True)
    peru['Área_Construcción'].fillna(peru['Área_Construcción'].mode().iloc[0], inplace=True)

    # if data is None:
    #     peru['Valor_comercial_(USD)'] = peru['Valor_comercial_(USD)'].astype('float64')
    #
    #     print("Visualizing the correlations between numerical variables".center(columns))
    #
    #     plt.figure(figsize=(10, 8))
    #     sns.heatmap(peru.select_dtypes(include=np.number).corr(), cmap="RdBu", annot=True)
    #     plt.title("Correlations between variables", size=15)
    #     plt.show()
    #
    #
    #     print(Fore.CYAN + "Saving prepared data".center(columns))
    #     peru.to_csv('./dataset/prepared_data.csv', index=False)
    #     print(Style.RESET_ALL, end="")
    #


    # if data is not None:
    #     X_val = pd.read_csv('dataset/X_val.csv')
    #     columnas = X_val.columns
    #     print("WTF ", columnas)
    #     df = pd.DataFrame(columns=columnas)
    #     print(peru.columns)
    #     df = pd.concat([peru, df], axis=0)
    #     df = df.fillna(0)
    #     print("LARGOOOO")
    #     difference = [item for item in df.columns if item not in columnas]
    #     print(difference)
    #     print(list(df.columns) == list(columnas))
    #     print(list(df.iloc[0]))
    #     df = df.reindex(sorted(df.columns), axis=1)
    #     # print(df.columns)
    #     # print(len(df.columns))
    #     return [df.iloc[0]]


    print("One hot encoding".center(columns))

    cat_cols = ["Departamento",
            "Provincia",
            "Distrito",
            "Categoría_del_bien",
            "Estado_de_conservación",
            "Método_Representado"]
    peru = pd.get_dummies(peru, columns=cat_cols)



    print("Standardizing the data".center(columns))

    important_num_cols = [
            "Tipo_de_vía",
            "Número_de_estacionamiento",
            "Depósitos",
            "Latitud_(Decimal)",
            "Longitud_(Decimal)",
            "Edad",
            "Área_Terreno",
            "Área_Construcción",
            ]
    print("important_num_cols")
    scaler = StandardScaler()
    peru[important_num_cols] = scaler.fit_transform(peru[important_num_cols])
    

    print("Splitting data X - y".center(columns))
    y = peru['Valor_comercial_(USD)']
    X = peru.drop(['Valor_comercial_(USD)'], axis=1)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
            test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
            test_size=0.25, random_state=1)
    logger.info('Data fetched and prepared')
    train = pd.concat([pd.Series(y_train, index=X_train.index,
        name='price', dtype=int), X_train], axis=1)
    test = pd.concat([pd.Series(y_test, index=X_test.index,
        name='price', dtype=int), X_test], axis=1)
    validation = pd.concat([pd.Series(y_val, index=X_val.index,
        name='price', dtype=int), X_val], axis=1)

    train.to_csv('dataset/train.csv', index=False, header=True)
    validation.to_csv('dataset/validation.csv', index=False, header=True)
    test.to_csv('dataset/test.csv', index=False, header=True)

    logging.info('Data prepared...')

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   prepareData()
