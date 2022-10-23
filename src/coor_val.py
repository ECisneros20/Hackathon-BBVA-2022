#/usr/bin/env python3


from dvc import api
import pandas as pd
from geopy.geocoders import Nominatim
import logging
import sys
from io import StringIO


geoLoc = Nominatim(user_agent="GetLoc")

def location(dataframe=None):
    if dataframe is None:
        dataframe = pd.read_excel('dataset/dataset.xlsx')
    else:
        dataframe = dataframe
    dataframe = dataframe[['Latitud (Decimal)', 'Longitud (Decimal)']]
    countries = []
    fail_loc = []
    no_peru = []
    
    for i in range(len(dataframe)):
        # print("yo soy latitud: ",dataframe.iloc[i,0])
        # print("yo soy longitud: ",dataframe.iloc[i,1])
        latitud = dataframe.iloc[i,0]
        longitud = dataframe.iloc[i,1]
        try:
            locname = geoLoc.reverse((latitud, longitud))
            print("FINAL {}>>> {}".format(locname[0][-4:], i))
            countries.append(locname)
            if locname[0][-4] == "Perú":
                no_peru.append(i)
        except Exception:
            fail_loc.append(i)
    print("FAIL COORDENADAS: ", fail_loc)
    print("NO PERU COUNTRY: ", no_peru)
    print(len(countries))
    return fail_loc + no_peru
    
