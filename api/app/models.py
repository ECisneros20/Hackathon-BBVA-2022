#!/usr/bin/env python3

from typing import Union
from pydantic import BaseModel
from datetime import datetime

class PredictionRequest(BaseModel):
    Fecha_entrega_del_Informe: Union[object, None] = None 
    Tipo_de_vía: float
    Piso: object
    Departamento: object
    Provincia: object
    Distrito: object
    Número_de_estacionamiento: float
    Depósitos: float
    Latitud: float
    Longitud: float
    Categoría_del_bien: object
    Posición: object
    Número_de_frentes: float
    Edad: float
    Elevador: float
    Estado_de_conservación: object
    Método_Representado: object
    Área_Terreno: float
    Área_Construcción: float

class PredictionResponse(BaseModel):
    Valor_comercial: float


if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()
