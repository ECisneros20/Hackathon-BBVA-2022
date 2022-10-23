from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

def test_null_prediction():
    response = client.post('/v1/prediction', json = {
        "Fecha_entrega_del_Informe": "string",
        "Tipo_de_vía": 0,
        "Piso": "string",
        "Departamento": "string",
        "Provincia": "string",
        "Distrito": "string",
        "Número_de_estacionamiento": 0,
        "Depósitos": 0,
        "Latitud": 0,
        "Longitud": 0,
        "Categoría_del_bien": "string",
        "Posición": "string",
        "Número_de_frentes": 0,
        "Edad": 0,
        "Elevador": 0,
        "Estado_de_conservación": "string",
        "Método_Representado": "string",
        "Área_Terreno": 0,
        "Área_Construcción": 0
        })
    assert response.status_code == 200
    assert response.json()['Valor_comercial'] == 0

def test_random_prediction():
    response = client.post('/v1/prediction', json ={
        "Fecha_entrega_del_Informe": "string",
        "Tipo_de_vía": 0,
        "Piso": "string",
        "Departamento": "string",
        "Provincia": "string",
        "Distrito": "string",
        "Número_de_estacionamiento": 0,
        "Depósitos": 0,
        "Latitud": 0,
        "Longitud": 0,
        "Categoría_del_bien": "string",
        "Posición": "string",
        "Número_de_frentes": 0,
        "Edad": 0,
        "Elevador": 0,
        "Estado_de_conservación": "string",
        "Método_Representado": "string",
        "Área_Terreno": 0,
        "Área_Construcción": 0
        }) 
    assert response.status_code == 200
    assert response.json()['Valor_comercial'] != 0 
