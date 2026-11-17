import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

print("Iniciando la generación de 'Historial_Servipan.xlsx' con el portafolio y cobertura oficial...")

random.seed(2026)
np.random.seed(2026)

# 1. Catálogo oficial de Servipan (4 líneas · 23 referencias · canales de distribución)
catalogo_productos = [
    # --- LÍNEA: PANES ---
    {"Ref": "PAN-01", "Desc item": "Pan Mogolla Blanca", "Sub linea": "Panes", "Canales": ["Tiendas", "Supermercados"], "Precio": 3400, "Base": 95},
    {"Ref": "PAN-02", "Desc item": "Pan Mogolla Integral", "Sub linea": "Panes", "Canales": ["Tiendas", "Supermercados"], "Precio": 3600, "Base": 45},
    {"Ref": "PAN-03", "Desc item": "Pan Mogolla con bocadillo", "Sub linea": "Panes", "Canales": ["Tiendas"], "Precio": 1800, "Base": 85},
    {"Ref": "PAN-04", "Desc item": "Pan Hamburguesa", "Sub linea": "Panes", "Canales": ["Supermercados"], "Precio": 4500, "Base": 60},
    {"Ref": "PAN-05", "Desc item": "Pan Perro", "Sub linea": "Panes", "Canales": ["Tiendas", "Supermercados"], "Precio": 4300, "Base": 55},
    {"Ref": "PAN-06", "Desc item": "Pan de Queso", "Sub linea": "Panes", "Canales": ["Tiendas", "Supermercados"], "Precio": 3800, "Base": 90},

    # --- LÍNEA: TAJADOS ---
    {"Ref": "TAJ-01", "Desc item": "Tajado Multicereal", "Sub linea": "Tajados", "Canales": ["Tiendas", "Supermercados"], "Precio": 5800, "Base": 38},
    {"Ref": "TAJ-02", "Desc item": "Bolsa de Pan Artesanal", "Sub linea": "Tajados", "Canales": ["Tiendas", "Supermercados"], "Precio": 6200, "Base": 42},
    {"Ref": "TAJ-03", "Desc item": "Pan Tajado Blanco XL", "Sub linea": "Tajados", "Canales": ["Institucional"], "Precio": 6500, "Base": 75},
    {"Ref": "TAJ-04", "Desc item": "Pan Tajado Integral", "Sub linea": "Tajados", "Canales": ["Tiendas", "Supermercados"], "Precio": 5400, "Base": 52},
    {"Ref": "TAJ-05", "Desc item": "Pan Tajado de Leche", "Sub linea": "Tajados", "Canales": ["Tiendas", "Supermercados"], "Precio": 5200, "Base": 75},
    {"Ref": "TAJ-06", "Desc item": "Pan Tajado Casero", "Sub linea": "Tajados", "Canales": ["Tiendas", "Supermercados"], "Precio": 4800, "Base": 62},
    {"Ref": "TAJ-07", "Desc item": "Pan Tajado de Mantequilla", "Sub linea": "Tajados", "Canales": ["Tiendas", "Supermercados"], "Precio": 5500, "Base": 58},
    {"Ref": "TAJ-08", "Desc item": "Pan Tajado Blanco", "Sub linea": "Tajados", "Canales": ["Tiendas", "Supermercados"], "Precio": 4400, "Base": 105},

    # --- LÍNEA: ROSQUITAS ---
    {"Ref": "ROS-01", "Desc item": "Rosquitas Integrales de Yuca Grandes", "Sub linea": "Rosquitas", "Canales": ["Supermercados", "Institucional"], "Precio": 6800, "Base": 35},
    {"Ref": "ROS-02", "Desc item": "Rosquitas Quinua y Chia Grandes", "Sub linea": "Rosquitas", "Canales": ["Supermercados", "Institucional"], "Precio": 7200, "Base": 30},
    {"Ref": "ROS-03", "Desc item": "Rosquitas Yuca y Queso Grandes", "Sub linea": "Rosquitas", "Canales": ["Supermercados", "Institucional"], "Precio": 7000, "Base": 40},
    {"Ref": "ROS-04", "Desc item": "Rosquitas integrales de Yuca y queso", "Sub linea": "Rosquitas", "Canales": ["Supermercados", "Institucional"], "Precio": 1500, "Base": 55},
    {"Ref": "ROS-05", "Desc item": "Rosquitas integrales de Yuca", "Sub linea": "Rosquitas", "Canales": ["Supermercados", "Institucional"], "Precio": 1500, "Base": 50},
    {"Ref": "ROS-06", "Desc item": "Rosquitas de Quinua y Chia", "Sub linea": "Rosquitas", "Canales": ["Supermercados", "Institucional"], "Precio": 1600, "Base": 45},

    # --- LÍNEA: GALLETAS ---
    {"Ref": "GAL-01", "Desc item": "Punto Rojo", "Sub linea": "Galletas", "Canales": ["Tiendas"], "Precio": 1400, "Base": 90},
    {"Ref": "GAL-02", "Desc item": "Chelitas de Guayaba", "Sub linea": "Galletas", "Canales": ["Tiendas"], "Precio": 3200, "Base": 70},
    {"Ref": "GAL-03", "Desc item": "Galleta Cuca", "Sub linea": "Galletas", "Canales": ["Tiendas", "Supermercados", "Institucional"], "Precio": 1300, "Base": 115},
]

# 2. Regionales y municipios de distribución en la Costa Caribe
regionales_distribucion = {
    "Cesar - Valledupar": {
        "peso": 0.36,
        "bodega": "Bodega Principal Valledupar",
        "municipios": [
            "Valledupar", "La Paz", "San Diego", "Manaure", "Codazzi", "La Jagua",
            "Chiriguana", "La Loma", "Curumani", "Aguachica", "Bosconia", "Copey",
            "El paso", "Astrea", "Pailitas", "Pelaya", "Urumita", "Villanueva",
        ],
    },
    "Guajira - Riohacha": {
        "peso": 0.15,
        "bodega": "CEDI Riohacha",
        "municipios": [
            "Riohacha", "San Juan", "Maicao", "Fonseca", "Barrancas",
            "Distraccion", "Hatonuevo", "Albania", "La Junta", "La Peña",
        ],
    },
    "Atlántico - Barranquilla": {
        "peso": 0.18,
        "bodega": "CEDI Barranquilla",
        "municipios": [
            "Barranquilla", "Soledad", "Sabanalarga", "Baranoa", "Sabanagrande",
            "Santo Tomas", "Palmar de Varela", "Ponedera", "Repelon", "Luruaco",
        ],
    },
    "Magdalena - Santa Marta": {
        "peso": 0.13,
        "bodega": "CEDI Santa Marta",
        "municipios": [
            "Santa Marta", "Fundación", "Cienaga", "Plato", "El Banco",
            "El Dificil", "Chivolo", "Tenerife", "Guamal", "Santana",
        ],
    },
    "Bolívar - Cartagena": {
        "peso": 0.09,
        "bodega": "CEDI Cartagena",
        "municipios": [
            "Cartagena", "Mompox", "Talaigua Nuevo", "San Sebastian", "Guataca",
        ],
    },
    "Sucre - Sincelejo": {
        "peso": 0.09,
        "bodega": "CEDI Sincelejo",
        "municipios": [
            "Sincelejo", "Corozal", "Sampues", "San Marcos", "Majagual",
            "Chinu", "Ovejas", "Carmen de Bolivar", "Magangue",
        ],
    },
}

# 3. Clientes segmentados por canal oficial (Tiendas, Supermercados, Institucional)
clientes_por_canal = {
    "Tiendas": [
        {"Nombre": "DISTRIBUIDORA TIENDAS DEL CARIBE", "Sucursal": "Ruta T.A.T. Local"},
        {"Nombre": "GRANERO Y ABARROTES LA ECONOMIA", "Sucursal": "Sede Mercado Público"},
        {"Nombre": "AUTOSERVICIO Y TIENDA EL PROGRESO", "Sucursal": "Sucursal Barrio Centro"},
        {"Nombre": "COMERCIALIZADORA PUNTO FRESCO", "Sucursal": "Bodega Mayorista"},
    ],
    "Supermercados": [
        {"Nombre": "SUPERMERCADOS MI FUTURO SAS", "Sucursal": "Sucursal Novalito"},
        {"Nombre": "SUPERTIENDAS OLIMPICA SA", "Sucursal": "Olímpica Guatapurí"},
        {"Nombre": "ALMACENES EXITO SA", "Sucursal": "Éxito Las Flores"},
        {"Nombre": "INVERSIONES EL JARDIN SAS", "Sucursal": "Sucursal Principal"},
    ],
    "Institucional": [
        {"Nombre": "CASINOS Y CATERING DEL CESAR SAS", "Sucursal": "Operación Minera / Institucional"},
        {"Nombre": "COMFAMILIAR CESAR - CANAL INSTITUCIONAL", "Sucursal": "Colegios y Recreación"},
        {"Nombre": "OPERADORA HOTELERA DEL CARIBE", "Sucursal": "Suministro Hoteles y Clínicas"},
    ],
}

# 4. Generación del historial diario de facturación
datos_historial = []
fecha_inicio = datetime(2026, 5, 1)
fecha_fin = datetime(2026, 11, 17)
dias_totales = (fecha_fin - fecha_inicio).days

nombres_regionales = list(regionales_distribucion.keys())
pesos_regionales = [regionales_distribucion[r]["peso"] for r in nombres_regionales]

for i in range(dias_totales + 1):
    fecha_actual = fecha_inicio + timedelta(days=i)
    dow = fecha_actual.weekday()  # 0=Lunes ... 6=Domingo

    # Estacionalidad semanal + efecto quincena (días 14-16 y 29-31)
    factor_dia = {0: 0.92, 1: 0.95, 2: 0.98, 3: 1.02, 4: 1.15, 5: 1.24, 6: 0.85}[dow]
    es_quincena = fecha_actual.day in (14, 15, 16, 29, 30, 31)
    factor_quincena = 1.14 if es_quincena else 1.0

    # Asegurar movimiento diario de todas las referencias para alimentar series de tiempo continuas
    for p in catalogo_productos:
        # 1 a 3 facturas por producto al día según rotación
        n_facturas = random.randint(1, 3)
        for _ in range(n_facturas):
            canal = random.choice(p["Canales"])
            cliente = random.choice(clientes_por_canal[canal])

            regional = random.choices(nombres_regionales, weights=pesos_regionales, k=1)[0]
            info_reg = regionales_distribucion[regional]
            municipio = random.choice(info_reg["municipios"])

            # Cantidad ajustada por canal, día de la semana y quincena
            mult_canal = 1.35 if canal == "Supermercados" else (1.50 if canal == "Institucional" else 0.85)
            ruido = np.random.normal(1.0, 0.12)
            cant = max(8, int(round((p["Base"] / n_facturas) * factor_dia * factor_quincena * mult_canal * ruido)))

            valor_total = cant * p["Precio"]

            datos_historial.append({
                "Fecha": fecha_actual.strftime("%Y-%m-%d"),
                "Zona": municipio,
                "Ruta": f"Ruta {regional} ({canal})",
                "Razon social cliente factura": cliente["Nombre"],
                "Des Sucursal Factura": f"{cliente['Sucursal']} - {municipio}",
                "Sub linea": p["Sub linea"],
                "Ref": p["Ref"],
                "Desc item": p["Desc item"],
                "Cant inv": cant,
                "Valor": valor_total,
                "Bodega": info_reg["bodega"],
                "Motivo": "Venta Directa",
            })

df_historial = pd.DataFrame(datos_historial)
df_historial.to_excel("Historial_Servipan.xlsx", index=False)

print(f"✅ Archivo 'Historial_Servipan.xlsx' generado con éxito ({len(df_historial):,} filas).")
print(f"📦 Productos incluidos: {df_historial['Desc item'].nunique()} referencias en {df_historial['Sub linea'].nunique()} líneas.")
print(f"📍 Cobertura geográfica: {df_historial['Zona'].nunique()} municipios atendidos.")