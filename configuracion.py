"""Configuracion compartida: el orden de clases forma parte del modelo guardado."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
TAMANO_IMAGEN = 128
SEMILLA = 42
CLASES = ["jeringas", "guantes", "mascarillas", "gasas_vendas", "frascos_ampollas", "medicamentos", "papel_carton"]
NOMBRES = ["Jeringas", "Guantes", "Mascarillas / barbijos", "Gasas y vendas", "Frascos y ampollas", "Medicamentos / blísteres", "Papel y cartón"]
PELIGRO = ["CORTOPUNZANTE", "BIOINFECCIOSO POTENCIAL", "BIOINFECCIOSO POTENCIAL", "BIOINFECCIOSO POTENCIAL", "ENVASE MÉDICO / REVISAR CONTENIDO", "PELIGROSO / FARMACÉUTICO", "COMÚN / NO PELIGROSO"]
COLORES = ["#dc3545"] * 6 + ["#17834a"]
MODELO = RAIZ / "salidas" / "modelo_residuos.keras"
MAPA = RAIZ / "salidas" / "clases.json"
MODELO_DEMO = RAIZ / "salidas" / "ajuste_fotos_propias" / "modelo_candidato.keras"
REFERENCIAS_DEMO = RAIZ / "salidas" / "ajuste_fotos_propias" / "referencias.npz"
