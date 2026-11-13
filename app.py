"""
Servicio web Flask que sirve el modelo de suscripción a depósitos a plazo.

Endpoints:
    GET  /         -> formulario HTML para probar el modelo en el navegador
    POST /predict  -> API JSON (application/json) o formulario HTML
    GET  /health   -> verificación de estado + metadatos del modelo

Uso:
    python app.py
    # luego abre http://localhost:9696
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, Response, jsonify, redirect, render_template_string, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "bank_pipeline.joblib"

MESES = {
    "jan": "Enero",
    "feb": "Febrero",
    "mar": "Marzo",
    "apr": "Abril",
    "may": "Mayo",
    "jun": "Junio",
    "jul": "Julio",
    "aug": "Agosto",
    "sep": "Septiembre",
    "oct": "Octubre",
    "nov": "Noviembre",
    "dec": "Diciembre",
}

# Metadatos de cada campo: tipo, valor por defecto, rango, opciones y etiquetas.
FIELD_INFO = {
    "age": {
        "label": "Edad (años)",
        "type": "number",
        "default": 40,
        "min": 18,
        "max": 95,
        "step": 1,
        "help": "Edad del cliente",
    },
    "job": {
        "label": "Trabajo",
        "type": "select",
        "default": "management",
        "options": [
            "admin.",
            "blue-collar",
            "entrepreneur",
            "housemaid",
            "management",
            "retired",
            "self-employed",
            "services",
            "student",
            "technician",
            "unemployed",
            "unknown",
        ],
        "labels": {
            "admin.": "Administrativo",
            "blue-collar": "Obrero",
            "entrepreneur": "Emprendedor",
            "housemaid": "Empleo doméstico",
            "management": "Gerencia",
            "retired": "Jubilado",
            "self-employed": "Autónomo",
            "services": "Servicios",
            "student": "Estudiante",
            "technician": "Técnico",
            "unemployed": "Desempleado",
            "unknown": "Desconocido",
        },
        "help": "Tipo de trabajo del cliente",
    },
    "marital": {
        "label": "Estado civil",
        "type": "select",
        "default": "married",
        "options": ["divorced", "married", "single"],
        "labels": {
            "divorced": "Divorciado/a",
            "married": "Casado/a",
            "single": "Soltero/a",
        },
        "help": "Estado civil del cliente",
    },
    "education": {
        "label": "Educación",
        "type": "select",
        "default": "secondary",
        "options": ["primary", "secondary", "tertiary", "unknown"],
        "labels": {
            "primary": "Primaria",
            "secondary": "Secundaria",
            "tertiary": "Universitaria",
            "unknown": "Desconocida",
        },
        "help": "Nivel educativo del cliente",
    },
    "default": {
        "label": "¿Crédito en mora?",
        "type": "select",
        "default": "no",
        "options": ["no", "yes"],
        "labels": {"no": "No", "yes": "Sí"},
        "help": "Si el cliente tiene un crédito en mora",
    },
    "balance": {
        "label": "Saldo medio anual (€)",
        "type": "number",
        "default": 448,
        "min": -8019,
        "max": 102127,
        "step": 1,
        "help": "Saldo medio anual del cliente en euros",
    },
    "housing": {
        "label": "¿Préstamo hipotecario?",
        "type": "select",
        "default": "yes",
        "options": ["no", "yes"],
        "labels": {"no": "No", "yes": "Sí"},
        "help": "Si el cliente tiene un préstamo hipotecario",
    },
    "loan": {
        "label": "¿Préstamo personal?",
        "type": "select",
        "default": "no",
        "options": ["no", "yes"],
        "labels": {"no": "No", "yes": "Sí"},
        "help": "Si el cliente tiene un préstamo personal",
    },
    "contact": {
        "label": "Medio de contacto",
        "type": "select",
        "default": "cellular",
        "options": ["cellular", "telephone", "unknown"],
        "labels": {
            "cellular": "Móvil",
            "telephone": "Teléfono fijo",
            "unknown": "Desconocido",
        },
        "help": "Medio de comunicación usado en la campaña",
    },
    "day": {
        "label": "Día del último contacto",
        "type": "number",
        "default": 16,
        "min": 1,
        "max": 31,
        "step": 1,
        "help": "Día del mes en que se hizo el último contacto",
    },
    "month": {
        "label": "Mes del último contacto",
        "type": "select",
        "default": "may",
        "options": [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ],
        "labels": MESES,
        "help": "Mes en que se hizo el último contacto",
    },
    "campaign": {
        "label": "Contactos en esta campaña",
        "type": "number",
        "default": 2,
        "min": 1,
        "max": 63,
        "step": 1,
        "help": "Número de contactos realizados durante esta campaña",
    },
    "pdays": {
        "label": "Días desde el contacto anterior",
        "type": "number",
        "default": -1,
        "min": -1,
        "max": 871,
        "step": 1,
        "help": "-1 significa que el cliente no fue contactado antes",
    },
    "previous": {
        "label": "Contactos en campañas anteriores",
        "type": "number",
        "default": 0,
        "min": 0,
        "max": 275,
        "step": 1,
        "help": "Número de contactos previos a esta campaña",
    },
    "poutcome": {
        "label": "Resultado de la campaña anterior",
        "type": "select",
        "default": "unknown",
        "options": ["failure", "other", "success", "unknown"],
        "labels": {
            "failure": "Fracaso",
            "other": "Otro",
            "success": "Éxito",
            "unknown": "Desconocido",
        },
        "help": "Resultado de la campaña de marketing anterior",
    },
}

# Agrupación visual del formulario.
SECTIONS = [
    ("Perfil del cliente", ["age", "job", "marital", "education"]),
    ("Situación financiera", ["default", "balance", "housing", "loan"]),
    ("Detalles de la campaña", ["contact", "day", "month", "campaign"]),
    ("Historial de contacto", ["pdays", "previous", "poutcome"]),
]

if MODEL_PATH.exists():
    BUNDLE = joblib.load(MODEL_PATH)
    MODEL = BUNDLE["model"]
    FEATURES = BUNDLE["features"]
    THRESHOLD = BUNDLE.get("threshold", 0.5)
    MODEL_NAME = BUNDLE.get("model_name", "desconocido")
else:  # pragma: no cover - solo si el modelo aún no se entrenó
    BUNDLE = None
    MODEL = None
    FEATURES = list(FIELD_INFO)
    THRESHOLD = 0.5
    MODEL_NAME = "modelo no entrenado"

app = Flask(__name__)

# Logo de la aplicación: una diana con un dardo (priorizar a quién llamar).
FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" aria-label="Predictor de campaña de marketing">
  <defs>
    <linearGradient id="t" x1="18" y1="18" x2="46" y2="46" gradientUnits="userSpaceOnUse">
      <stop stop-color="#34d399"/>
      <stop offset="1" stop-color="#22d3ee"/>
    </linearGradient>
  </defs>
  <rect width="64" height="64" rx="14" fill="#0f172a"/>
  <circle cx="30" cy="34" r="16" fill="none" stroke="url(#t)" stroke-width="3.2"/>
  <circle cx="30" cy="34" r="9.5" fill="none" stroke="url(#t)" stroke-width="3.2" opacity=".7"/>
  <circle cx="30" cy="34" r="3.6" fill="url(#t)"/>
  <path d="M47 17 33.5 30.5" stroke="#fbbf24" stroke-width="3.2" stroke-linecap="round"/>
  <path d="m47 17-1.2 9.2-8-1L47 17Z" fill="#fbbf24"/>
</svg>
"""

PAGE_TEMPLATE = """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Predictor de Suscripción Bancaria</title>
  <link rel="icon" type="image/svg+xml" href="/favicon.svg">
  <meta name="theme-color" content="#0f172a">
  <style>
    :root { --bg:#0f172a; --card:#1e293b; --accent:#34d399; --accent2:#22d3ee;
            --text:#e2e8f0; --muted:#94a3b8; --line:#334155; }
    * { box-sizing: border-box; }
    body { margin:0; min-height:100vh; display:flex; align-items:flex-start; justify-content:center;
           background: radial-gradient(circle at top, #064e3b 0%, var(--bg) 55%);
           font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; color:var(--text);
           padding:32px 20px; }
    .card { width:100%; max-width:820px; background:rgba(30,41,59,.92); border:1px solid var(--line);
            border-radius:18px; padding:30px; box-shadow:0 20px 50px rgba(0,0,0,.45); }
    .brand { display:flex; align-items:center; gap:14px; margin-bottom:10px; }
    .brand img { width:48px; height:48px; border-radius:12px; flex-shrink:0; }
    h1 { margin:0 0 4px; font-size:1.55rem; }
    .sub { margin:0; color:var(--muted); font-size:.95rem; }
    h2 { font-size:.85rem; text-transform:uppercase; letter-spacing:.08em; color:var(--accent);
         margin:26px 0 12px; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(210px, 1fr)); gap:14px; }
    label { display:block; font-size:.82rem; margin-bottom:5px; color:var(--muted); }
    input, select { width:100%; padding:10px 12px; border-radius:9px; border:1px solid #475569;
                    background:#0f172a; color:var(--text); font-size:.97rem; }
    input:focus, select:focus { outline:2px solid var(--accent); border-color:transparent; }
    button { margin-top:26px; width:100%; padding:14px; border:0; border-radius:11px;
             background:linear-gradient(90deg,#10b981,#22d3ee); color:#06202f; font-weight:700;
             font-size:1rem; cursor:pointer; }
    button:hover { filter:brightness(1.08); }
    .result { margin-top:24px; padding:20px; border-radius:13px; border:1px solid var(--line); background:#0b1220; }
    .pill { display:inline-block; padding:5px 12px; border-radius:999px; font-weight:700; font-size:.85rem; }
    .high { background:#064e3b; color:#a7f3d0; }
    .low  { background:#7f1d1d; color:#fecaca; }
    .prob { font-size:2.1rem; font-weight:800; margin:10px 0 4px; }
    .bar { height:10px; border-radius:999px; background:#1e293b; overflow:hidden; margin:12px 0 8px; }
    .bar span { display:block; height:100%; background:linear-gradient(90deg,#22c55e,#eab308,#ef4444); }
    .advice { margin-top:10px; font-size:.92rem; }
    .error { margin-top:18px; padding:12px; border-radius:10px; background:#7f1d1d; color:#fecaca; }
    code { color:var(--accent); }
  </style>
</head>
<body>
  <div class="card">
    <div class="brand">
      <img src="/favicon.svg" alt="Logo del predictor" width="48" height="48">
      <div>
        <h1>Predictor de Suscripción Bancaria</h1>
        <p class="sub">Estima la probabilidad de que un cliente contrate un depósito a plazo tras una campaña telefónica.</p>
      </div>
    </div>
    <form method="post" action="/predict">
      {% for title, names in sections %}
      <h2>{{ title }}</h2>
      <div class="grid">
        {% for name in names %}
        {% set info = fields[name] %}
        <div>
          <label for="{{ name }}" title="{{ info.help }}">{{ info.label }}</label>
          {% if info.type == 'number' %}
          <input type="number" id="{{ name }}" name="{{ name }}" value="{{ values[name] }}"
                 min="{{ info.min }}" max="{{ info.max }}" step="{{ info.step }}" required>
          {% else %}
          <select id="{{ name }}" name="{{ name }}">
            {% for option in info.options %}
            <option value="{{ option }}" {% if values[name] == option %}selected{% endif %}>{{ info.labels[option] }}</option>
            {% endfor %}
          </select>
          {% endif %}
        </div>
        {% endfor %}
      </div>
      {% endfor %}
      <button type="submit">Estimar probabilidad</button>
    </form>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
    {% if result %}
    <div class="result">
      <span class="pill {{ 'high' if result.prediction == 1 else 'low' }}">
        {{ 'Alta probabilidad' if result.prediction == 1 else 'Baja probabilidad' }}
      </span>
      <div class="prob">{{ '%.1f'|format(result.probability * 100) }}%</div>
      <div class="bar"><span style="width: {{ (result.probability * 100)|round(1) }}%"></span></div>
      <div class="advice">
        {% if result.prediction == 1 %}
        <strong>Recomendación:</strong> incluir al cliente en la lista de llamadas prioritarias.
        {% else %}
        <strong>Recomendación:</strong> baja prioridad; conviene postergar la llamada.
        {% endif %}
        Umbral {{ result.threshold }} · Modelo: {{ result.model }}.
      </div>
    </div>
    {% endif %}
  </div>
</body>
</html>
"""


def parse_payload(payload: dict) -> dict:
    """Valida y convierte los valores recibidos según el tipo de cada campo."""
    values = {}
    missing = [
        name
        for name in FEATURES
        if name not in payload or str(payload[name]).strip() == ""
    ]
    if missing:
        raise ValueError(f"Faltan campos: {', '.join(missing)}")

    for name in FEATURES:
        info = FIELD_INFO[name]
        raw = payload[name]
        if info["type"] == "number":
            try:
                value = float(raw)
            except (TypeError, ValueError):
                raise ValueError(
                    f"El campo '{name}' debe ser un número, se recibió {raw!r}"
                )
            if not (info["min"] <= value <= info["max"]):
                raise ValueError(
                    f"El campo '{name}' debe estar entre {info['min']} y "
                    f"{info['max']}, se recibió {value}"
                )
            values[name] = value
        else:
            value = str(raw).strip().lower()
            if value not in info["options"]:
                raise ValueError(
                    f"El campo '{name}' debe ser uno de "
                    f"{', '.join(info['options'])}, se recibió {raw!r}"
                )
            values[name] = value
    return values


def make_prediction(values: dict) -> dict:
    row = pd.DataFrame([values], columns=FEATURES)
    probability = float(MODEL.predict_proba(row)[0, 1])
    prediction = int(probability >= THRESHOLD)
    return {
        "prediction": prediction,
        "label": "sí suscribirá" if prediction == 1 else "no suscribirá",
        "probability": round(probability, 4),
        "threshold": THRESHOLD,
        "model": MODEL_NAME,
    }


def render_page(values: dict, result=None, error=None):
    default_values = {name: info["default"] for name, info in FIELD_INFO.items()}
    merged = {**default_values, **values}
    return render_template_string(
        PAGE_TEMPLATE,
        fields=FIELD_INFO,
        sections=SECTIONS,
        values=merged,
        result=result,
        error=error,
    )


@app.get("/favicon.svg")
def favicon():
    """Sirve el logo usado como icono de la pestaña del navegador."""
    return Response(
        FAVICON_SVG,
        mimetype="image/svg+xml",
        headers={"Cache-Control": "public, max-age=604800"},
    )


@app.get("/favicon.ico")
def favicon_ico():
    """Los navegadores antiguos piden /favicon.ico; se redirigen al SVG."""
    return redirect("/favicon.svg", code=308)


@app.get("/")
def index():
    return render_page({}, result=None, error=None)


@app.post("/predict")
def predict():
    if MODEL is None:
        return (
            jsonify(
                {"error": "No se encontró el modelo. Ejecuta 'python train.py' primero."}
            ),
            503,
        )

    payload = request.get_json(silent=True) if request.is_json else request.form.to_dict()
    if payload is None:
        return jsonify({"error": "Envía un cuerpo JSON o un formulario."}), 400

    try:
        values = parse_payload(payload)
    except ValueError as exc:
        if request.is_json:
            return jsonify({"error": str(exc)}), 400
        return render_page({}, result=None, error=str(exc)), 400

    result = make_prediction(values)

    if request.is_json:
        return jsonify(result)

    return render_page(values, result=result, error=None)


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": MODEL is not None,
            "model": MODEL_NAME,
            "features": FEATURES,
            "trained_at": BUNDLE.get("trained_at") if BUNDLE else None,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696, debug=False)
