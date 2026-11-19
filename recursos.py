"""Recursos visuales de marca."""
import base64
import os

def cargar_imagen_base64(ruta: str):
    if os.path.exists(ruta):
        ext = os.path.splitext(ruta)[1].lower().replace(".", "")
        mime = "jpeg" if ext in ("jpg", "jpeg") else "png"
        with open(ruta, "rb") as f:
            return f"data:image/{mime};base64,{base64.b64encode(f.read()).decode('utf-8')}"
    return None

SVG_PAN_ARTESANAL = '<svg viewBox="0 0 260 170" fill="none" xmlns="http://www.w3.org/2000/svg"><ellipse cx="132" cy="108" rx="108" ry="44" transform="rotate(-24 132 108)" fill="#E6D9C5"/><path d="M35 112C22 92 38 62 84 42C130 22 192 14 218 36C244 58 228 92 182 114C136 136 48 132 35 112Z" fill="#DF7A22" stroke="#6B2E12" stroke-width="5.5" stroke-linejoin="round"/><path d="M46 108C72 122 134 122 178 102C206 89 220 71 222 55C228 73 214 96 180 112C134 133 56 128 46 108Z" fill="#F6B25E"/><path d="M72 54C82 65 89 79 90 93" stroke="#6B2E12" stroke-width="4.5" stroke-linecap="round"/><path d="M106 41C117 53 124 68 125 83" stroke="#6B2E12" stroke-width="4.5" stroke-linecap="round"/><path d="M142 31C153 43 160 58 161 73" stroke="#6B2E12" stroke-width="4.5" stroke-linecap="round"/><path d="M177 27C187 38 193 51 194 64" stroke="#6B2E12" stroke-width="4.5" stroke-linecap="round"/></svg>'

SVG_ESPIGA_TRIGO = '<svg viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M22 98C44 76 68 52 96 24" stroke="#6B2E12" stroke-width="2.2" stroke-linecap="round"/><ellipse cx="48" cy="60" rx="5" ry="11" transform="rotate(-45 48 60)" stroke="#6B2E12" stroke-width="2"/><ellipse cx="60" cy="72" rx="5" ry="11" transform="rotate(45 60 72)" stroke="#6B2E12" stroke-width="2"/><ellipse cx="62" cy="46" rx="5" ry="11" transform="rotate(-45 62 46)" stroke="#6B2E12" stroke-width="2"/><ellipse cx="74" cy="58" rx="5" ry="11" transform="rotate(45 74 58)" stroke="#6B2E12" stroke-width="2"/><ellipse cx="76" cy="32" rx="5" ry="11" transform="rotate(-45 76 32)" stroke="#6B2E12" stroke-width="2"/><ellipse cx="88" cy="44" rx="5" ry="11" transform="rotate(45 88 44)" stroke="#6B2E12" stroke-width="2"/><path d="M96 24L106 14M88 22L94 12M98 32L108 26" stroke="#6B2E12" stroke-width="1.8" stroke-linecap="round"/></svg>'
