# core/ai_core/preview_builder.py
from typing import List, Dict, Any
from markdown import markdown

def build_competencias_preview(usuario_info: Dict[str,Any],
                               competencias_rows: List[Dict[str,Any]],
                               target_table: str,
                               iluo: int) -> str:
    """
    Devuelve HTML estilizado (línea gráfica del sintetizador)
    con la tabla de previsualización para confirmar inserción.
    """

    nombre_usuario = (
        usuario_info.get("nombre") 
        or usuario_info.get("correo") 
        or f"ID {usuario_info.get('id')}"
    )

    intro_md = f"""
**Previsualización de inserción de evaluación**

- **Usuario evaluado:** {nombre_usuario}  
- **Tabla objetivo:** `{target_table}`  
- **Nivel ILUO solicitado:** {iluo}  
- **Total de competencias detectadas:** {len(competencias_rows)}
    """

    intro_html = markdown(intro_md)

    # ===========================
    # TABLA HTML
    # ===========================
    table_html = """
        <table class="aigr-table" style="margin-top: 10px;">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Pregunta</th>
                    <th>Respuesta sugerida</th>
                    <th>ILUO</th>
                </tr>
            </thead>
            <tbody>
    """

    for idx, row in enumerate(competencias_rows, start=1):
        pregunta = (row.get("pregunta") or "").replace("\n", " ")
        respuesta = (row.get("respuesta") or "").replace("\n", " ")
        iluo_val = row.get("id_iluo")

        table_html += f"""
            <tr>
                <td>{idx}</td>
                <td>{pregunta}</td>
                <td>{respuesta}</td>
                <td>{iluo_val}</td>
            </tr>
        """

    table_html += """
            </tbody>
        </table>
    """

    # ===========================
    # FOOTER CON CONFIRMACIÓN
    # ===========================
    footer_html = """
        <div style="margin-top: 15px; font-size: 15px;">
            <strong>¿Deseas confirmar esta inserción?</strong><br>
            Responde <strong>"Sí"</strong> para continuar o <strong>"No"</strong> para cancelar.
        </div>
    """

    # ===========================
    # CONTENEDOR FINAL
    # ===========================
    final_html = f"""
        <div class="aigr-card" style="padding: 18px;">
            {intro_html}
            {table_html}
            {footer_html}
        </div>
    """

    return final_html
