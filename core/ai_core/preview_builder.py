# core/ai_core/preview_builder.py
from typing import List, Dict, Any
from markdown import markdown


def build_single_preview(usuario_info: Dict[str,Any],
                         competencias: List[Dict[str,Any]],
                         target_table: str,
                         iluo: int) -> str:
    """Construye la tarjeta HTML para UN usuario."""

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
- **Total de competencias detectadas:** {len(competencias)}
    """

    intro_html = markdown(intro_md)

    table_html = """
        <table class="aigr-table" style="margin-top: 10px; width: 100%;">
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

    for idx, row in enumerate(competencias, start=1):
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

    # Mensaje final para un solo usuario
    footer_html = """
        <br>
        <strong>¿Deseas confirmar esta inserción?</strong><br>
        Responde <strong>"Sí"</strong> para continuar o <strong>"No"</strong> para cancelar.
    """

    return f"""
        <div class="aigr-card" style="padding: 18px; margin-bottom: 25px;">
            {intro_html}
            {table_html}
            {footer_html}
        </div>
    """


def build_multiuser_summary_item(usuario_info, competencias, iluo):
    nombre = (
        usuario_info.get("nombre")
        or usuario_info.get("correo")
        or f"ID {usuario_info.get('id')}"
    )

    total = len(competencias)

    return f"""
    <div class="aigr-card" style="padding: 12px; margin: 12px 0; border: 1px solid #ccc;">
        <strong>👤 {nombre}</strong><br>
        ILUO solicitado: {iluo}<br>
        Total de competencias: {total}<br>
        <em>Pide: "detalles {nombre}" para ver la tabla completa.</em>
    </div>
    """



def build_multiuser_preview(usuarios: List[Dict[str, Any]], target_table: str) -> str:
    """
    Genera la vista previa para múltiples usuarios con encabezado correcto.
    Convierte el encabezado Markdown a HTML, luego agrega las tarjetas HTML de cada usuario.
    """

    # Generar el encabezado en Markdown
    header_md = f"""
**Previsualización de inserción de evaluación**

- Tabla objetivo: `{target_table}`
- Resumen por usuario. Pide "detalles [nombre]" para ver su tabla completa.
    """
    # Convertir el encabezado de Markdown a HTML
    header_html = markdown(header_md)

    # Inicializar el contenedor principal de la vista previa
    html = f'<div class="aigr-container" style="padding: 15px;">{header_html}'

    # Agregar las tarjetas de cada usuario
    for item in usuarios:
        usuario_info = item.get("usuario_info") or item
        competencias = item.get("competencias") or []
        iluo = item.get("iluo") or "N/A"

        # Reutilizamos la función que ya genera HTML por usuario
        html += build_multiuser_summary_item(usuario_info, competencias, iluo)

    # Agregar confirmación final solo una vez
    confirm_md = """
**¿Confirmas las inserciones para TODOS los usuarios?**  
Responde "Sí" o "No".
    """
    html += markdown(confirm_md)

    # Cerrar el contenedor
    html += "</div>"

    return html


def build_competencias_preview(usuario_info_or_list, competencias=None, target_table=None, iluo=None):
    """
    Genera la vista previa de competencias.
    - Si recibe una lista → multiusuario.
    - Si recibe un dict → un solo usuario.
    Compatibilidad hacia atrás:
        build_competencias_preview(usuario_info, competencias, target_table, iluo)
    """
    if isinstance(usuario_info_or_list, list):
        # Lista de usuarios: cada item debe tener 'usuario_info', 'competencias', 'iluo'
        if target_table is None:
            raise ValueError("Se requiere target_table para multiusuario")
        return build_multiuser_preview(usuario_info_or_list, target_table)

    # Caso un solo usuario
    usuario_info = usuario_info_or_list
    if competencias is None or target_table is None or iluo is None:
        raise ValueError("Se requiere usuario_info, competencias, target_table e iluo para un solo usuario")
    
    return build_single_preview(usuario_info, competencias, target_table, iluo)
