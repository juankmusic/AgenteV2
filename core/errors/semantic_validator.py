# core/errors/semantic_validator.py

"""
Validador Semántico de SQL
Previene la ejecución de queries lógicamente inválidas ANTES de enviarlas a la BD.
Ahora usa el mismo schema_loader que el resto de la aplicación.
"""

import re
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class ValidationResult:
    """Estructura simple para guardar el resultado de validar una consulta SQL."""
    is_valid: bool                      # Dice si la consulta es válida o no
    error_message: Optional[str] = None # Mensaje explicando el error encontrado
    suggestion: Optional[str] = None    # Sugerencia de cómo corregirlo
    fixed_query: Optional[str] = None   # Versión corregida de la consulta (si aplica)

class SemanticValidator:
    """
    Esta clase revisa si una consulta SQL tiene sentido.
    No solo si está bien escrita, sino si las operaciones son correctas
    según los tipos de datos de la base de datos.
    """

    # Lista de funciones que solo funcionan con números
    NUMERIC_OPERATIONS = {
        'SUM', 'AVG', 'COUNT', 'MAX', 'MIN', 
        'ROUND', 'CEIL', 'FLOOR', 'ABS'
    }
    
    # Lista de nombres de tipos que la BD considera numéricos
    NUMERIC_TYPES = {
        'integer', 'bigint', 'smallint', 'decimal', 
        'numeric', 'real', 'double precision', 'serial',
        'bigserial', 'int', 'int4', 'int8', 'float', 'float4', 'float8',
        'money'
    }
    
    # Lista de tipos que representan texto
    TEXT_TYPES = {
        'character varying', 'varchar', 'character', 'char',
        'text', 'name', 'uuid'
    }
    
    # Lista de tipos relacionados con fechas y tiempo
    DATE_TYPES = {
        'date', 'timestamp', 'timestamp without time zone',
        'timestamp with time zone', 'time', 'interval'
    }
    
    def __init__(self, schema_semantic: Dict[str, Any] = None):
        """
        Crea el validador recibiendo el esquema de la base de datos.
        El esquema dice qué tablas existen y qué tipo tiene cada columna.
        """
        self.schema = schema_semantic or {}      # Guarda el esquema
        self.column_types = self._build_column_type_map()  # Mapa: "tabla.columna" -> "tipo"

    def _build_column_type_map(self) -> Dict[str, str]:
        """
        Crea un mapa para poder encontrar rápidamente el tipo de cualquier columna.
        Esto permite validar cosas como "SUM(nombre)" y saber si 'nombre' es texto o número.
        """
        column_map = {}
        
        # El esquema puede venir con la clave "tables"; si no, se usa como viene
        tables_dict = self.schema.get('tables', self.schema)
        
        # Recorremos cada tabla del esquema
        for table_name, table_info in tables_dict.items():
            
            # Tomamos las columnas de esa tabla
            columns = table_info.get('columns', {})
            
            if not columns:
                continue  # Si la tabla no tiene columnas, la saltamos
            
            for col_name, col_info in columns.items():
                # Tratamos de descubrir el tipo de la columna
                data_type = None
                
                # Si la columna es un diccionario, buscamos la clave donde puede estar el tipo
                if isinstance(col_info, dict):
                    data_type = (col_info.get('type') or 
                                col_info.get('tipo') or 
                                col_info.get('data_type') or
                                'unknown')
                
                # Si la columna solo tiene un string, ese string es el tipo
                elif isinstance(col_info, str):
                    data_type = col_info
                
                # Si logramos obtener un tipo, lo guardamos en el mapa
                if data_type:
                    key = f"{table_name}.{col_name}".lower()
                    column_map[key] = data_type.lower()
        
        # Mensajes informativos para saber si el mapa se construyó bien
        if column_map:
            print(f"✅ Validador: Mapa de tipos construido con {len(column_map)} columnas")
        else:
            print(f"⚠️ Validador: No se pudo construir el mapa de columnas desde el esquema")
            print(f"⚠️ Esquema recibido tiene {len(self.schema)} claves en nivel superior")
        
        return column_map
    
    def validate_query(self, sql_query: str) -> ValidationResult:
        """
        Método principal. Revisa una consulta SQL completa para ver si
        tiene errores semánticos antes de ejecutarla.
        """
        # Si no hay información de columnas, no podemos validar; dejamos pasar la consulta
        if not self.column_types:
            print("⚠️ Validador: Sin mapa de columnas - validación deshabilitada")
            return ValidationResult(is_valid=True)

        # 1. Revisa si se usan funciones numéricas en columnas que no son números
        numeric_validation = self._validate_numeric_operations(sql_query)
        if not numeric_validation.is_valid:
            return numeric_validation
        
        # 2. Revisa si se usan funciones de fecha en columnas que no son fecha
        date_validation = self._validate_date_operations(sql_query)
        if not date_validation.is_valid:
            return date_validation
        
        # 3. Revisa que cada JOIN tenga su condición ON
        join_validation = self._validate_joins(sql_query)
        if not join_validation.is_valid:
            return join_validation
        
        # Si pasó todas las validaciones, entonces está bien
        return ValidationResult(is_valid=True)
    
    def _validate_numeric_operations(self, sql: str) -> ValidationResult:
        """
        Revisa que funciones como SUM, AVG, etc.,
        solo se apliquen a columnas numéricas.
        """
        for operation in self.NUMERIC_OPERATIONS:
            
            # Esta expresión busca cosas como SUM(columna)
            pattern = rf'{operation}\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\s*\)'
            matches = re.finditer(pattern, sql, re.IGNORECASE)
            
            for match in matches:
                column_ref = match.group(1)  # Ej: "edad" o "usuario.edad"
                
                # Intentamos descubrir qué tipo de dato tiene esa columna
                column_type = self._get_column_type(column_ref, sql)
                
                # Si la columna es texto, es un error usar SUM, AVG, etc.
                if column_type and column_type in self.TEXT_TYPES:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"No puedo aplicar la función {operation} sobre '{column_ref}' porque es un campo de texto, no numérico.",
                        suggestion=self._suggest_text_alternative(operation, column_ref)
                    )
                
                # Si es fecha y se usa SUM o AVG, también es un error
                if column_type and column_type in self.DATE_TYPES and operation in ['SUM', 'AVG']:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"No puedo aplicar {operation} sobre '{column_ref}' porque es una fecha.",
                        suggestion=f"Para fechas, podrías usar MAX({column_ref}) o MIN({column_ref}) para encontrar la más reciente o antigua."
                    )
        
        return ValidationResult(is_valid=True)
    
    def _validate_date_operations(self, sql: str) -> ValidationResult:
        """Revisa que funciones de fecha solo se usen en columnas tipo fecha."""
        date_functions = ['EXTRACT', 'DATE_PART', 'TO_CHAR']
        
        for func in date_functions:
            # Busca patrones como EXTRACT(..., fecha_columna)
            pattern = rf'{func}\s*\([^,]+,\s*([a-zA-Z_][a-zA-Z0-9_.]*)\s*\)'
            matches = re.finditer(pattern, sql, re.IGNORECASE)
            
            for match in matches:
                column_ref = match.group(1)
                column_type = self._get_column_type(column_ref, sql)
                
                # Si esa columna NO es de fecha, error
                if column_type and column_type not in self.DATE_TYPES:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"No puedo extraer información de fecha de '{column_ref}' porque no es una columna de tipo fecha/hora.",
                        suggestion=f"'{column_ref}' parece ser de tipo {column_type}. Verifica que sea la columna correcta."
                    )
        
        return ValidationResult(is_valid=True)
    
    def _validate_joins(self, sql: str) -> ValidationResult:
        """Revisa que cada JOIN tenga su condición ON, necesaria para relacionar tablas."""
        
        # Busca todos los JOIN que existan en la consulta
        all_joins = re.findall(
            r'(?:INNER|LEFT|RIGHT)?\s*JOIN\s+[a-zA-Z_][a-zA-Z0-9_]*',
            sql,
            re.IGNORECASE
        )

        # Busca todos los JOIN que sí tengan ON correcto
        valid_joins = re.findall(
            r'(?:INNER|LEFT|RIGHT)?\s*JOIN\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s+(?:AS\s+)?[a-zA-Z_][a-zA-Z0-9_]*)?\s+ON\b',
            sql,
            re.IGNORECASE
        )

        # Si hay JOINs sin ON → error
        if len(valid_joins) < len(all_joins):
            return ValidationResult(
                is_valid=False,
                error_message="Hay un JOIN sin condición ON.",
                suggestion="Revisa que cada JOIN incluya su condición ON correspondiente."
            )

        return ValidationResult(is_valid=True)
        
    def _get_column_type(self, column_ref: str, sql: str) -> Optional[str]:
        """
        Trata de averiguar el tipo de una columna.
        Puede venir como 'tabla.columna' o solo 'columna'.
        """
        column_ref_lower = column_ref.lower()
        
        # Si el formato es "tabla.columna", buscamos en el mapa directamente
        if '.' in column_ref_lower:
            return self.column_types.get(column_ref_lower)
        
        # Si solo es "columna", tratamos de encontrar su tabla revisando el FROM y JOINs
        tables = self._extract_tables_from_query(sql)
        
        for table in tables:
            key = f"{table}.{column_ref_lower}"
            if key in self.column_types:
                return self.column_types[key]
        
        return None
    
    def _extract_tables_from_query(self, sql: str) -> list:
        """Busca todas las tablas mencionadas en una consulta SQL."""
        
        tables = []
        
        # Tablas después de FROM
        from_pattern = r'FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        from_matches = re.finditer(from_pattern, sql, re.IGNORECASE)
        tables.extend([m.group(1).lower() for m in from_matches])
        
        # Tablas después de JOIN
        join_pattern = r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        join_matches = re.finditer(join_pattern, sql, re.IGNORECASE)
        tables.extend([m.group(1).lower() for m in join_matches])
        
        return list(set(tables))  # Quitamos duplicados
    
    def _suggest_text_alternative(self, operation: str, column: str) -> str:
        """
        Da sugerencias amigables cuando alguien intenta usar funciones numéricas sobre texto.
        """
        suggestions = {
            'SUM': f"Si querías contar cuántos registros tienen un valor en '{column}', usa COUNT({column}). Si querías concatenar los valores, eso requiere funciones especiales de texto.",
            'AVG': f"No se puede calcular un promedio de texto. Si '{column}' contiene números como texto, primero necesitarías convertirlo. ¿Querías contar registros? Usa COUNT({column}).",
            'COUNT': f"COUNT funciona con cualquier tipo de dato, pero asegúrate de que sea lo que necesitas.",
            'MAX': f"MAX sobre texto ordenará alfabéticamente. Si esto es lo que quieres, está bien. Si no, verifica que '{column}' sea la columna correcta.",
            'MIN': f"MIN sobre texto ordenará alfabéticamente. Si esto es lo que quieres, está bien. Si no, verifica que '{column}' sea la columna correcta."
        }
        
        # Si no tenemos sugerencia específica, damos una genérica
        return suggestions.get(operation, f"Verifica que '{column}' sea del tipo correcto para {operation}.")
        

# --- Función de conveniencia para usar en chatbot_logic.py ---
def validate_sql_semantics(sql_query: str, schema_semantic: Dict[str, Any] = None) -> ValidationResult:
    """
    Función auxiliar para validar una consulta sin tener que crear el validador a mano.
    """
    # Si no recibimos el esquema, intentamos cargarlo automáticamente
    if schema_semantic is None:
        try:
            from core.ai_core.schema_loader import load_schema
            schema_semantic, _ = load_schema()  # Carga el esquema semántico
        except Exception as e:
            print(f"⚠️ No se pudo cargar el esquema: {e}")
            schema_semantic = {}
    
    # Creamos el validador con el esquema y validamos la query
    validator = SemanticValidator(schema_semantic)
    return validator.validate_query(sql_query)
