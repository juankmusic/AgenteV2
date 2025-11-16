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
    """Resultado de la validación semántica"""
    is_valid: bool
    error_message: Optional[str] = None
    suggestion: Optional[str] = None
    fixed_query: Optional[str] = None

class SemanticValidator:
    """
    Valida que las operaciones SQL tengan sentido semántico
    basándose en el esquema de la base de datos.
    """
    
    # Operaciones numéricas que requieren tipos compatibles
    NUMERIC_OPERATIONS = {
        'SUM', 'AVG', 'COUNT', 'MAX', 'MIN', 
        'ROUND', 'CEIL', 'FLOOR', 'ABS'
    }
    
    # Tipos de datos considerados numéricos
    NUMERIC_TYPES = {
        'integer', 'bigint', 'smallint', 'decimal', 
        'numeric', 'real', 'double precision', 'serial',
        'bigserial', 'int', 'int4', 'int8', 'float', 'float4', 'float8',
        'money'
    }
    
    # Tipos de datos considerados texto
    TEXT_TYPES = {
        'character varying', 'varchar', 'character', 'char',
        'text', 'name', 'uuid'
    }
    
    # Tipos de datos considerados fecha/hora
    DATE_TYPES = {
        'date', 'timestamp', 'timestamp without time zone',
        'timestamp with time zone', 'time', 'interval'
    }
    
    def __init__(self, schema_semantic: Dict[str, Any] = None):
        """
        Inicializa el validador con el esquema semántico.
        
        Args:
            schema_semantic: Diccionario con el esquema (output de schema_loader)
        """
        self.schema = schema_semantic or {}
        self.column_types = self._build_column_type_map()
    
    def _build_column_type_map(self) -> Dict[str, str]:
        """
        Construye un mapa de 'tabla.columna' -> 'tipo_de_dato'
        Compatible con el formato de semantic_schema.json que tiene "tables" en el nivel superior
        """
        column_map = {}
        
        # Si el esquema tiene una clave "tables", usarla
        tables_dict = self.schema.get('tables', self.schema)
        
        for table_name, table_info in tables_dict.items():
            # Buscar las columnas
            columns = table_info.get('columns', {})
            
            if not columns:
                continue
            
            for col_name, col_info in columns.items():
                # Extraer el tipo de dato
                data_type = None
                
                if isinstance(col_info, dict):
                    data_type = (col_info.get('type') or 
                                col_info.get('tipo') or 
                                col_info.get('data_type') or
                                'unknown')
                elif isinstance(col_info, str):
                    data_type = col_info
                
                if data_type:
                    key = f"{table_name}.{col_name}".lower()
                    column_map[key] = data_type.lower()
        
        if column_map:
            print(f"✅ Validador: Mapa de tipos construido con {len(column_map)} columnas")
        else:
            print(f"⚠️ Validador: No se pudo construir el mapa de columnas desde el esquema")
            print(f"⚠️ Esquema recibido tiene {len(self.schema)} claves en nivel superior")
        
        return column_map
    
    def validate_query(self, sql_query: str) -> ValidationResult:
        """
        Valida una query SQL antes de ejecutarla.
        
        Args:
            sql_query: La consulta SQL a validar
            
        Returns:
            ValidationResult con el resultado de la validación
        """
        # Si no hay esquema cargado, permitir la ejecución (fallback)
        if not self.column_types:
            print("⚠️ Validador: Sin mapa de columnas - validación deshabilitada")
            return ValidationResult(is_valid=True)
        
        # 1. Validar operaciones numéricas sobre columnas de texto
        numeric_validation = self._validate_numeric_operations(sql_query)
        if not numeric_validation.is_valid:
            return numeric_validation
        
        # 2. Validar operaciones de fecha sobre tipos incorrectos
        date_validation = self._validate_date_operations(sql_query)
        if not date_validation.is_valid:
            return date_validation
        
        # 3. Validar JOINs sin condición ON
        join_validation = self._validate_joins(sql_query)
        if not join_validation.is_valid:
            return join_validation
        
        # Si todo está bien
        return ValidationResult(is_valid=True)
    
    def _validate_numeric_operations(self, sql: str) -> ValidationResult:
        """
        Valida que las funciones numéricas (SUM, AVG, etc.) 
        se apliquen solo sobre columnas numéricas.
        """
        for operation in self.NUMERIC_OPERATIONS:
            # Busca patrones como "SUM(columna)" o "SUM(tabla.columna)"
            pattern = rf'{operation}\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\s*\)'
            matches = re.finditer(pattern, sql, re.IGNORECASE)
            
            for match in matches:
                column_ref = match.group(1)  # Ej: "nombre" o "usuario.nombre"
                
                # Intentar resolver el tipo de la columna
                column_type = self._get_column_type(column_ref, sql)
                
                if column_type and column_type in self.TEXT_TYPES:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"No puedo aplicar la función {operation} sobre '{column_ref}' porque es un campo de texto, no numérico.",
                        suggestion=self._suggest_text_alternative(operation, column_ref)
                    )
                
                # Validar que no sea fecha si es SUM o AVG
                if column_type and column_type in self.DATE_TYPES and operation in ['SUM', 'AVG']:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"No puedo aplicar {operation} sobre '{column_ref}' porque es una fecha.",
                        suggestion=f"Para fechas, podrías usar MAX({column_ref}) o MIN({column_ref}) para encontrar la más reciente o antigua."
                    )
        
        return ValidationResult(is_valid=True)
    
    def _validate_date_operations(self, sql: str) -> ValidationResult:
        """Valida operaciones con fechas"""
        date_functions = ['EXTRACT', 'DATE_PART', 'TO_CHAR']
        
        for func in date_functions:
            pattern = rf'{func}\s*\([^,]+,\s*([a-zA-Z_][a-zA-Z0-9_.]*)\s*\)'
            matches = re.finditer(pattern, sql, re.IGNORECASE)
            
            for match in matches:
                column_ref = match.group(1)
                column_type = self._get_column_type(column_ref, sql)
                
                if column_type and column_type not in self.DATE_TYPES:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"No puedo extraer información de fecha de '{column_ref}' porque no es una columna de tipo fecha/hora.",
                        suggestion=f"'{column_ref}' parece ser de tipo {column_type}. Verifica que sea la columna correcta."
                    )
        
        return ValidationResult(is_valid=True)
    
    def _validate_joins(self, sql: str) -> ValidationResult:
        """Valida que los JOINs tengan condición ON"""
        all_joins = re.findall(r'(?:INNER|LEFT|RIGHT)?\s*JOIN\s+[a-zA-Z_][a-zA-Z0-9_]*', sql, re.IGNORECASE)

        # Buscar JOINs correctos
        valid_joins = re.findall(
            r'(?:INNER|LEFT|RIGHT)?\s*JOIN\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s+(?:AS\s+)?[a-zA-Z_][a-zA-Z0-9_]*)?\s+ON\b',
            sql,
            re.IGNORECASE
        )

        # Si hay JOINs pero no todos están en la lista de válidos, marcar error
        if len(valid_joins) < len(all_joins):
            return ValidationResult(
                is_valid=False,
                error_message="Hay un JOIN sin condición ON.",
                suggestion="Revisa que cada JOIN incluya su condición ON correspondiente."
            )

        return ValidationResult(is_valid=True)
        
    def _get_column_type(self, column_ref: str, sql: str) -> Optional[str]:
        """
        Intenta determinar el tipo de una columna basándose en el esquema.
        
        Args:
            column_ref: Referencia a la columna (ej: "nombre" o "usuario.nombre")
            sql: La query completa (para inferir la tabla si no está especificada)
        """
        column_ref_lower = column_ref.lower()
        
        # Si tiene formato "tabla.columna"
        if '.' in column_ref_lower:
            return self.column_types.get(column_ref_lower)
        
        # Si solo es "columna", intentar inferir la tabla del FROM
        tables = self._extract_tables_from_query(sql)
        
        # Buscar en cada tabla mencionada
        for table in tables:
            key = f"{table}.{column_ref_lower}"
            if key in self.column_types:
                return self.column_types[key]
        
        return None
    
    def _extract_tables_from_query(self, sql: str) -> list:
        """Extrae los nombres de las tablas de una query SQL"""
        tables = []
        
        # Buscar tablas después de FROM
        from_pattern = r'FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        from_matches = re.finditer(from_pattern, sql, re.IGNORECASE)
        tables.extend([m.group(1).lower() for m in from_matches])
        
        # Buscar tablas después de JOIN
        join_pattern = r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        join_matches = re.finditer(join_pattern, sql, re.IGNORECASE)
        tables.extend([m.group(1).lower() for m in join_matches])
        
        return list(set(tables))
    
    def _suggest_text_alternative(self, operation: str, column: str) -> str:
        """Genera una sugerencia cuando se intenta operar sobre texto"""
        suggestions = {
            'SUM': f"Si querías contar cuántos registros tienen un valor en '{column}', usa COUNT({column}). Si querías concatenar los valores, eso requiere funciones especiales de texto.",
            'AVG': f"No se puede calcular un promedio de texto. Si '{column}' contiene números como texto, primero necesitarías convertirlo. ¿Querías contar registros? Usa COUNT({column}).",
            'COUNT': f"COUNT funciona con cualquier tipo de dato, pero asegúrate de que sea lo que necesitas.",
            'MAX': f"MAX sobre texto ordenará alfabéticamente. Si esto es lo que quieres, está bien. Si no, verifica que '{column}' sea la columna correcta.",
            'MIN': f"MIN sobre texto ordenará alfabéticamente. Si esto es lo que quieres, está bien. Si no, verifica que '{column}' sea la columna correcta."
        }
        
        return suggestions.get(operation, f"Verifica que '{column}' sea del tipo correcto para {operation}.")


# --- Función de conveniencia para usar en chatbot_logic.py ---
def validate_sql_semantics(sql_query: str, schema_semantic: Dict[str, Any] = None) -> ValidationResult:
    """
    Función de conveniencia para validar una query.
    
    Args:
        sql_query: La consulta SQL a validar
        schema_semantic: Esquema semántico (opcional, si no se pasa se debe cargar)
    
    Returns:
        ValidationResult
    """
    # Si no se pasó el esquema, intentar cargarlo
    if schema_semantic is None:
        try:
            from core.ai_core.schema_loader import load_schema
            schema_semantic, _ = load_schema()
        except Exception as e:
            print(f"⚠️ No se pudo cargar el esquema: {e}")
            schema_semantic = {}
    
    validator = SemanticValidator(schema_semantic)
    return validator.validate_query(sql_query)