import numpy as np

# ---------------------------------------------------------
# FUNCIONES DE AYUDA
# ---------------------------------------------------------

def formatear_sistema(A, b):
    """
    Convierte la matriz A y el vector b en una cadena de texto legible.
    """
    n = len(b)
    lineas = ["Sistema a resolver (Ax = b):"]
    
    # Formatear la matriz y el vector de forma matricial
    for i in range(n):
        linea = "["
        for j in range(n):
            # Formatea los coeficientes
            val = A[i, j]
            # Usa signo para términos después del primero (j > 0)
            if j > 0:
                signo = " +" if val >= 0 else " "
                linea += f"{signo}{val:.3f}x{j+1}"
            else:
                linea += f"{val:.3f}x{j+1}"

        linea += f" ] = [ {b[i]:.3f} ]"
        lineas.append(linea)

    return "\n".join(lineas)


# ---------------------------------------------------------
# SISTEMAS DISPONIBLES PARA EL USUARIO
# ---------------------------------------------------------

def sistema1():
    A = np.array([
        [10, 2, 1],
        [2, 10, 1],
        [1, 1, 10]
    ], float)

    b = np.array([14, 14, 14], float)

    return A, b


def sistema2():
    A = np.array([
        [4, -1, 1],
        [1, 6, 2],
        [-1, 2, 5]
    ], float)

    b = np.array([7, 9, 3], float)

    return A, b


# ---------------------------------------------------------
# MÉTODO DE GAUSS CON ELIMINACIÓN (SALIDA MODIFICADA)
# ---------------------------------------------------------

def gauss(A, b):
    A = A.astype(float)
    b = b.astype(float)

    n = len(b)
    # Lista que almacenará diccionarios de cada paso
    proceso_gauss = []

    # Eliminación
    for k in range(n-1):
        for i in range(k+1, n):
            if A[k][k] == 0:
                return {"error": "Error: Pivote cero. El método falla."}

            m = A[i][k] / A[k][k]

            # 1. Guardar la matriz antes de la operación
            # Copiamos la matriz y el vector b como listas para que sean serializables
            proceso_gauss.append({
                "paso": f"Eliminación de x{k+1} en la Fila {i+1}. Multiplicador m({i+1},{k+1})",
                "matriz": A.copy().tolist(), 
                "vector": b.copy().tolist(),
                "multiplicador": f"{m:.4f}",
                # Nuevas variables para la plantilla HTML
                "row_modificada": i + 1,
                "row_pivot": k + 1,
            })
            
            # Operación de eliminación
            A[i] = A[i] - m * A[k]
            b[i] = b[i] - m * b[k]

    # Sustitución regresiva 
    x = np.zeros(n)
    
    sustitucion_regresiva = []

    for i in range(n-1, -1, -1):
        # np.dot calcula la suma: A[i][i+1]*x[i+1] + A[i][i+2]*x[i+2] + ...
        suma = np.dot(A[i][i+1:], x[i+1:])
        x[i] = (b[i] - suma) / A[i][i]
        
        sustitucion_regresiva.append(f"x{i+1} = {x[i]:.6f}")

    # Devolvemos un diccionario con la información estructurada
    return {
        "proceso_gauss": proceso_gauss,
        "solucion_final": x.tolist(),
        "sustitucion_regresiva": sustitucion_regresiva,
        "matriz_triangular": A.tolist(),
        "vector_final_b": b.tolist(),
    }


# ---------------------------------------------------------
# MÉTODO DE JACOBI (SALIDA MODIFICADA)
# ---------------------------------------------------------

def jacobi(A, b, tol=1e-6, max_iter=50):
    A = A.astype(float)
    b = b.astype(float)

    n = len(b)
    x_old = np.zeros(n)
    # Lista que almacenará las iteraciones [it, x1, x2, ..., xn, error]
    proceso_iterativo = []
    
    proceso_iterativo.append([0, *x_old.tolist(), 0.0]) # Iteración inicial

    for it in range(max_iter):
        x_new = np.zeros(n)

        for i in range(n):
            # Calculamos la suma de los términos fuera de la diagonal
            suma = np.dot(A[i, :], x_old) - A[i, i] * x_old[i]
            x_new[i] = (b[i] - suma) / A[i, i]
        
        error = np.linalg.norm(x_new - x_old, np.inf)

        proceso_iterativo.append([it + 1, *x_new.tolist(), error])

        if error < tol:
            break

        x_old = x_new.copy()

    return {
        "proceso_iterativo": proceso_iterativo,
        "solucion_final": x_new.tolist(),
        "variables": [f"x{i+1}" for i in range(n)],
        "iteraciones_usadas": it + 1 if error < tol else max_iter,
        "convergio": error < tol
    }


# ---------------------------------------------------------
# MÉTODO DE GAUSS-SEIDEL (SALIDA MODIFICADA)
# ---------------------------------------------------------

def gauss_seidel(A, b, tol=1e-6, max_iter=50):
    A = A.astype(float)
    b = b.astype(float)

    n = len(b)
    x = np.zeros(n)
    proceso_iterativo = []

    proceso_iterativo.append([0, *x.tolist(), 0.0]) # Iteración inicial
    
    for it in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            # Se usan los valores de 'x' que ya fueron actualizados en la iteración actual
            suma = np.dot(A[i, :], x) - A[i, i] * x[i]
            x[i] = (b[i] - suma) / A[i, i]
            
        error = np.linalg.norm(x - x_old, np.inf)

        proceso_iterativo.append([it + 1, *x.tolist(), error])

        if error < tol:
            break

    return {
        "proceso_iterativo": proceso_iterativo,
        "solucion_final": x.tolist(),
        "variables": [f"x{i+1}" for i in range(n)],
        "iteraciones_usadas": it + 1 if error < tol else max_iter,
        "convergio": error < tol
    }


# ---------------------------------------------------------
# FUNCIÓN PRINCIPAL LLAMADA DESDE FLASK (MODIFICADA)
# ---------------------------------------------------------

def metodo_sistemas_lineales(metodo, sistema):
    
    # Elegir sistema
    if sistema == "s1":
        A, b = sistema1()
    elif sistema == "s2":
        A, b = sistema2()
    else:
        return "Error al cargar sistema.", "gauss", {"error": "Sistema no válido"}

    # Formatear el texto del sistema
    sistema_texto = formatear_sistema(A.copy(), b.copy()) # Usamos copias para el texto
    
    # El resultado ahora es un diccionario de datos estructurados
    if metodo == "gauss":
        datos_resultado = gauss(A, b)

    elif metodo == "jacobi":
        datos_resultado = jacobi(A, b)

    elif metodo == "gauss_seidel":
        datos_resultado = gauss_seidel(A, b)

    else:
        datos_resultado = {"error": "Método no válido"}
        
    # Devolvemos el texto del sistema, el método y los datos estructurados
    return sistema_texto, metodo, datos_resultado