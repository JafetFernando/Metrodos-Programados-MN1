import numpy as np
import math

# -----------------------------
# UTIL: Formato numérico
# -----------------------------
def fmt(x):
    """Formatea un número flotante a 4 decimales para mejorar la legibilidad."""
    try:
        # ¡IMPORTANTE! Reducir a 4 decimales para que la tabla se vea mejor
        return f"{x:.4f}" 
    except:
        return str(x)

# -----------------------------
# FUNCIONES DE EJEMPLO
# -----------------------------

# Matriz A para los ejemplos
def get_matrix(ejemplo):
    if ejemplo == 'e1':
        # Ejemplo 1
        return np.array([[2.0, 5.0], [1.0, 4.0]], dtype=float)
    elif ejemplo == 'e2':
        # Ejemplo 2 (Para Potencia Inversa, que podría ser más grande)
        return np.array([[4.0, 2.0, 2.0], [2.0, 4.0, 2.0], [2.0, 2.0, 4.0]], dtype=float)
    return None

# Vector inicial x0
def get_x0(ejemplo):
    if ejemplo == 'e1':
        return np.array([1.0, 1.0], dtype=float)
    elif ejemplo == 'e2':
        return np.array([1.0, 1.0, 1.0], dtype=float)
    return None

# -----------------------------
# MÉTODOS DE EIGEN (VALORES Y VECTORES PROPIOS)
# -----------------------------

def potencia(A, x0, tol=1e-4, max_iter=100):
    n = A.shape[0]
    x = x0 / np.linalg.norm(x0, np.inf) # Normalizar x0 usando norma infinito
    
    logs = []
    
    for k in range(1, max_iter + 1):
        y = A @ x
        lambda_aprox = np.linalg.norm(y, np.inf)
        x_new = y / lambda_aprox

        log_data = {
            'iteracion': k,
            'y': [fmt(val) for val in y],
            'lambda_aprox': fmt(lambda_aprox),
            'x_new': [fmt(val) for val in x_new]
        }
        logs.append(log_data)
        
        # Criterio de parada: cambio en el vector x
        if np.linalg.norm(x_new - x, np.inf) < tol:
            break
        
        x = x_new
        
    return {
        'titulo': 'Método de la Potencia para Autovalor Dominante',
        'descripcion': f'Encuentra el autovalor dominante (el de mayor magnitud). Se detuvo en {k} iteraciones.',
        'A': A.tolist(),
        'iteraciones': k,
        'logs': logs,
        'lambda_final': fmt(lambda_aprox),
        'vector_final': [fmt(val) for val in x_new]
    }

def potencia_inversa(A, x0, tol=1e-4, max_iter=100):
    try:
        # Calcular la inversa de A
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return {'error': 'La matriz no es invertible (singular). No se puede aplicar el Método de la Potencia Inversa.', 'A': A.tolist()}
        
    n = A.shape[0]
    x = x0 / np.linalg.norm(x0, np.inf) # Normalizar x0
    
    logs = []
    
    for k in range(1, max_iter + 1):
        y = A_inv @ x
        lambda_inv_aprox = np.linalg.norm(y, np.inf) # Esta es la aproximación de 1/lambda
        x_new = y / lambda_inv_aprox

        # El autovalor real se aproxima como el inverso de lambda_inv_aprox
        lambda_real = 1.0 / lambda_inv_aprox

        log_data = {
            'iteracion': k,
            'y': [fmt(val) for val in y],
            'lambda_inv_aprox': fmt(lambda_inv_aprox),
            'lambda_real': fmt(lambda_real),
            'x_new': [fmt(val) for val in x_new]
        }
        logs.append(log_data)
        
        # Criterio de parada: cambio en el vector x
        if np.linalg.norm(x_new - x, np.inf) < tol:
            break
        
        x = x_new
        
    return {
        'titulo': 'Método de la Potencia Inversa para Autovalor Mínimo',
        'descripcion': f'Encuentra el autovalor de menor magnitud. Se detuvo en {k} iteraciones.',
        'A': A.tolist(),
        'iteraciones': k,
        'logs': logs,
        'lambda_final': fmt(lambda_real),
        'vector_final': [fmt(val) for val in x_new]
    }

# -----------------------------
# CONTROLADOR PRINCIPAL
# -----------------------------

def metodo_eigen(metodo, ejemplo):
    A = get_matrix(ejemplo)
    x0 = get_x0(ejemplo)

    if A is None or x0 is None:
        return {'error': 'Ejemplo no válido.', 'A': []}

    if metodo == 'potencia':
        return potencia(A, x0)
    elif metodo == 'potencia_inversa':
        return potencia_inversa(A, x0)
    else:
        return {'error': 'Método no válido.', 'A': A.tolist()}