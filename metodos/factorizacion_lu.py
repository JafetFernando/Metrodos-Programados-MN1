import numpy as np
import math

# -----------------------------
# EJEMPLOS (dos por método)
# -----------------------------
def ejemplo_crout_1():
    # Matriz 3x3 (general)
    A = np.array([
        [4.0, 2.0, 1.0],
        [2.0, 5.0, 3.0],
        [1.0, 2.0, 4.0]
    ])
    b = np.array([7.0, 18.0, 20.0])
    return A, b

def ejemplo_crout_2():
    # Matriz 3x3 con ceros estratégicos
    A = np.array([
        [2.0, 1.0, 0.0],
        [0.0, 3.0, 2.0],
        [1.0, 0.0, 4.0]
    ])
    b = np.array([3.0, 10.0, 12.0])
    return A, b

def ejemplo_cholesky_1():
    # Matriz simétrica definida positiva
    A = np.array([
        [6.0, 3.0, 2.0],
        [3.0, 2.0, 1.0],
        [2.0, 1.0, 3.0]
    ])
    b = np.array([9.0, 5.0, 7.0])
    return A, b

def ejemplo_cholesky_2():
    # Otra simétrica PD (4x4 reducible a 3x3 en ejemplo)
    A = np.array([
        [4.0, 2.0, 0.0],
        [2.0, 5.0, 1.0],
        [0.0, 1.0, 4.0]
    ])
    b = np.array([6.0, 8.0, 7.0])
    return A, b

# -----------------------------
# UTIL: Formato numérico
# -----------------------------
def fmt(x):
    try:
        # Usamos f-string para formatear el número
        return f"{x:.6f}"
    except:
        return str(x)

def formatear_sistema_lu(A, b):
    """
    Convierte la matriz A y el vector b en una cadena de texto legible para el sistema Ax=b.
    """
    n = len(b)
    lineas = ["Sistema a resolver (Ax = b):"]
    
    for i in range(n):
        linea = ""
        for j in range(n):
            val = A[i, j]
            if j > 0:
                signo = " +" if val >= 0 else " "
                linea += f"{signo}{val:.2f}x{j+1}"
            else:
                linea += f"{val:.2f}x{j+1}"

        linea += f" = {b[i]:.2f}"
        lineas.append(linea)

    return "\n".join(lineas)

# -----------------------------
# CROUT: descomposición LU (L con diagonales libres, U con diag=1)
# -----------------------------
def crout_decomposition(A):
    n = A.shape[0]
    L = np.zeros((n,n), dtype=float)
    U = np.zeros((n,n), dtype=float)
    # U con diag 1
    for i in range(n):
        U[i,i] = 1.0

    logs = []

    for j in range(n):  # columna j
        # calcular L[i,j] para i=j..n-1
        for i in range(j, n):
            suma = 0.0
            for k in range(j):
                suma += L[i,k] * U[k,j]
            L[i,j] = A[i,j] - suma
            logs.append(f"L[{i+1},{j+1}] = A[{i+1},{j+1}] - Σ(L[{i+1},k]*U[k,{j+1}]) = {fmt(L[i,j])}")

        # calcular U[j,k] para k=j+1..n-1
        if L[j,j] == 0:
            raise ZeroDivisionError("Pivote L[j,j] = 0 en Crout.")
        for k in range(j+1, n):
            suma = 0.0
            for r in range(j):
                suma += L[j,r] * U[r,k]
            U[j,k] = (A[j,k] - suma) / L[j,j]
            logs.append(f"U[{j+1},{k+1}] = (A[{j+1},{k+1}] - Σ(L[{j+1},r]*U[r,{k+1}])) / L[{j+1},{j+1}] = {fmt(U[j,k])}")
        
        # Almacenar el estado de las matrices L y U después de cada columna/fila
        logs.append({
            "paso": f"Iteración {j+1}: Cálculo de L[i,{j+1}] y U[{j+1},k]",
            "L_actual": L.copy().tolist(),
            "U_actual": U.copy().tolist(),
            "type": "matrix_update"
        })

    return L, U, logs

def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n, dtype=float)
    logs = []
    
    for i in range(n):
        suma = 0.0
        for j in range(i):
            suma += L[i,j] * y[j]
        y[i] = (b[i] - suma) / L[i,i]
        logs.append(f"y{i+1} = ({fmt(b[i])} - ({fmt(suma)})) / {fmt(L[i,i])} = {fmt(y[i])}")
    
    return y, logs

def back_substitution(U, x_rhs):
    n = len(x_rhs)
    x = np.zeros(n, dtype=float)
    logs = []
    
    for i in range(n-1, -1, -1):
        suma = 0.0
        for j in range(i+1, n):
            suma += U[i,j] * x[j]
        denom = U[i,i]
        if denom == 0:
            raise ZeroDivisionError("Pivote cero en sustitución regresiva.")
        x[i] = (x_rhs[i] - suma) / denom
        logs.append(f"x{i+1} = ({fmt(x_rhs[i])} - ({fmt(suma)})) / {fmt(U[i,i])} = {fmt(x[i])}")
        
    return x, logs

def crout_solve(A, b):
    A = A.astype(float)
    b = b.astype(float)
    
    L, U, logs_decomp = crout_decomposition(A)

    y, logs_fwd = forward_substitution(L, b)
    x, logs_back = back_substitution(U, y)

    return {
        "metodo": "Crout (U con diagonal unitaria)",
        "L": L.tolist(),
        "U": U.tolist(),
        "proceso_decomp": logs_decomp,
        "proceso_fwd": logs_fwd,
        "proceso_back": logs_back,
        "y_vector": y.tolist(),
        "solucion_final": x.tolist()
    }

# -----------------------------
# CHOLESKY: Descomposición A = L · L^T
# -----------------------------
def cholesky_decomposition(A):
    # Usaremos L tal que A = L · L^T (L triangular inferior)
    n = A.shape[0]
    L = np.zeros((n,n), dtype=float)
    logs = []

    for i in range(n):
        for j in range(i+1):
            suma = 0.0
            if i == j:
                for k in range(j):
                    suma += L[j,k]**2
                val = A[j,j] - suma
                if val <= 0:
                    raise ValueError(f"Matriz no definida positiva: A[{j+1},{j+1}] - suma <= 0. (Valor: {val})")
                L[j,j] = math.sqrt(val)
                logs.append(f"L[{j+1},{j+1}] = sqrt(A[{j+1},{j+1}] - Σ(L[{j+1},k]^2)) = {fmt(L[j,j])}")
            else:
                for k in range(j):
                    suma += L[i,k] * L[j,k]
                L[i,j] = (A[i,j] - suma) / L[j,j]
                logs.append(f"L[{i+1},{j+1}] = (A[{i+1},{j+1}] - Σ(L[{i+1},k]*L[{j+1},k])) / L[{j+1},{j+1}] = {fmt(L[i,j])}")
        
        # Almacenar el estado de la matriz L después de cada fila
        logs.append({
            "paso": f"Iteración {i+1}: Cálculo de los elementos de la fila {i+1} de L",
            "L_actual": L.copy().tolist(),
            "type": "matrix_update"
        })

    return L, logs

def cholesky_solve(A, b):
    A = A.astype(float)
    b = b.astype(float)
    
    L, logs_decomp = cholesky_decomposition(A)
    U = L.T # U es L transpuesta
    
    y, logs_fwd = forward_substitution(L, b)
    x, logs_back = back_substitution(U, y)

    return {
        "metodo": "Cholesky (A = L·Lᵀ)",
        "L": L.tolist(),
        "U": U.tolist(),
        "proceso_decomp": logs_decomp,
        "proceso_fwd": logs_fwd,
        "proceso_back": logs_back,
        "y_vector": y.tolist(),
        "solucion_final": x.tolist()
    }

# -----------------------------
# FUNCIÓN PRINCIPAL LLAMADA DESDE FLASK
# -----------------------------
def metodo_lu(metodo, ejemplo):
    """
    metodo: "crout" o "cholesky"
    ejemplo: "e1" o "e2"
    """
    
    # 1. Selección del sistema
    if metodo == "crout":
        if ejemplo == "e1":
            A, b = ejemplo_crout_1()
            ejemplo_titulo = "Ejemplo 1 (Crout: Matriz General)"
        else:
            A, b = ejemplo_crout_2()
            ejemplo_titulo = "Ejemplo 2 (Crout: Matriz General con ceros)"
        solver_func = crout_solve
    elif metodo == "cholesky":
        if ejemplo == "e1":
            A, b = ejemplo_cholesky_1()
            ejemplo_titulo = "Ejemplo 1 (Cholesky: Matriz Simétrica)"
        else:
            A, b = ejemplo_cholesky_2()
            ejemplo_titulo = "Ejemplo 2 (Cholesky: Matriz Simétrica con ceros)"
        solver_func = cholesky_solve
    else:
        return {"error": "Método de factorización no válido"}
        
    # 2. Formatear sistema original
    sistema_texto = formatear_sistema_lu(A, b)

    # 3. Resolver y obtener resultados estructurados
    try:
        resultado = solver_func(A, b)
        resultado["sistema_texto"] = sistema_texto
        resultado["ejemplo_titulo"] = ejemplo_titulo
        return resultado
    except (ZeroDivisionError, ValueError) as e:
        return {"error": f"Error de cálculo: {str(e)}", "sistema_texto": sistema_texto}
    except Exception as e:
        return {"error": f"Error inesperado: {str(e)}", "sistema_texto": sistema_texto}