import sympy as sp
from math import sin, cos

# ---------------------------------------------------------
# FUNCIONES DISPONIBLES PARA EL USUARIO
# ---------------------------------------------------------

def f1(x):
    return x**3 - x - 1   # Función 1: Tiene una raíz real

def f2(x):
    return sin(x) - 0.5   # Función 2: Comportamiento trigonométrico

# ---------------------------------------------------------
# MÉTODO DE BISECCIÓN
# ---------------------------------------------------------

def biseccion(func, a, b, tol=1e-6, max_iter=100):
    proceso = []
    fa = func(a)
    fb = func(b)

    if fa * fb > 0:
        return f"Error: f({a})={fa} y f({b})={fb}. Deben tener signos opuestos."

    for i in range(max_iter):
        m = (a + b) / 2
        fm = func(m)

        proceso.append(f"Iteración {i+1}: a={a:.6f}, b={b:.6f}, m={m:.6f}, f(m)={fm:.6f}")

        if abs(fm) < tol:
            proceso.append(f"\nConvergencia lograda en x = {m:.6f}")
            break

        if fa * fm < 0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm
    else:
        proceso.append("\nMáximo de iteraciones alcanzado.")

    return "\n".join(proceso)

# ---------------------------------------------------------
# MÉTODO DE NEWTON-RAPHSON
# ---------------------------------------------------------

def newton(func, x0, tol=1e-6, max_iter=50):
    proceso = []

    # Derivada simbólica
    x = sp.symbols('x')
    f_sym = func(x)
    df_sym = sp.diff(f_sym, x)

    f = sp.lambdify(x, f_sym, 'math')
    df = sp.lambdify(x, df_sym, 'math')

    for i in range(max_iter):
        fx = f(x0)
        dfx = df(x0)

        if abs(dfx) < 1e-12: # Usamos un número pequeño para evitar divisiones por cero
            proceso.append("Error: Derivada cercana a cero. Método falla.")
            return "\n".join(proceso)

        x1 = x0 - fx/dfx

        proceso.append(f"Iteración {i+1}: x={x0:.6f}, f(x)={fx:.6f}, f'(x)={dfx:.6f}, x_nuevo={x1:.6f}")

        if abs(x1 - x0) < tol:
            proceso.append(f"\nConvergencia lograda en x = {x1:.6f}")
            break

        x0 = x1
    else:
        proceso.append("\nMáximo de iteraciones alcanzado.")


    return "\n".join(proceso)

# ---------------------------------------------------------
# MÉTODO DE LA SECANTE
# ---------------------------------------------------------

def secante(func, x0, x1, tol=1e-6, max_iter=50):
    proceso = []

    for i in range(max_iter):
        f0 = func(x0)
        f1 = func(x1)

        if abs(f1 - f0) < 1e-12:
            proceso.append("Error: f(x1) - f(x0) cercano a cero. Método falla.")
            return "\n".join(proceso)

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)

        proceso.append(f"Iteración {i+1}: x0={x0:.6f}, x1={x1:.6f}, x_nuevo={x2:.6f}")

        if abs(x2 - x1) < tol:
            proceso.append(f"\nConvergencia lograda en x = {x2:.6f}")
            break

        x0, x1 = x1, x2
    else:
        proceso.append("\nMáximo de iteraciones alcanzado.")


    return "\n".join(proceso)

# ---------------------------------------------------------
# FUNCIÓN PRINCIPAL LLAMADA DESDE FLASK (CORREGIDA)
# ---------------------------------------------------------
# AHORA ACEPTA TRES ARGUMENTOS
def metodo_ecuaciones_no_lineales(metodo, funcion, x0):
    # Selección de función
    funcs = {
        "f1": f1,
        "f2": f2
    }

    f = funcs.get(funcion)
    
    # Manejo de error si la función no se encuentra
    if f is None:
        return "Error: La función seleccionada no está definida en el código."

    # Usamos x0 como valor de referencia o inicial, asumiendo 
    # a=x0 y b=x0+1 para Bisección si es necesario.
    
    if metodo == "biseccion":
        # Usamos un intervalo predefinido para Bisección, ya que solo se envía un x0 desde Flask
        # Podrías modificar Flask para enviar 'a' y 'b'
        a = x0
        b = x0 + 1 # Asumimos un intervalo pequeño, pero esto puede fallar si no hay cambio de signo.
        return biseccion(f, a=a, b=b)

    elif metodo == "newton":
        # Newton requiere derivada → usamos sympy en la función
        f_sym = lambda x: f(x)        
        return newton(lambda x: f_sym(x), x0=x0) # Usamos el x0 que envió el usuario

    elif metodo == "secante":
        # Secante requiere x0 y x1
        x1 = x0 + 0.1 # Asumimos un pequeño desplazamiento para x1
        return secante(f, x0=x0, x1=x1)

    else:
        return "Método no válido"