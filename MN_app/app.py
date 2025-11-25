from flask import Flask, render_template, request
# ----------------------------------------------------
# Importaciones de los módulos de métodos numéricos
# Asegúrate de que los archivos .py estén en la carpeta 'metodos/'.
from metodos.ecuaciones_no_lineales import metodo_ecuaciones_no_lineales
from metodos.sistemas_lineales import metodo_sistemas_lineales
from metodos.factorizacion_lu import metodo_lu
from metodos.menu_eigen import metodo_eigen 
# ----------------------------------------------------

app = Flask(__name__)

# ---------------------------
# PAGINAS PRINCIPALES
# ---------------------------

@app.route("/")
def portada():
    return render_template("portada.html")

@app.route("/introduccion")
def introduccion():
    return render_template("introduccion.html")

@app.route("/menu")
def menu():
    return render_template("menu.html")

@app.route("/salir")
def salir():
    return render_template("despedida.html")

# ---------------------------
# PAGINAS DE MÉTODOS (POST & GET)
# ---------------------------

@app.route("/ecuaciones-no-lineales", methods=["GET", "POST"])
def ecuaciones_no_lineales():
    if request.method == "POST":
        funcion_str = request.form["funcion"]
        x0_str = request.form.get("x0", "0") 
        
        try:
            x0 = float(x0_str) 
        except ValueError:
            return render_template("ecuaciones-no-lineales.html", 
                                   resultado="Error: El punto inicial debe ser un número.",
                                   funcion=funcion_str)

        metodo = request.form.get("metodo", "biseccion") 
        resultado = metodo_ecuaciones_no_lineales(metodo, funcion_str, x0) 

        return render_template("ecuaciones-no-lineales.html", resultado=resultado)

    return render_template("ecuaciones-no-lineales.html", resultado=None)


@app.route("/sistemas-lineales", methods=["GET", "POST"])
def sistemas_lineales():
    if request.method == "POST":
        # 1. Obtener datos del formulario
        metodo = request.form["metodo"]
        sistema_key = request.form["sistema"]

        # 2. Llamar a la lógica
        sistema_texto, metodo_usado, datos_resultado = metodo_sistemas_lineales(metodo, sistema_key)

        # 3. Renderizar la plantilla con los resultados
        return render_template("sistemas-lineales.html", 
                               sistema_texto=sistema_texto,
                               metodo_usado=metodo_usado,
                               datos_resultado=datos_resultado)

    # Caso GET inicial
    return render_template("sistemas-lineales.html", 
                           sistema_texto=None, 
                           metodo_usado=None, 
                           datos_resultado=None)

@app.route("/factorizacion-lu", methods=["GET", "POST"])
def factorizacion_lu():
    if request.method == "POST":
        metodo = request.form["metodo"]        # "crout" o "cholesky"
        ejemplo = request.form["ejemplo"]      # "e1" o "e2"
        resultado = metodo_lu(metodo, ejemplo)
        return render_template("factorizacion-lu.html", resultado=resultado) 
    return render_template("factorizacion-lu.html", resultado=None)


# ----------------------------------------------------------------------
# RUTA FINAL Y CORRECTA PARA VALORES Y VECTORES PROPIOS (Opción 4)
# ----------------------------------------------------------------------
@app.route("/valores-propios", methods=["GET", "POST"])
def valores_propios():
    resultado = None
    
    # Maneja la solicitud POST (cálculo al presionar el botón)
    if request.method == "POST":
        try:
            metodo = request.form["metodo"]        # "potencia" o "potencia_inversa"
            ejemplo = request.form["ejemplo"]      # "e1" o "e2"
            
            # Llama a la función de cálculo 
            resultado = metodo_eigen(metodo, ejemplo) 
            
        except Exception as e:
            # Captura errores de cálculo
            resultado = {"error": f"Error al ejecutar el método de valores propios: {str(e)}", "A": []}
        
    # Retorna la plantilla, ya sea en GET (inicio) o POST (resultados)
    return render_template("eigen.html", resultado=resultado)


# ---------------------------
# EJECUTAR SERVIDOR
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)