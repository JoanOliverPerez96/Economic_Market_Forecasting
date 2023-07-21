from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def learn():
    return "Flask for web developers!"

@app.route("/<your_name>")
def greetings(your_name):
    """Función de vista para saludar al usuario por su nombre."""
    return "¡Bienvenido/a " + your_name + "!"

@app.route("/user/<name>")
def index(name):
    return render_template("home.html", name=name)

@app.route("/user/<name>/<int:index>")
def index2(name, index):
    mylist = ['elemento 1', 'elemento 2', 'elemento 3', 'elemento 4']
    mydict = {'key': 'estupor'}
    mytuple = ('tuple 1', 'tuple 2', 'tuple 3', 'tuple 4')
    return render_template("test.html", name=name, myindex=index, mylist=mylist, mydict=mydict, mytuple=mytuple)
@app.route('/saludo')
def saludo():
    genero = 'perro'  # Esta variable puede ser definida dinámicamente
    name = 'Carla'  # Esta también puede ser definida dinámicamente
    return render_template('saludo.html', genero=genero, name=name)

@app.route('/nombres')
def nombres():
    nombres = ['Juan', 'Ana', 'Carlos', 'María']  # Esta lista puede ser obtenida de forma dinámica
    return render_template('nombres.html', nombres=nombres)

if __name__ == "__main__":
    app.run(debug=True)