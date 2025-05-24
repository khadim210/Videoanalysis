# code permettant de résoudre des equations du second degre
# Auteur: Ahmath B. MBACKE

import math

import mysql.connector


def equation(a, b, c):
    delta = b**2 - 4*a*c
    if delta < 0:
        return "Pas de solution"
    elif delta == 0:
        return -b/(2*a)
    else:
        return (-b-math.sqrt(delta))/(2*a), (-b+math.sqrt(delta))/(2*a)

# fonction permettant de résoudre une dizaine d'équations du second degre dont les
# parametres sont dans un vecteur
def equations(params):
    solutions = []
    for param in params:
        solutions.append(equation(param[0], param[1], param[2]))
    return solutions
# test de la fonction equations avec un vecteur de parametres

params = [(1, 2, 1), (1, 3, 2), (1, 4, 3), (1, 5, 4), (1, 6, 5), (1, 7, 6), (1, 8, 7), (1, 9, 8), (1, 10, 9), (1, 11, 10)]
print(equations(params))

# test de la fonction equation
print(equation(1, 2, 1))

# enregistrement de la liste des solutions dans une table sous mysql avec la librairie mysql.connector
try:
    connection = mysql.connector.connect(host='localhost', database='equations', user='root', password='')
    if connection.is_connected():
        db_Info = connection.get_server_info()
        print("Connected to MySQL Server version ", db_Info)
        cursor = connection.cursor()
        cursor.execute("select database();")
        record = cursor.fetchone()
        print("You're connected to database: ", record)
        cursor.execute("DROP TABLE IF EXISTS solutions")
        cursor.execute("CREATE TABLE solutions (id INT AUTO_INCREMENT PRIMARY KEY, solution VARCHAR(255))")
        solutions = equations(params)
        for solution in solutions:
            cursor.execute("INSERT INTO solutions (solution) VALUES ('"+str(solution)+"')")
        connection.commit()
        cursor.execute("SELECT * FROM solutions")
        records = cursor.fetchall()
        print("List of solutions")
        for record in records:
            print(record)

except mysql.connector.Error as e:
    print("Error while connecting to MySQL", e)