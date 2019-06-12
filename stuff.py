import csv
import statistics as st
read_file = open('crimen.csv', 'r')
file = csv.DictReader(read_file)

colonias = []
mes = []
for line in file:

    col = line['Colonia']
    delito = line['Delito']
    mes = line ['Mes']

    if col != 'CULIACAN' and 'DESCONOCIDO':
        colonias.append(col)
    #print(mes)


#print(colonias)
print("la colonia mas culera es ", st.mode(colonias))
del colonias
