import pandas as pd
import numpy as np


df = pd.read_csv('countries.csv')


#dfpara imprimir en la consola#

# df.tail()  vemos los ultimos registros: podemos especificar cuantos o por defecto serian 5
#df.shape  NOS DEJA VER CUANTOS DATOS Y EL NUMERO DE COLUMNAS
# df.info()  Nos deja ver un recuento de los datos que tengamos
# df.colun   vemos ls datos de la columna

# df.describe()

# df.values, nos regresa un frame de los datos que le estamos asignando

#df.mean()
# df.max()


# df =df.rename(columns={'nombre_columna':'gdp'})      CAMBIAMOS EL NOMBRE DE LA COLUMNA

# df.nom,_columna    /df['nom_column']         dataframe es mas rapido que una lista

#Filtros como en excel----------------------------------------------------------------

# dfmex = df[df.country == 'Mexico']   filtramos todos los datos que tenemos con un filtro

# dfmex = df[df.country == 'Mexico']amperson  df[df.year >= 1998]
# df[df.country == 'Mexico'][df.year >= 1998]      DA MUCHISIMOS PEDOS

#REINDEXAR Y ELIMINAR EL INDICE ORIGINAL 
# DF_MAX.SORT_VALUES('DATE')



# df.hist()
# df_max.plot(x='ýear', y='country')
# pd.plotting.sacter_matrix(dato)    ESTOS SON LOS GRAFICOS DE DSPERSION


# cuantos y cuales paises tienen una esperanza de vida mayor o igual al 2002
# paises = liat(df[df.year==2002][df.lifeExp>=80].country)

# Pais con el mayor producto interno bruto
# pais = list(df[df.gdpPercap == '976.729.753'])
# solucion profe  df[df.gdpPercap==max(df.gdpPercap)].country.iat[0]

# en que año mexico supro los 70 millones de habitantes
# df[df.pop>=7000000][df.country == 'Mexico'].year.iat[0]

# df[df['pop'] > 70000000][df.country == 'Mexico'].sort_values('pop').year.iat[0]

# */