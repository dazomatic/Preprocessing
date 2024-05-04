#Imported the DF
import pandas as pd
columns = ['code', 'lc', 'product_name_en', 'quantity', 'serving_size', 'packaging_tags', 'brands', 'brands_tags', 'categories_tags', 'labels_tags', 'countries', 'countries_tags', 'origins','origins_tags']
avocado = pd.read_csv('data/avocado.csv', sep='\t', usecols = columns)
olive = pd.read_csv('data/olive_oil.csv', sep='\t', usecols = columns)
sourdough = pd.read_csv('data/sourdough.csv', sep='\t', usecols = columns)

#Creating a list of relevant 'tags'
with open('data/relevant_avocado_categories.txt', "r") as file:
    relevant_avocado_categories = file.read().splitlines()
    file.close()

#splitting the tags into a list that were separated by commas
avocado['categories_tags'] = avocado['categories_tags'].str.split(',')
avocado.dropna(subset=['categories_tags'], inplace=True)

#run a lambda apply on each value to check if it matches necessary tag list 
avocado = avocado[avocado['categories_tags'].apply(lambda x: any([i for i in x if i in relevant_avocado_categories]))]