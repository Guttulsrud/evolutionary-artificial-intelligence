import pandas as pd
import json
import plotly.express as px

file_path = 'data.json'

with open(file_path, ) as file:
    data = json.load(file)

individuals = [generation['individuals'] for generation in data['generations'] if 'individuals' in generation.keys()]
df = pd.DataFrame(individuals)

fig = px.violin(df.transpose(), points=False)
fig.show()