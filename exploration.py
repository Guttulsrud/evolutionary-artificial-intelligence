import json
import plotly.express as px
import pandas as pd

f = open('res.json', )
data = json.load(f)

output = {}

for x in data['configs']:

    if x['config']['selection_criterion'] == 'fitness_proportional':
        output[x['config']['mutation_rate']] = x['config']['stats']['time_steps_history']

df = pd.DataFrame(output).reset_index()
df.columns = ['generation', '0.2percent', '3percent', '5percent']
print(df)

fig = px.line(df, x="generation", y="0.2percent", title='Nje')
fig.show()