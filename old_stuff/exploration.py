import json
import plotly.express as px
import pandas as pd

f = open('config_results.json', )
data = json.load(f)

output = {}
configs = []
for x in data['configs']:
    configs.append({
        'population_limit': x['evolution']['population_limit'],
        'fitness_function': x['evolution']['fitness_function'],
        'selection_criterion': x['evolution']['selection_criterion'],
        'episodes_per_individual': x['evolution']['episodes_per_individual'],
        'mutation_rate': x['evolution']['mutation_rate'],
        'survival_rate': x['evolution']['survival_rate'],
        'steps': x['stats']['Time steps'],
        'rule': x['stats']['Rule'],
        'step_history': x['stats']['Time step history']
    })


df = pd.DataFrame(configs)

print(df.columns)

#fig = px.line(df, x="generation", y="0.2percent", title='Nje')
#fig.show()
