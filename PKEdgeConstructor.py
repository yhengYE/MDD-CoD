import pandas as pd
import re
from itertools import combinations
from collections import defaultdict

df = pd.read_excel(r'data/drug/kd2用药3.xlsx', usecols=[4])

df['Medications'] = df.iloc[:, 0].apply(lambda x: re.split(r'[，, ]{1}', x) if isinstance(x, str) else [])

co_occurrence = defaultdict(int)

for meds in df['Medications']:
    for med1, med2 in combinations(set(meds), 2):
        if med1 > med2:
            med1, med2 = med2, med1
        co_occurrence[(med1.strip(), med2.strip())] += 1

co_occurrence_df = pd.DataFrame(((med1, med2, count) for (med1, med2), count in co_occurrence.items()), columns=['Med1', 'Med2', 'Count'])

max_count = co_occurrence_df['Count'].max()
co_occurrence_df['Weight'] = co_occurrence_df['Count'] / max_count

output_path = 'data\drug/药物共现kd2.xlsx'
co_occurrence_df.to_excel(output_path, index=False)

print('文件已保存至:', output_path)
