import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv')

sequence_lengths = sorted(df['sequence_length'].unique())
error_rates = sorted(df['error_rate'].unique())

colors = {5: 'blue', 10: 'green', 15: 'red'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for er in error_rates:
    subset = df[df['error_rate'] == er]
    ax1.plot(subset['sequence_length'], subset['throughput'], marker='o', label=f'Error Rate {er}%', color=colors[er])

ax1.set_title('Throughput vs Sequence Length')
ax1.set_xlabel('Sequence Length')
ax1.set_ylabel('Alignments per Second')
ax1.grid(True)
ax1.legend()

for er in error_rates:
    subset = df[df['error_rate'] == er]
    ax2.plot(subset['sequence_length'], subset['execution_time'], marker='o', label=f'Error Rate {er}%', color=colors[er])

ax2.set_title('Execution Time vs Sequence Length')
ax2.set_xlabel('Sequence Length')
ax2.set_ylabel('Total Execution Time (s)')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()
