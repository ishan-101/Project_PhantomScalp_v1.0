
import matplotlib.pyplot as plt
import json

data = [{"script": "test_all_features.py", "count": 58}, {"script": "test_labels.py", "count": 44}, {"script": "prepare_labels_dataset.py", "count": 30}, {"script": "validate_labels_dataset.py", "count": 18}, {"script": "inspect_reversal_debug.py", "count": 17}]

scripts = [d['script'] for d in data]
counts = [d['count'] for d in data]

plt.figure(figsize=(10, 6))
plt.barh(scripts, counts, color='steelblue')
plt.xlabel('Number of Dependencies (Features + Labels + File I/O)')
plt.title('Top 5 Scripts by Dependencies')
plt.tight_layout()
plt.savefig('.ag_artifacts/scripts_file_summaries/scripts_dependency_top5.png', dpi=150, bbox_inches='tight')
print('✓ Generated visualization')
