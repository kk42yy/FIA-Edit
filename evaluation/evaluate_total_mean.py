import numpy as np
import csv

def calculate_mean(file_path, output_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data_list = list(reader)

    header = data_list[0]
    rows = data_list[1:]


    labels = []
    values = []

    for row in rows:
        if not row or len(row) < 2:
            continue

        category = row[0][0]
        if category not in '0123456789':
            continue
        labels.append(int(category))
    
        value_row = [float(x) if x != '' and x.lower() != 'nan' else np.nan for x in row[1:]]
        values.append(value_row)


    labels = np.array(labels)
    values = np.array(values)


    overall_mean = np.nanmean(values, axis=0)

    results = [overall_mean]
    result_labels = ['Overall']

    for i in range(10):
        class_values = values[labels == i]
        if class_values.size == 0:
            mean_vals = np.full(values.shape[1], np.nan)
        else:
            mean_vals = np.nanmean(class_values, axis=0)
        results.append(mean_vals)
        result_labels.append(f'Class_{i}')


    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
    
        metric_header = header[1:]
        writer.writerow(['Category'] + metric_header)

        for label, row in zip(result_labels, results):
            writer.writerow([label] + list(row))

    print(f'✅ NumPy处理完成，结果已保存到:\n{output_path}')