import pandas as pd
import csv
import numpy as np

def check_none_validations(data):
    none_group = {}
    for _, row in data.iterrows():
        if row['Ground Truth'] != 0 and row['Ground Truth'] != 1:
            none_group[(row['News'], row['Perspective'])] = row['Ground Truth']
    return none_group

def check_inconsistencies(data):

    inconsistencies = {}
    for _, row in data.iterrows():
        for prespective in ['Democrat', 'Republican', 'Neutral']:
            if prespective == 'Democrat':
                if row['Count True Democrat'] != iter and row['Count False Democrat'] != iter:
                    inconsistency_number = min(row['Count True Democrat'], row['Count False Democrat'])
            elif prespective == 'Republican':
                if row['Count True Republican'] != iter and row['Count False Republican'] != iter:
                    inconsistency_number = min(row['Count True Republican'], row['Count False Republican'])
            elif prespective == 'Neutral':
                if row['Count True Neutral'] != iter and row['Count False Neutral'] != iter:
                    inconsistency_number = min(row['Count True Neutral'], row['Count False Neutral'])
            if inconsistency_number != 0:
                inconsistencies[(row['News'], prespective)] = inconsistency_number

    return inconsistencies

def save_inconsistencies_to_csv(inconsistencies, file_path):
    with open(file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        writer.writerow(['News', 'Perspective', 'Inconsistency Number'])

        for (news, perspective), count in inconsistencies.items():
            writer.writerow([news, perspective, count])

def save_none_group_to_csv(none_group, file_path):
    with open(file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        writer.writerow(['News', 'Perspective', 'Validations'])

        for (news, perspective), count in none_group.items():
            writer.writerow([news, perspective, count])


# inconsistencies = check_inconsistencies("updated.csv")
# none_group = check_none_validations("explanations.csv")
# save_inconsistencies_to_csv(inconsistencies, "inconsistencies.csv")
# save_none_group_to_csv(none_group, "none_values.csv")

import pandas as pd
import numpy as np

def compute_confusion_matrices(updated_file, leaning_file, iter):
    df_updated = pd.read_csv(updated_file, delimiter=',')
    df_leaning = pd.read_csv(leaning_file, delimiter=',')
    
    df_updated = df_updated.merge(df_leaning[['News', 'Leaning']], on='News', how='left')
    
    filtered_df = df_updated[df_updated['Leaning'] == 'R']
    
    prob_1_democrat = filtered_df['Count True Democrat'] / iter
    prob_0_democrat = filtered_df['Count False Democrat'] / iter
    prob_1_republican = filtered_df['Count True Republican'] / iter
    prob_0_republican = filtered_df['Count False Republican'] / iter
    prob_1_neutral = filtered_df['Count True Neutral'] / iter
    prob_0_neutral = filtered_df['Count False Neutral'] / iter
    ground_truth = filtered_df['Ground Truth']
    
    def get_confusion_matrix(ground_truth, prob_1, prob_0):
        TP = np.sum(ground_truth * prob_1)
        FP = np.sum((1 - ground_truth) * prob_1)
        FN = np.sum(ground_truth * prob_0)
        TN = np.sum((1 - ground_truth) * prob_0)
        return np.array([[TP, FP], [FN, TN]])
    
    confusion_matrix_prob_democrat = get_confusion_matrix(ground_truth, prob_1_democrat, prob_0_democrat)
    confusion_matrix_prob_republican = get_confusion_matrix(ground_truth, prob_1_republican, prob_0_republican)
    confusion_matrix_prob_neutral = get_confusion_matrix(ground_truth, prob_1_neutral, prob_0_neutral)
    
    return confusion_matrix_prob_democrat, confusion_matrix_prob_republican, confusion_matrix_prob_neutral

confusion_matrix_democrat, confusion_matrix_republican, confusion_matrix_neutral = compute_confusion_matrices(
   updated_file='/LLaMa/updated1.csv',
   leaning_file='/news_leaning_dataset.csv',
   iter=10
)

print("Confusion Matrix - Democrat:\n", confusion_matrix_democrat)
print("Confusion Matrix - Republican:\n", confusion_matrix_republican)
print("Confusion Matrix - Neutral:\n", confusion_matrix_neutral)


def report_performance_through_leanings():
    df = pd.read_csv('news_leaning_dataset.csv', delimiter=',')
    
    r001 = []
    d110 = []
    d010 = []
    r101 = []
    
    for _, row in df.iterrows():
        leaning = row['Leaning']
        democrat_response = row['ChatGPT’s response from the perspective of a Conservative (Democrat) viewpoint']
        republican_response = row['ChatGPT’s response from the perspective of a Rdaical (Republican) viewpoint']
        if leaning == 'R' and row['Ground Truth'] == 0 and democrat_response == 0 and republican_response == 1:
            r001.append(row['News'])
        elif leaning == 'D' and row['Ground Truth'] == 1 and democrat_response == 1 and republican_response == 0:
            d110.append(row['News'])
        elif leaning == 'D' and row['Ground Truth'] == 0 and democrat_response == 1 and republican_response == 0:
            d010.append(row['News'])
        elif leaning == 'R' and row['Ground Truth'] == 1 and democrat_response == 0 and republican_response == 1:
            r101.append(row['News'])
    print('_______________________________________________________')
    print('Leaning: R, Ground Truth: 0, Democrat: 0, Republican: 1')
    for news in r001:
        print(news)
    print('_______________________________________________________')
    print('Leaning: D, Ground Truth: 1, Democrat: 1, Republican: 0')
    for news in d110:
        print(news)
    print('_______________________________________________________')
    print('Leaning: D, Ground Truth: 0, Democrat: 1, Republican: 0')
    for news in d010:
        print(news)
    print('_______________________________________________________')
    print('Leaning: R, Ground Truth: 1, Democrat: 0, Republican: 1')
    for news in r101:
        print(news)

# report_performance_through_leanings()
df = pd.read_csv('news_leaning_dataset.csv', delimiter=',')
iter = 10
d1 = 0
r1 = 0
d2 = 0
r2 = 0
d3 = 0
r3 = 0
for _, row in df.iterrows():
        if row['Ground Truth'] == 1 and row['Leaning'] == 'N':
            d1 += 1
        elif row['Ground Truth'] == 0 and row['Leaning'] == 'N':
            r1 += 1
        elif row['Ground Truth'] == 1 and row['Leaning'] == 'R':
            d2 += 1
        elif row['Ground Truth'] == 0 and row['Leaning'] == 'R':
            r2 += 1
        elif row['Ground Truth'] == 1 and row['Leaning'] == 'V':
            d3 += 1
        elif row['Ground Truth'] == 0 and row['Leaning'] == 'V':
            r3 += 1
# print(d1, r1, d2, r2, d3, r3)