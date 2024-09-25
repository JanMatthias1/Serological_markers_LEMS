import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.patches as patches


x_axis_names=["Erythrocytes","Hemoglobin","Hematocrit","MCHC","MCV","Thrombocytes", "Leucocytes","Hemoglobin per erythrocyte",
   "Alkaline Phosphatase", "ASAT","ALAT", "Total Bilirubin", "Gamma GT", "Lactate Dehydrogenase", "Calcium",
  "Creatin", "Total Proteins", "Blood Urea Nitrogen", "Potassium","Sodium", "Cholinesterase","Amylase",
   "Lipase","Glucose", "INR","Partial Thromboplasmin Time","CRP","Quick Test","VA Lems", "Age"]

index_names=["erythrocytes","hemoglobin","hematocrit","MCHC","MCV","thrombocytes", "leucocytes","hemoglobin_per_erythrocyte",
   "alkaline_phosphatase", "ASAT","ALAT", "total_bilirubin", "gamma_GT", "lactate_dehydrogenase", "calcium",
  "creatin", "total_proteins", "blood_urea_nitrogen", "potassium","sodium", "cholinesterase","amylase",
   "lipase","glucose", "INR","partial_thromboplasmin_time","CRP","quick_test", "age","va_lems_imputed"]

df=pd.DataFrame(index=index_names + ["lems_q6"])

for subset in ["mean","median","min","max","range","measured_7"]:
    if subset=="measured_7":
        data= pd.read_csv(r'')
        data=data.dropna(subset=["va_lems_imputed", "lems_q6"])
        marker = [x for x in data.columns if x.endswith("_measured_7")] + ["va_lems_imputed","age"]
        data=data.dropna(subset=marker)
        features = [col for col in data[marker].columns if col != 'lems_q6']

        print(data["age"].mean(), data["lems_q6"].mean())
    else:
        data = pd.read_csv(r'')
        data=data.dropna(subset=["va_lems_imputed", "lems_q6"])

        marker = [x for x in data.columns if x.endswith(f'_{subset}')] + ["va_lems_imputed","age"]
        data=data.dropna(subset=marker)
        features = [col for col in data[marker].columns if col != 'lems_q6']

        print(data["age"].mean(), data["lems_q6"].mean())

    correlation_matrix=data[marker +["lems_q6"]].corr()

    if subset=="measured_7":
        new_index = [item.replace(f'_times_{subset}', '') for item in correlation_matrix.index]
        correlation_matrix.index = new_index
    else:
        new_index = [item.replace(f'_{subset}', '') for item in correlation_matrix.index]
        correlation_matrix.index = new_index

    df[subset] = correlation_matrix["lems_q6"]

    np.random.seed(0)
    fig, axes = plt.subplots(6, 5, figsize=(15, 18))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Plot pairwise scatter plots
    for i, feature in enumerate(features):
        sns.regplot(x=data[feature], y=data['lems_q6'], ax=axes[i], scatter_kws={'s': 10, "color":"black"}, line_kws={'color': 'black'})
        axes[i].set_title("")
        axes[i].set_xlabel(x_axis_names[i], fontweight="bold", fontsize=12)
        axes[i].set_ylabel('Chronic LEMS', fontweight="bold", fontsize=12)
        axes[i].set_ylim(-5, 55)

    plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05, wspace=0.4, hspace=0.4)

    # Add box around the last two subplots
    # Get the positions of the 29th and 30th subplots
    pos1 = axes[28].get_position()
    pos2 = axes[29].get_position()

    # Calculate the combined position of the two subplots
    width = pos1.x1 - pos1.x0
    height = pos1.y1 - pos1.y0
    rect = patches.Rectangle((pos1.x0, pos1.y0), width, height, linewidth=2, edgecolor='r', facecolor='none')
    fig.add_artist(rect)

    width = pos2.x1 - pos2.x0
    height = pos2.y1 - pos2.y0
    rect = patches.Rectangle((pos2.x0, pos2.y0), width, height, linewidth=2, edgecolor='r', facecolor='none')
    # Add the Rectangle patch to the figure
    fig.add_artist(rect)

    plt.savefig(f"")
