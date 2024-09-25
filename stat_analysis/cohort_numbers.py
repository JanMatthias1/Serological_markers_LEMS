import pandas as pd
import numpy as np
import seaborn as sns
import pylab as pl
import matplotlib
import matplotlib.pyplot as plt

"""Feature MISSING creation from Murnau data set

Script with all of the cohort calculations 

"""

# metadata, for included in the study (calculates amount of values present)
def metadata (data, baseline, outcome, baseline_asia, name,meta_frame):
    # first input is the actul data frame we want to analyse
    #  second is the outcome, q3 or q4,...
    # meta_frame is the data frame on which we want it to append

    name = name
    patients = str(len(data))
    mean_age = (data["age"]).mean()
    sd_age = (data["age"]).std()
    sex = data["Sex"].value_counts()
    sex_normalise = data["Sex"].value_counts(normalize=True)
    acute = (data[baseline]).mean()
    acute_n = data[baseline].isnull().value_counts()[False]
    acute_sd = (data[baseline]).std()
    chronic = (data[outcome]).mean()
    chronic_n = chronic_n = data[outcome].isnull().value_counts()[False]
    chronic_sd = (data[outcome]).std()

    #print(chronic_sd, (data[outcome]).std(), np.nanstd(data[outcome]),np.std(data[outcome]))

    asia = pd.DataFrame(data[baseline_asia].value_counts()).reset_index()
    asia.columns = ['Category', 'Count']


    asia_normalised = pd.DataFrame(data[baseline_asia].value_counts(normalize=True)).reset_index()
    asia_normalised.columns = ['Category', 'Count']

    required_categories = ['A', 'B', 'C', 'D', 'E']
    missing_categories = set(required_categories) - set(asia['Category']) # will display the missing categories if any

    for category in missing_categories: # if there is, add a rows with the missing category and count 0
        print(category)
        new_row=pd.DataFrame({'Category': [category],'Count': [0]})
        asia = pd.concat([asia,new_row], ignore_index=True)
        print(asia)

    for category in missing_categories:
        print(category)
        new_row=pd.DataFrame({'Category': [category],'Count': [0]})
        asia_normalised = pd.concat([asia_normalised,new_row], ignore_index=True)
        print(asia)

    data= [patients , mean_age, sd_age,sex["m"], sex["f"], sex_normalise["m"], sex_normalise["f"], acute,acute_n, acute_sd, chronic,chronic_n, chronic_sd,
           asia.loc[asia["Category"]=="A", "Count"].values[0],asia_normalised.loc[asia_normalised["Category"]=="A", "Count"].values[0],
           asia.loc[asia["Category"]=="B", "Count"].values[0],asia_normalised.loc[asia_normalised["Category"]=="B", "Count"].values[0],
           asia.loc[asia["Category"]=="C", "Count"].values[0],asia_normalised.loc[asia_normalised["Category"]=="C", "Count"].values[0],
           asia.loc[asia["Category"]=="D", "Count"].values[0],asia_normalised.loc[asia_normalised["Category"]=="D", "Count"].values[0],
           asia.loc[asia["Category"]=="E", "Count"].values[0],asia_normalised.loc[asia_normalised["Category"]=="E", "Count"].values[0]]
    # need to do .values[0] as the filter gives a series and need to select the asia score

    meta_frame[name]= data
    # add the column

    return meta_frame

# metadata, for included in the study (calculates amount of values NOT present)
def metadata_missing (data, baseline, outcome, baseline_asia, name,meta_frame):
    # need to use to different ones, as the metadata_missing will contain null values
    # first input is the actul data frame we want to analyse
    #  second is the outcome, q3 or q4,...
    # meta_frame is the data frame on which we want it to append

    name = name
    patients = str(len(data))
    mean_age = (data["age"]).mean()
    sd_age = (data["age"]).std()
    sex = data["Sex"].value_counts()
    sex_normalise = data["Sex"].value_counts(normalize=True)
    acute = (data[baseline]).mean()
    acute_n = data[baseline].isnull().value_counts()[False]
    acute_sd = (data[baseline]).std()
    chronic = (data[outcome]).mean()
    chronic_n = chronic_n = data[outcome].isnull().value_counts()[False]
    chronic_sd = (data[outcome]).std()

    asia = pd.DataFrame(data[baseline_asia].value_counts()).reset_index()
    asia.columns = ['Category', 'Count']

    asia_normalised = pd.DataFrame(data[baseline_asia].value_counts(normalize=True)).reset_index()
    asia_normalised.columns = ['Category', 'Count']

    required_categories = ['A', 'B', 'C', 'D', 'E']
    missing_categories = set(required_categories) - set(asia['Category'])

    for category in missing_categories:
        print(category)
        new_row = pd.DataFrame({'Category': [category], 'Count': [0]})
        asia = pd.concat([asia, new_row], ignore_index=True)
        print(asia)

    for category in missing_categories:
        print(category)
        new_row = pd.DataFrame({'Category': [category], 'Count': [0]})
        asia_normalised = pd.concat([asia_normalised, new_row], ignore_index=True)
        print(asia)

    data = [patients, mean_age, sd_age, sex["m"], sex["f"], sex_normalise["m"], sex_normalise["f"], acute, acute_n,
            acute_sd, chronic, chronic_n, chronic_sd,
            asia.loc[asia["Category"] == "A", "Count"].values[0],
            asia_normalised.loc[asia_normalised["Category"] == "A", "Count"].values[0],
            asia.loc[asia["Category"] == "B", "Count"].values[0],
            asia_normalised.loc[asia_normalised["Category"] == "B", "Count"].values[0],
            asia.loc[asia["Category"] == "C", "Count"].values[0],
            asia_normalised.loc[asia_normalised["Category"] == "C", "Count"].values[0],
            asia.loc[asia["Category"] == "D", "Count"].values[0],
            asia_normalised.loc[asia_normalised["Category"] == "D", "Count"].values[0],
            asia.loc[asia["Category"] == "E", "Count"].values[0],
            asia_normalised.loc[asia_normalised["Category"] == "E", "Count"].values[0]]
    meta_frame[name] = data

    return(meta_frame)
def data_set(name, marker_excluded,meta_frame,data, table_name, va_lems, lems_chronic, asia):

    # name --> is how the serological marker column ends
    # marker_excluded --> markers to exclude from the data set
    # meta_frame, data frame to which we are appending the column
    # data --> data to use
    # table_name, name for the column --> key of dictionary

    df_marker= data.copy()

    marker = [x for x in df_marker.columns if x.endswith(name)]
    print(marker)
    # markers to remove, due to correlation > 0.70
    marker_filtered = [ele for ele in marker if ele not in marker_excluded]

    if table_name == "Measured_7" or table_name == "Marker_measured_3":
        marker_filtered=marker_excluded

    # adding columns that are necessary
    # age, sex, patient number are present in all values
    # want to drop patient if they do not have a marker_filtered or va_lems or lems_chronic
    columns = marker + [va_lems, lems_chronic] # will use this variable to drop subject
    df_marker = df_marker.dropna(subset=columns) # drop for these columns,

    df_marker=df_marker[columns + ["age", "patient_number", "Sex","va_ais"]]
    print(" THE COLUMNS ARE",df_marker.columns)
    # patients that are included in the included cohort
    patients_included_marker= df_marker["patient_number"]
    # meta_frame is the initial data frame need to give, it will output the modified version
    meta_frame= metadata(df_marker, va_lems, lems_chronic, asia, f"{table_name} included",meta_frame )
    # here lems_q6 is the outcome varibale (c_lems) with LOCF
    # initial variable is va_ais which depending on the data set will have carried over vs

    marker_excluded=data.copy()
    # ~ negates it, take patients in smaller data set and see if in the bigger ones they are present
    # taking all the patients that were excluded by doing it this way
    marker_excluded= marker_excluded[~marker_excluded.patient_number.isin(patients_included_marker)]
    print(marker_excluded["patient_number"])
    print(len(marker_excluded))
    meta_frame= metadata_missing(marker_excluded, va_lems, lems_chronic, asia, f"{table_name} excluded",meta_frame )

    return meta_frame

def loop(data, data_missing, name_data_frame_export,meta_frame, acute, lems_chronic, asia):

    mean = ["hemoglobin_mean", "hematocrit_mean", "hemoglobin_per_erythrocyte_mean", "blood_urea_nitrogen_mean",
            "quick_test_mean", "total_proteins_mean"]

    median = ["hemoglobin_median", "hematocrit_median", "hemoglobin_per_erythrocyte_median", "total_proteins_median",
              "blood_urea_nitrogen_median", "lipase_median", "quick_test_median"]

    min = ["hemoglobin_min", "hematocrit_min", "hemoglobin_per_erythrocyte_min", "total_proteins_min"]

    marker_max = ["hemoglobin_max", "hematocrit_max", "hemoglobin_per_erythrocyte_max"]

    marker_range = ["hemoglobin_range", "hematocrit_range", "cholinesterase_range", "total_proteins_range",
                    "alkaline_phosphatase_range", "quick_test_range"]

    measured_7 = ["hemoglobin_per_erythrocyte_times_measured_7", "total_bilirubin_times_measured_7"]

    marker_measured_3 = ["hemoglobin_per_erythrocyte_times_measured_3", "total_bilirubin_times_measured_3"]

    marker_dictionary={"Mean":mean,
                       "Median":median,
                       "Min":min,
                        "Max":marker_max,
                        "Range":marker_range,
                       }
    # create dictionary which has the serological markers to exclude due to high correlation

    name=["_mean", "_median", "_min","_max", "_range"] #how the name of the column is (serologicalmarker_name)
    name_missing=["_measured_7", "_measured_3"]

    marker_dictionary_missing={"Measured_7": measured_7,
                       "Marker_measured_3": marker_measured_3}

    for idx, (key, value) in enumerate(marker_dictionary.items()):
        print(key)
        meta_frame= data_set(name[idx], value, meta_frame, data, key, acute, lems_chronic, asia)

        # call data_set function, which will create a column with the patients included in this specific cohort
        # and also a second column with the patients being excluded

    for idx, (key, value) in enumerate(marker_dictionary_missing.items()):
        print(key)
        meta_frame= data_set(name_missing[idx], value, meta_frame, data_missing, key, acute, lems_chronic,asia)

    #meta_frame.to_csv(f'/Users/janmatthias/Documents/GitHub/Internship_BDS/Final_cohort_info/{name_data_frame_export}_05_02.csv')

meta_frame_original= pd.DataFrame(index= ["patients","mean_age", "sd_age", "sex_males","sex_females", "sex_normalise_male", "sex_normalise_female", "acute","acute_n",
                                 "acute_sd", "chronic","chronic_n", "chronic_sd", "asia_A","asia_A_norm", "asia_B","asia_B_norm","asia_C","asia_C_norm", "asia_D",
                                 "asia_D_norm", "asia_E","asia_E_norm",])

df_imputed= pd.read_csv(r'')
df_missing_imputed= pd.read_csv(r'')

meta_frame=meta_frame_original.copy()
# imputed va_lems and LOCF lems chronic --> va_lems imputed and lems_q6
loop(df_imputed, df_missing_imputed, "meta_data_imputed",meta_frame, "va_lems_imputed","lems_q6","va_ais")


# No impuation at all, how would the cohort look?
meta_frame=meta_frame_original.copy()
loop(df_imputed, df_missing_imputed, "meta_data",meta_frame,"va_lems", "c_lems","va_ais")