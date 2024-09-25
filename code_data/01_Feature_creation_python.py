"""Feature creation from Murnau data set

The purpose of this script is to create the features needed for the machine learning models

    1) As the orginal data set contains different amounts of columns for the different bloodmarkers and contains multiple
null values, here we are taking the average, median, min, max and range value over a 7 days period. By doing this we are
minimising the amount of null values present.

    2) Aditionally we are also accounting for the amount of blood values drawn. To do this we are counting the amount of times
the specific serological markers was measured.

"""
import pandas as pd
import numpy as np

# loading the various data frames from the server
data=pd.read_csv(r'', encoding='latin-1', low_memory=False)
data.rename(columns={"ï»¿patientennummer": "patient_number"}, inplace=True)

df_sex= pd.read_excel(r'')
df_sex.rename(columns={"Patientennummer": "patient_number"}, inplace=True)

df_age_calculated= pd.read_csv('')

# basic conversion on the whole data set
values_to_replace = ['+', ' +', '++', ' ++', '+++', ' +++','(+)', 'negativ', 'positiv', 'zu hämolytisch', "ND", 'nan']

for value in values_to_replace:
    data = data.replace([value], [np.nan])
# creating a new data frame with all  necessary features --> will be called df
# creating the new data frame
df  = pd.DataFrame(columns=['patient_number'])

#getting the patient numbers from data
df= df.assign(patient_number=data["patient_number"])

# merging data column with sex of all patients
df=pd.merge(df, df_sex[["patient_number","Sex"]], how= "left", on="patient_number")

# some females are marked as w, need to change to f
df['Sex'] = df['Sex'].replace(["w"], ["f"])

# merging column with age of patients
df=pd.merge(df, df_age_calculated[["patient_number","together"]], how= "left", on="patient_number")
df.rename(columns={"together": "age"}, inplace=True)

# merging the LEMS scores
df=pd.merge(df, data[["patient_number","va_lems", "ai_lems","aii_lems","aiii_lems","c_lems"]],how= "left", on="patient_number")

# merging the ASIA scores
df=pd.merge(df, data[["patient_number","va_ais","ai_ais","aii_aiis","aiii_aiiis","c_cs", "va_nli", "ai_nli", "aii_nli", "aiii_nli", "c_nli",
                      "va_testdate", "ai_testdate", "aii_testdate", "aiii_testdate", "c_testdate", "date_of_injury"]], how= "left", on="patient_number")

# need to make new columns for the various blood markers, for the mean, median, min, max and range
suffixes = ["_mean", "_median", "_min", "_max", "_range"]

columns = ["erythrocytes","hemoglobin","hematocrit","MCHC","MCV","thrombocytes", "leucocytes","hemoglobin_per_erythrocyte",
   "alkaline_phosphatase", "ASAT","ALAT", "total_bilirubin", "gamma_GT", "lactate_dehydrogenase", "calcium",
  "creatin", "total_proteins", "blood_urea_nitrogen", "potassium","sodium", "cholinesterase","amylase",
   "lipase","glucose", "INR","partial_thromboplastin_time","CRP","quick_test"]

patient_numbers = data["patient_number"]

# making the lists with the specific column names (hemoglobin_meean, MCV_mean,...)
bloodmarkers_lists = {suffix: [col + suffix for col in columns] for suffix in suffixes}

# making dataframes from the suffix, each suffix creates a different dataset
bloodmarkers_dataframes = {suffix: pd.DataFrame(columns=columns, index=patient_numbers) for suffix, columns in bloodmarkers_lists.items()}

# creating a dictionary with the initial names for each column, as each bloodmarker has a different length

names_to_add=["ery_","hb_","hk_","mchc_","mcv_", "thrombo_","leuco_nl_", "hbe_",
       "ap_","got_","gpt_", "gesamt_bilirubin_", "gamma_gt_","ldh_","calcium_",
       "creatinin_", "gesamteiweiss_", "harnstoff_", "kalium_", "natrium_","che_", "amylase_",
       "lipase_", "glucose_", "inr_","ptt_", "crp_", "quick_" ]

# will be adding this string to names_to_add
string=["0", "1","2","3", "4", "5","6"]

# days that we are looking at
names=["names_0", "names_1", "names_2", "names_3", "names_4", "names_5", "names_6"]

# making a dictionary, where names_0, will contain a list of the various column names at day 0 for the bloodmarker
name_dictionary = {name: np.array([]) for name in names}

# adding the modified names to the dictionary
for i in range(len(name_dictionary)):
    name_dictionary[names[i]]= (pd.Series(names_to_add) + string[i]).tolist()
    # with the name, will have the beginning of all 28 serological marker columns

# to filter the various names, we need the length of each one
# here we are making an array with the length of all the variables
length = []

for i in range(len(name_dictionary[names[0]])):
    length = np.append(length, len(name_dictionary[names[0]][i]) + 1)
    # here I am using the array, only the first list in it (0) and then iterating through it
    # doing it with the first position, ery_0 as this ensures we are not selecting ery_110

# loop for all 28 blood markers, and calculate average, median, min and max

# Creating a dictionary that contains the initial column names for all of the bloodmarkers, it will be used
# to search for the specific column in the whole data frame

for b in range(len(columns)):
    # here starting with 0 to 6, if the data columns starts with, for example ery_0 add columns to marker_1
    # names goes from day 0 to day 7 --> names_0, names_01
    # b iterates through all the columns

    marker_1 = [x for name in names for x in data.columns if x.startswith(name_dictionary[name][b])]
    # here iterating through all names (name_0, name_1,...) and adding all the columns from data, that
    # start with the initial part

    # marker_1 will contain all columns for the first 7 days of that specific serological marker

    marker_tot = []

    # filtering it for the correct length
    for i in range(len(marker_1)):
        if len(marker_1[i]) <= (length[b]):
            # here iterating through the entire marker_1 and making sure they all are of the correct length
            marker_tot = np.append(marker_tot, marker_1[i])

    # adding the various data to the bloodmarkers_mean data frame
    # makes the average of marker_tot (all the various markers for that time period, in this case 7 days)
    # makes the average for each row, which corrisponds to each patient

    for i, patient in enumerate(patient_numbers):

        data_patient = data[data["patient_number"] == patient]

        if not data_patient[marker_tot].isnull().all(axis=1).any():

            bloodmarkers_dataframes["_mean"].iloc[i, b]= np.nanmean(data_patient[marker_tot], axis=1).item()
            bloodmarkers_dataframes["_median"].iloc[i, b] = np.nanmedian(data_patient[marker_tot], axis=1).item()
            bloodmarkers_dataframes["_min"].iloc[i, b] = np.nanmin(data_patient[marker_tot], axis=1).item()
            bloodmarkers_dataframes["_max"].iloc[i, b] = np.nanmax(data_patient[marker_tot], axis=1).item()

        else:
            bloodmarkers_dataframes["_mean"].iloc[i, b] = np.nan
            bloodmarkers_dataframes["_median"].iloc[i, b] = np.nan
            bloodmarkers_dataframes["_min"].iloc[i, b] = np.nan
            bloodmarkers_dataframes["_max"].iloc[i, b]= np.nan

# after adding the various numbers to the appropriate data frame, need to merge it all with the initial data frame
merge_list = [bloodmarkers_dataframes["_mean"], bloodmarkers_dataframes["_median"], bloodmarkers_dataframes["_min"], bloodmarkers_dataframes["_max"]]

for marker_df in merge_list:
    df = df.merge(marker_df, how="right", on="patient_number")

# Calculating the RANGE between min and max values for the various bloodmarker values
# need the patient numbers in a list in order to iterate through it
patient_number = data["patient_number"].to_list()

# the two lists with the various column names for min and max values --> are present in the min and max columns of df dataframe
for i in range(len(patient_number)):

    # filtering it for the correct patient number
    patient = patient_number[i]

    # using df data frame as it has the min and max values in one data frame
    # here we select only one patient row
    data_patient = df[df["patient_number"] == patient]

    for b in range(len(bloodmarkers_dataframes["_max"].columns)):
        # after choosing the specific patient, it iterates through all the columns of the patient

        bloodmarkers_dataframes["_range"].iloc[i, b] = pd.Series.abs(data_patient[bloodmarkers_lists["_max"][b]] - data_patient[bloodmarkers_lists["_min"][b]])[i]
        # taking the patient's max value and subtracting it to the min
        # by putting i outside it filters it for the specific patint, and adds only that value to the bloodmarker_range data frame

df=pd.merge(df, bloodmarkers_dataframes["_range"], how= "right", on="patient_number")

df['lems_q6'] = df['c_lems'].fillna(df["aiii_lems"])
df['lems_q6_q3'] = df['lems_q6'].fillna(df["aii_lems"])

# carrying over the asia score to a new column called asia_chronic
#"va_ais", "ai_ais", "aii_aiis", "aiii_aiiis", "c_cs"
df['asia_chronic_q6'] = df['c_cs'].fillna(df["aiii_aiiis"])
df['asia_chronic_q6_q3'] = df['asia_chronic_q6'].fillna(df["aii_aiis"])


# index for patients who have all 0 values
all_0_index=[] # insert 0

df["va_lems_imputed"]=df["va_lems"]
for i in range(len(all_0_index)):
    # create a new columns with the imputed values
    df.at[all_0_index[i],"va_lems_imputed"]= 0

# removed due to abnormal decrease in LEMS score
df=df.drop(index=[])

df.to_csv('')