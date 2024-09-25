"""Feature MISSING creation from Murnau data set

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
df=pd.merge(df, data[["patient_number","va_ais","ai_ais","aii_aiis","aiii_aiiis","c_cs"]], how= "left", on="patient_number")


# need to make new columns for the various serological marker
count_=["_count_1","_count_2","_count_3","_count_4","_count_5","_count_6","_count_7"]

columns = ["erythrocytes","hemoglobin","hematocrit","MCHC","MCV","thrombocytes", "leucocytes","hemoglobin_per_erythrocyte",
   "alkaline_phosphatase", "ASAT","ALAT", "total_bilirubin", "gamma_GT", "lactate_dehydrogenase", "calcium",
  "creatin", "total_proteins", "blood_urea_nitrogen", "potassium","sodium", "cholinesterase","amylase",
   "lipase","glucose", "INR","partial_thromboplastin_time","CRP","quick_test"]

# create dictionary with all the possible bloodmarkers (count_1 --> gives all the columns for count_1)
bloodmarkers_count_lists = {count: [col + count for col in columns] for count in count_}

bloodmarkers_count_dataframes = {suffix: pd.DataFrame(columns=columns, index=data["patient_number"]) for suffix, columns in bloodmarkers_count_lists.items()}

# SEARCH DICTIONARY
# creating a dictionary with the initial names for each column, as each bloodmarker has a different length
names_to_add=["ery_","hb_","hk_","mchc_","mcv_", "thrombo_","leuco_nl_", "hbe_",
       "ap_","got_","gpt_", "gesamt_bilirubin_", "gamma_gt_","ldh_","calcium_",
       "creatinin_", "gesamteiweiss_", "harnstoff_", "kalium_", "natrium_","che_", "amylase_",
       "lipase_", "glucose_", "inr_","ptt_", "crp_", "quick_" ]

# will be adding this string to names_to_add
string=["0", "1","2","3", "4","5","6"]

# days that we are looking at
names=["names_0", "names_1", "names_2", "names_3", "names_4", "names_5", "names_6"]

# making a dictionary, where names_0, will contain a list of the various column names at day 0 for the bloodmarker
name_dictionary = {name: np.array([]) for name in names}

# adding the modified names to the dictionary --> contains the initial part of each column name for the serological marker
for i in range(len(name_dictionary)):
    name_dictionary[names[i]]= (pd.Series(names_to_add) + string[i]).tolist()
    # with the name, will have the beginning of all 28 serological marker columns
    # key --> names_0, gives 28 serological markers, with ery_0, hb_0,... which will be used to search in the Murnau data set

# to filter the various names, we need the length of each one
# here we are making an array with the length of all the variables
length = []
for i in range(len(name_dictionary[names[0]])):
    length = np.append(length, len(name_dictionary[names[0]][i]) + 1)
    # here I am using the array, only the first list in it (0) and then iterating through it
    # doing it with the first position, ery_0b (+1) as this ensures we are not selecting ery_110

patient_number = data["patient_number"].to_list()

for i in range(len(patient_number)):
    # filtering it for the correct patient number
    patient = patient_number[i]

    # data_patient only contains the data from that one specific patient, from the original data set (data)
    data_patient = data[data["patient_number"] == patient]

    # first we are choosing a column, this will iterate through all the 28 columns
    for b in range(len(bloodmarkers_count_dataframes["_count_1"].columns)):

        # now calculate notnull() values for the various days
        for a in range(7):
            if a == 0:
                # marker_0 will contain only the column names from the first day
                #
                marker_0 = [x for x in data.columns if x.startswith(name_dictionary[names[0]][b])]

                # calculating how many notnull values we have within data_patient for that specific blood marker and specific patient
                null_amount = data_patient[marker_0].notnull().sum(axis=1)
                print(marker_0, null_amount)
                # inserting the value in the specific column and row for the patient
                bloodmarkers_count_dataframes["_count_1"].iloc[i, b] = (null_amount)[i]

            if a == 1:

                # marker_1 will contain only the column names from the second day
                marker_1 = ([x for x in data.columns if x.startswith(name_dictionary[names[1]][b])])

                marker_tot = []

                # from a==1 on, need to filter for the correct length
                for e in range(len(marker_1)):
                    if len(marker_1[e]) <= (length[b]):
                        marker_tot = np.append(marker_tot, marker_1[e])
                

                null_amount = data_patient[marker_tot].notnull().sum(axis=1)
                print(marker_tot, null_amount)
                bloodmarkers_count_dataframes["_count_2"].iloc[i, b] = (null_amount)[i]

            if a == 2:

                marker_2 = ([x for x in data.columns if x.startswith(name_dictionary[names[2]][b])])
                marker_tot = []
                for e in range(len(marker_2)):
                    if len(marker_2[e]) <= (length[b]):
                        marker_tot = np.append(marker_tot, marker_2[e])
                null_amount = data_patient[marker_tot].notnull().sum(axis=1)
                print(marker_tot, null_amount)
                bloodmarkers_count_dataframes["_count_3"].iloc[i, b] = (null_amount)[i]

            if a == 3:

                marker_3 = ([x for x in data.columns if x.startswith(name_dictionary[names[3]][b])])
                marker_tot = []
                for e in range(len(marker_3)):
                    if len(marker_3[e]) <= (length[b]):
                        marker_tot = np.append(marker_tot, marker_3[e])
                null_amount = data_patient[marker_tot].notnull().sum(axis=1)
                print(marker_tot, null_amount)
                bloodmarkers_count_dataframes["_count_4"].iloc[i, b] = (null_amount)[i]

            if a == 4:

                marker_4 = ([x for x in data.columns if x.startswith(name_dictionary[names[4]][b])])
                marker_tot = []
                for e in range(len(marker_4)):
                    if len(marker_4[e]) <= (length[b]):
                        marker_tot = np.append(marker_tot, marker_4[e])
                null_amount = data_patient[marker_tot].notnull().sum(axis=1)
                print(marker_tot, null_amount)
                bloodmarkers_count_dataframes["_count_5"].iloc[i, b] = (null_amount)[i]

            if a == 5:

                marker_5 = ([x for x in data.columns if x.startswith(name_dictionary[names[5]][b])])
                marker_tot = []
                for e in range(len(marker_5)):
                    if len(marker_5[e]) <= (length[b]):
                        marker_tot = np.append(marker_tot, marker_5[e])
                
                null_amount = data_patient[marker_tot].notnull().sum(axis=1)
                print(marker_tot, null_amount)
                bloodmarkers_count_dataframes["_count_6"].iloc[i, b] = (null_amount)[i]

            if a == 6:

                marker_6 = ([x for x in data.columns if x.startswith(name_dictionary[names[6]][b])])
                marker_tot = []
                for e in range(len(marker_6)):
                    if len(marker_6[e]) <= (length[b]):
                        marker_tot = np.append(marker_tot, marker_6[e])

                
                null_amount = data_patient[marker_tot].notnull().sum(axis=1)
                print(marker_tot, null_amount)
                bloodmarkers_count_dataframes["_count_7"].iloc[i, b] = (null_amount)[i]

bloodmarkers_count_dataframes_df = pd.concat(bloodmarkers_count_dataframes.values(), axis=1)
df=pd.merge(df, bloodmarkers_count_dataframes_df, how= "right", on="patient_number")


# Creating a data frame with the amount of times each bloodmarker was measured
string= ["_times_measured_7"]
string_3= ["_times_measured_3"]

#making the new column names
bloodmarkers_times_measured_7days = (pd.Series(columns) + string).tolist()
bloodmarkers_times_measured_3days = (pd.Series(columns) + string_3).tolist()

# creating the new data frame
bloodmarkers_times_measured_7= pd.DataFrame(columns = bloodmarkers_times_measured_7days, index=[data["patient_number"]])
bloodmarkers_times_measured_3= pd.DataFrame(columns = bloodmarkers_times_measured_3days, index=[data["patient_number"]])

# loop to calculate how many times each blood marker was measured --> MEASURED 7 DAYS
for i in range(len(patient_number)):

    # filtering it for the correct patient number
    patient = patient_number[i]
    data_patient = data[data["patient_number"] == patient]

    # first we are choosing a column
    # columns have 28 bloodmarkers in it, want to iterate through all 28
    for b in range(len(columns)):

        # marker_1 contains all the columns from day 0 to day 7, for the specific serological marker
        marker_1 = [x for name in names for x in data.columns if x.startswith(name_dictionary[name][b])]

        marker_tot = []

        # filter for the correct length
        for e in range(len(marker_1)):
            if len(marker_1[e]) <= (length[b]):
                marker_tot = np.append(marker_tot, marker_1[e])

        # print(marker_tot)

        # counting and adding to the correct position within the data frame
        null_amount = data_patient[marker_tot].notnull().sum(axis=1)
        bloodmarkers_times_measured_7.iloc[i, b] = (null_amount)[i]

# MEASURED 3 DAYS
for i in range(len(patient_number)):

        # filtering it for the correct patient number
        patient= patient_number[i]
        data_patient=data[data["patient_number"]==patient]

        # first we are choosing a column
        # columns have 28 bloodmarkers in it, want to iterate through all 28
        for b in range(len(columns)):

            # marker_1 contains all the columns from day 0 to day 7
            marker_1 = ([x for x in data.columns if x.startswith(name_dictionary[names[0]][b])] +
                        [x for x in data.columns if x.startswith(name_dictionary[names[1]][b])] +
                        [x for x in data.columns if x.startswith(name_dictionary[names[2]][b])])

            marker_tot=[]

            # filter for the correct length
            for e in range(len(marker_1)):
                if len(marker_1[e]) <= (length[b]):
                    marker_tot= np.append(marker_tot, marker_1[e])

            #print(marker_tot)

            # counting and adding to the correct position within the data frame
            null_amount=data_patient[marker_tot].notnull().sum(axis = 1)
            bloodmarkers_times_measured_3.iloc[i,b]=(null_amount)[i]

# merging to the final data frame
df=pd.merge(df, bloodmarkers_times_measured_7, how= "right", on="patient_number")
df=pd.merge(df, bloodmarkers_times_measured_3, how= "right", on="patient_number")

# ABNORMAL, NORMAL OR MISSING
abnormal_bloodmarker_value_max=[]
abnormal_bloodmarker_value_min=[]


filtered_bloodmarkers_final = pd.DataFrame(index=[data["patient_number"]])

for b in range(len(columns)):
    # make a list with all the columns of that specific serological marker (b selects a specific serological marker, first position within the dictionary)
    marker_1 = [x for name in names for x in data.columns if x.startswith(name_dictionary[name][b])]

    # need to add in the marker tot list the patient number, use this to join to the new matrix
    marker_tot=["patient_number"]

    # filter for the correct length
    for e in range(len(marker_1)):
        if len(marker_1[e]) <= (length[b]):
            marker_tot= np.append(marker_tot, marker_1[e])

    filtered_bloodmarkers = pd.DataFrame(index=[data["patient_number"]])
    # adding the specific columns to the data frame

    filtered_bloodmarkers= pd.merge(data[marker_tot], filtered_bloodmarkers, how= "right", on="patient_number")
    # this data frame only has the columns of this specific blood markers, after creating and filtering we will add it to the final bloodmarker table
    # data[marker_tot] contains all serological marker columns of that sepcific serological marker

    for c in range(len(filtered_bloodmarkers)):
     #--> allows us to do it for every patient, and later for every column
     # iterating through all the patients
     for d in range(1,len(filtered_bloodmarkers.columns)):
        # iterating through all the columns that we have just added
        if filtered_bloodmarkers.iloc[c,d] > abnormal_bloodmarker_value_max[b]:
             filtered_bloodmarkers.iloc[c,d]= np.where(filtered_bloodmarkers.iloc[c,d] > abnormal_bloodmarker_value_max[b], 1, filtered_bloodmarkers.iloc[c,d])

        elif filtered_bloodmarkers.iloc[c,d] < abnormal_bloodmarker_value_min[b]:
             filtered_bloodmarkers.iloc[c,d] = np.where(filtered_bloodmarkers.iloc[c,d] < abnormal_bloodmarker_value_min[b], 1, filtered_bloodmarkers.iloc[c,d])

        elif filtered_bloodmarkers.iloc[c,d] >= abnormal_bloodmarker_value_min[b] and filtered_bloodmarkers.iloc[c,d] <= abnormal_bloodmarker_value_max[b]:
            filtered_bloodmarkers.iloc[c,d] = np.where(filtered_bloodmarkers.iloc[c,d] >= abnormal_bloodmarker_value_min[b] and filtered_bloodmarkers.iloc[c,d] <= abnormal_bloodmarker_value_max[b], 0, filtered_bloodmarkers.iloc[c,d])

        else:
            filtered_bloodmarkers.iloc[c,d]= -1


    filtered_bloodmarkers_final= pd.merge(filtered_bloodmarkers, filtered_bloodmarkers_final, how= "right", on="patient_number")

df=pd.merge(df, filtered_bloodmarkers_final, how= "right", on="patient_number")

df_average= pd.read_csv(r'', encoding='latin-1', low_memory=False)

df_mean= df_average[["patient_number", 'erythrocytes_mean',
       'hemoglobin_mean', 'hematocrit_mean', 'MCHC_mean', 'MCV_mean',
       'thrombocytes_mean', 'leucocytes_mean',
       'hemoglobin_per_erythrocyte_mean', 'alkaline_phosphatase_mean',
       'ASAT_mean', 'ALAT_mean', 'total_bilirubin_mean', 'gamma_GT_mean',
       'lactate_dehydrogenase_mean', 'calcium_mean', 'creatin_mean',
       'total_proteins_mean', 'blood_urea_nitrogen_mean', 'potassium_mean',
       'sodium_mean', 'cholinesterase_mean', 'amylase_mean', 'lipase_mean',
       'glucose_mean', 'INR_mean', 'partial_thromboplasmin_time_mean',
       'CRP_mean', 'quick_test_mean']].copy()


for c in range(len(df_mean)): #--> allows us to do it for every patient, and later for every column
     # iterating through all the patients
    print(c)
    for d in range(1,len(df_mean.columns)):
        # iterating through all the columns that we have just added
        if df_mean.iloc[c,d] > abnormal_bloodmarker_value_max[b]:
             df_mean.iloc[c,d]= np.where(df_mean.iloc[c,d] > abnormal_bloodmarker_value_max[b], 1, df_mean.iloc[c,d])

        elif df_mean.iloc[c,d] < abnormal_bloodmarker_value_min[b]:
             df_mean.iloc[c,d] = np.where(df_mean.iloc[c,d] < abnormal_bloodmarker_value_min[b], 1, df_mean.iloc[c,d])

        elif df_mean.iloc[c,d] >= abnormal_bloodmarker_value_min[b] and df_mean.iloc[c,d] <= abnormal_bloodmarker_value_max[b]:
            df_mean.iloc[c,d] = np.where(df_mean.iloc[c,d] >= abnormal_bloodmarker_value_min[b] and df_mean.iloc[c,d] <= abnormal_bloodmarker_value_max[b], 0, df_mean.iloc[c,d])

        else:
            df_mean.iloc[c,d]= -1
df=pd.merge(df, df_mean, how= "right", on="patient_number")

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

#df.to_csv('')

