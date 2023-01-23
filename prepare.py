import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# Helper Functions:

def blood_type_seperation(new_data):
    group0_bool = new_data["blood_type"].isin(["O+", "O-"])
    group1_bool = new_data["blood_type"].isin(["B+", "B-", "AB+", "AB-"])
    group2_bool = new_data["blood_type"].isin(["A+", "A-"])

    new_blood_columns = [[], [], []]
    for i in range(0, len(group0_bool)):
        if np.array(group0_bool)[i]:
            new_blood_columns[0].append(1)
        else:
            new_blood_columns[0].append(0)

        if np.array(group1_bool)[i]:
            new_blood_columns[1].append(1)
        else:
            new_blood_columns[1].append(0)

        if np.array(group2_bool)[i]:
            new_blood_columns[2].append(1)
        else:
            new_blood_columns[2].append(0)

    new_data = new_data.drop('blood_type', inplace=False, axis=1)
    new_data["blood_type_group_zero"] = new_blood_columns[0]
    new_data["blood_type_group_one"] = new_blood_columns[1]
    new_data["blood_type_group_two"] = new_blood_columns[2]
    return new_data


def symptoms_correction(new_data):
    str_list = [string for string in new_data["symptoms"].unique() if not pd.isnull(string)]
    symptoms_list = []
    symptoms_columns = []

    for string in str_list:
        current_symptoms = string.split(';')
        for symptom in current_symptoms:
            if (not (symptom in symptoms_list)):
                symptoms_list.append(symptom)
                symptoms_columns.append([])

    symptoms_array = np.array(new_data["symptoms"])
    for i in range(new_data.shape[0]):
        j = 0
        for symptom in symptoms_list:
            symptom_exists = 0
            if not pd.isnull(symptoms_array[i]) and symptom in symptoms_array[i]:
                symptom_exists = 1
            symptoms_columns[j].append(symptom_exists)
            j += 1

    new_data = new_data.drop('symptoms', inplace=False, axis=1)
    j = 0
    for symptom in symptoms_list:
        new_data[symptom] = symptoms_columns[j]
        j += 1

    return new_data


def columns_correction_to_ordinal(new_data):
    male_on_index = np.array(new_data["sex"].isin(["M"]))
    location_list = np.array(new_data["current_location"])
    date_list = np.array(new_data["pcr_date"])

    sex_column = []
    location_x_column = []
    location_y_column = []
    day_list = []
    month_list = []
    year_list = []

    for i in range(new_data.shape[0]):
        binary_sex = 0
        if male_on_index[i]:
            binary_sex = 1
        sex_column.append(binary_sex)

        current_location = location_list[i].strip("(')").replace("'", "").replace(" ", "").split(',')
        location_x_column.append(float(current_location[0]))
        location_y_column.append(float(current_location[1]))

        current_date = date_list[i].split('-')
        year_list.append(int(current_date[0]))
        month_list.append(int(current_date[1]))
        day_list.append(int(current_date[2]))

    new_data = new_data.drop('sex', inplace=False, axis=1).drop('current_location', inplace=False, axis=1).drop(
        'pcr_date', inplace=False, axis=1)
    new_data["sex"] = sex_column
    new_data["x_location"] = location_x_column
    new_data["y_location"] = location_y_column
    new_data["pcr_day"] = day_list
    new_data["pcr_month"] = month_list
    new_data["pcr_year"] = year_list

    return new_data


def prepare_data(training_data, new_data):
    new_data_copy = new_data.copy()
    training_data_copy = training_data.copy()

    new_data_copy = blood_type_seperation(new_data_copy)
    new_data_copy = symptoms_correction(new_data_copy)
    new_data_copy = columns_correction_to_ordinal(new_data_copy)
    training_data_copy = blood_type_seperation(training_data_copy)
    training_data_copy = symptoms_correction(training_data_copy)
    training_data_copy = columns_correction_to_ordinal(training_data_copy)

    new_data_copy.reset_index()

    minmax_features_names = ["patient_id", "PCR_01", "PCR_02", "PCR_03", "PCR_04", "PCR_05", "PCR_09", "pcr_day",
                             "pcr_month", "pcr_year"]
    scalar_features_names = ["age", "weight", "num_of_siblings", "happiness_score", "household_income",
                             "conversations_per_day", "sugar_levels", "sport_activity", "PCR_06", "PCR_07", "PCR_08",
                             "PCR_09", "PCR_10", "x_location", "y_location"]
    data_normalized = pd.DataFrame(new_data_copy,
                                   columns=["cough", "fever", "shortness_of_breath", "low_appetite", "sore_throat",
                                            "sex", "blood_type_group_zero", "blood_type_group_one",
                                            "blood_type_group_two", "spread", "risk"])

    minmax = MinMaxScaler((-1, 1))

    for name in minmax_features_names:
        minmax.fit(pd.DataFrame(training_data_copy, columns=[name]))
        data_normalized[name] = minmax.transform(pd.DataFrame(new_data_copy, columns=[name]))

    scalar = StandardScaler()

    for name in scalar_features_names:
        scalar.fit(pd.DataFrame(training_data_copy, columns=[name]))
        data_normalized[name] = scalar.transform(pd.DataFrame(new_data_copy, columns=[name]))

    return data_normalized