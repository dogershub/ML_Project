import numpy as np
import pandas as pd

def impute_biometrics(main_df, result_df, col_to_impute, col_impute_val):
    """
    Function to fill in missing values in height and weight features
    """
    try:
        # Create a dictionary mapping (Sex, AgeCategory) to the corresponding imputation value
        impute_dict = {(row["Sex"], row["AgeCategory"]): row[col_impute_val] for _, row in result_df.iterrows()}

        # Define a function to apply imputation based on (Sex, AgeCategory)
        def impute_func(row):
            key = (row["Sex"], row["AgeCategory"])
            return impute_dict.get(key, row[col_to_impute])

        # Apply imputation function to fill missing values
        main_df[col_to_impute] = main_df.apply(lambda row: impute_func(row) if pd.isnull(row[col_to_impute]) else row[col_to_impute], axis=1)

    except Exception as err:
        print(f"ERROR: {err}")

    return main_df


def impute_bmi(df):
    """
    Function to fill in missing values in BMI column
    """
    try:
        # Assuming 'BMI', 'WeightInKilograms', and 'HeightInMeters' are columns in the DataFrame
        df['BMI'] = np.where(
            (df['BMI'].isnull()) & (df['WeightInKilograms'].notnull()) & (df['HeightInMeters'].notnull()),
            df['WeightInKilograms'] / (df['HeightInMeters']**2),
            df['BMI']
        )
    except Exception as err:
        print(f"ERROR: {err}")

    return df


def impute_based_on_sws(main_df, result_df, col_to_imputate):
    """
    Function to fill in missing values using window partitioning for sex and weight status
    """
    try:
        result_dict = result_df.to_dict(orient='records')

        for i in range(4):
            main_df[col_to_imputate] = main_df.apply(lambda row: result_dict[i][col_to_imputate]
                                                      if pd.isnull(row[col_to_imputate]) and
                                                      row['Sex'] == result_dict[i]['Sex'] and
                                                      row['WeightStatus'] == result_dict[i]['WeightStatus']
                                                      else row[col_to_imputate], axis=1)

        for i in range(4, 8):
            main_df[col_to_imputate] = main_df.apply(lambda row: result_dict[i][col_to_imputate]
                                                      if pd.isnull(row[col_to_imputate]) and
                                                      row['Sex'] == result_dict[i]['Sex'] and
                                                      row['WeightStatus'] == result_dict[i]['WeightStatus']
                                                      else row[col_to_imputate], axis=1)
    except Exception as err:
        print(f"ERROR: {err}")

    return main_df


def get_statistical_analysis(df, col1_, col2_, col3_):
    """
    Function to display each group having the maximum record count
    """
    try:
        result = df.groupby([col1_, col2_, col3_]).size().reset_index(name='count')
        result['max_count'] = result.groupby([col1_, col2_])['count'].transform('max')
        result = result[result['count'] == result['max_count']].drop('max_count', axis=1)
    except Exception as err:
        print(f"ERROR: {err}")
        result = pd.DataFrame()  # Return an empty DataFrame in case of an error

    return result

    
def impute_smoking_status(main_df):
    """
    Function to fill in missing values in smoker status based on the grouping with sex and age category
    """
    try:
        conditions = [
            (main_df["Sex"] == "Female") & (main_df["SmokerStatus"].isnull()),
            (main_df["Sex"] == "Male") & (main_df["SmokerStatus"].isnull()) & (main_df["AgeCategory"].isin(["Age 75 to 79", "Age 80 or older"])),
            (main_df["Sex"] == "Male") & (main_df["SmokerStatus"].isnull()) & ~(main_df["AgeCategory"].isin(["Age 75 to 79", "Age 80 or older"]))
        ]

        choices = ["Never smoked", "Former smoker", "Never smoker"]

        main_df["SmokerStatus"] = np.select(conditions, choices, default=main_df["SmokerStatus"])
    except Exception as err:
        print(f"ERROR: {err}")

    return main_df


def impute_alcohol_drinker(main_df):
    """
    Function to fill in missing values in alcohol drinker column based on the grouping with sex and age category
    """
    try:
        conditions = [
            (main_df["Sex"] == "Female") & (main_df["AlcoholDrinkers"].isnull()) & (main_df["AgeCategory"].isin(["Age 65 to 69", "Age 70 to 74", "Age 75 to 79", "Age 80 or older"])),
            (main_df["Sex"] == "Female") & (main_df["AlcoholDrinkers"].isnull()) & ~(main_df["AgeCategory"].isin(["Age 65 to 69", "Age 70 to 74", "Age 75 to 79", "Age 80 or older"])),
            (main_df["Sex"] == "Male") & (main_df["AlcoholDrinkers"].isnull()) & (main_df["AgeCategory"].isin(["Age 80 or older"])),
            (main_df["Sex"] == "Male") & (main_df["AlcoholDrinkers"].isnull()) & ~(main_df["AgeCategory"].isin(["Age 80 or older"]))
        ]

        choices = ["No", "Yes", "No", "Yes"]

        main_df["AlcoholDrinkers"] = np.select(conditions, choices, default=main_df["AlcoholDrinkers"])
    except Exception as err:
        print(f"ERROR: {err}")

    return main_df


def impute_sleep_hours(main_df):
    """
    Function to fill in missing values in sleep hours column based on the grouping with sex and age category
    """
    try:
        conditions = [
            (main_df["SleepHours"].isnull()) & (main_df["AgeCategory"].isin(["Age 65 to 69", "Age 70 to 74", "Age 75 to 79", "Age 80 or older"])),
            (main_df["SleepHours"].isnull()) & ~(main_df["AgeCategory"].isin(["Age 65 to 69", "Age 70 to 74", "Age 75 to 79", "Age 80 or older"]))
        ]

        main_df["SleepHours"] = np.select(conditions, [8, 7], default=main_df["SleepHours"])
    except Exception as err:
        print(f"ERROR: {err}")

    return main_df


