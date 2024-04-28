
#Simple loop over columns of object type to fill with mode

cc_apps_imputed = cc_apps_nans_replaced.copy()


# Columns with object data type
object_columns = cc_apps_imputed.select_dtypes(include=['object']).columns

# Step 2 & 3: Fill missing values with the most frequent value for each column
for column in object_columns:
    most_frequent_value = cc_apps_imputed[column].mode()[0]
    cc_apps_imputed[column] = cc_apps_imputed[column].fillna(most_frequent_value)