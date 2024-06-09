import pandas as pd

def append_to_csv_pandas(filename, data_dict):
  """
  Appends a row from a dictionary to a CSV file using pandas.

  Args:
      filename: The name of the CSV file.
      data_dict: A dictionary containing the data for the new row.
  """
  # Create a DataFrame from the dictionary
  df = pd.DataFrame([data_dict])

  # Read the existing CSV (or create an empty DataFrame if it doesn't exist)
  try:
    existing_df = pd.read_csv(filename)
  except FileNotFoundError:
    existing_df = pd.DataFrame()

  # Append the new data to the existing DataFrame
  df_combined = pd.concat([existing_df, df], ignore_index=True)

  # Save the combined DataFrame to the CSV file with index=False to avoid index column
  df_combined.to_csv(filename, index=False)