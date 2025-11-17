import pandas as pd
import os
import re

# Read Excel
file_path = "data.xlsx"  # change this to your actual file path
df = pd.read_excel(file_path)

# Reshape from wide to long
df_long = df.melt(
    id_vars=["product"], 
    var_name="date", 
    value_name="quantity"
)

# Drop rows with missing quantities
df_long = df_long.dropna(subset=["quantity"]).reset_index(drop=True)

# Ensure date is proper datetime (optional: first day of each month)
df_long["date"] = pd.to_datetime(df_long["date"], format="%B %Y")

# Create output folder
output_dir = "products_csv"
os.makedirs(output_dir, exist_ok=True)

# Group by product and save each group to a CSV
for product, group in df_long.groupby("product"):
    # Clean product name for filename (remove special chars, replace spaces with _)
    safe_name = re.sub(r'[^A-Za-z0-9]+', '_', product).strip('_')
    
    file_name = f"{safe_name}.csv"
    file_path = os.path.join(output_dir, file_name)
    
    group.to_csv(file_path, index=False)
    print(f"Saved: {file_path}")
