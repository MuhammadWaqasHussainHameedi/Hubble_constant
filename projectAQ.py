import numpy as np
from astroquery.sdss import SDSS
from astropy.table import Table
from uncertainties import ufloat # for error incountering
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt
from uncertainties import UFloat
#Access 50 celestial bodies red shifts redshifts from SDSS
query = "SELECT TOP 50 ra, dec, z FROM SpecObj WHERE z > 0"
results = SDSS.query_sql(query)

# Convert results to a table
data = Table(results).to_pandas()#.to_pandas for converting into pandas
pd.set_option('display.max_columns',None)#to show all columns
pd.set_option('display.max_rows',None)#to show all rows
pd.set_option('display.width',None)#for not reduced values
pd.set_option('display.max_colwidth', None) #for column width
print('sdss data of 50 celestial bodies')
print(data)
redshift = data['z']
c=300000#km/s
#give different H. from Nasa
# Hubble constant values (ufloat where applicable)
H_values = {
    "H1 (Cephied+SNLA 2021)": ufloat(73.04, 1.04),
    "H2 (SBF distances)": ufloat(73.3, 2.5),
    "H3 (eBOSS)": ufloat(67.35, 0.97),
    "H4 (Megamasers)": ufloat(73.9, 3),
    "H5 (TRGB)": ufloat(69.8, 1.9),
    "H6 (GravLens)": ufloat(73.3, 1.8),
    "H7 (XMM)": ufloat(67, 3),
    "H8 (Planck)": ufloat(67.66, 0.42),
    "H9 (INV)": ufloat(67.3, 1.1),
    "H10 (Cephied+SNLA 2011)": ufloat(73.8, 2.4),
    "H11 (HST)": ufloat(72, 8),
    "H12 (LIGO+Virgo+Kagra)": [80, 61],
    "H13 (Chandra+ISZ2006)": [83, 66.5],
    "H14 (Planck+ACTpol)": ufloat(68.7, 1.3),
    "H15 (SPT)": ufloat(68.8, 1.5),
    "H16 (ACTpol)": ufloat(67.9, 1.5),
    "H17 (Planck PR3)": ufloat(67.36, 0.54),
    "H18 (SPTpol)": ufloat(71.2, 2.1),
    "H19 (WMAP9)": ufloat(70, 2.2),
    "H20 (SPT2013)": ufloat(75, 3.5),
}

# Calculate velocities and distances for each Hubble constant
results_dict = {} #used to store calculated result For each value of h
for key, H in H_values.items():
    V = c * redshift
    if isinstance(H, list):  # Handle list of H values like if value in list
        distances = [V / h for h in H]
        D = np.stack(distances, axis=-1)  # Combine results . np array combines result of muliple array into single
        df = pd.DataFrame({'RA': data['ra'], 'DEC': data['dec'], 'Velocities (km/s)': V})
        for idx, h_val in enumerate(H):
            df[f'Distances (Mpc) for H={h_val}'] = D[:, idx]
    else:
        D = V / H
        df = pd.DataFrame({'RA': data['ra'], 'DEC': data['dec'], 'Velocities (km/s)': V, f'Distances (Mpc) for H ={H}': D})
    results_dict[key] = df # for each key the value of df (data frame)
    print(f"\nResults for {key}:\n")
    print(df)

#for comparing distances
combined_df = pd.DataFrame({'RA': data['ra'], 'DEC': data['dec']})

# Add all the distance columns from each Hubble constant
for key, df in results_dict.items():
    for col in df.columns:
        if col.startswith('Distances (Mpc)'):
            combined_df[col] = df[col]
"""print(combined_df)
print("\nShape of the combined dataframe:", combined_df.shape)"""

#for age
# Fixed matter density parameter for a flat universe
Omega_m = 0.3

# Calculate the age of the universe for each Hubble constant
ages = {}
for key, H in H_values.items():
    if isinstance(H, list):  # Handle lists of H values
        ages[key] = [FlatLambdaCDM(H0=h, Om0=Omega_m).age(0).value for h in H]
    else:
        cosmology = FlatLambdaCDM(H0=H.nominal_value, Om0=Omega_m)
        age = cosmology.age(0).value  # Age at z=0 in Gyr
        ages[key] = ufloat(age, H.std_dev)

# Convert results to a dataframe
age_df = pd.DataFrame([
    {"Hubble Source": key, "Hubble Constant (km/s/Mpc)": str(H), "Age of Universe (Gyr =10^9 yr)": ages[key]}
    for key, H in H_values.items()
])

# Print the results
print(age_df)

age_values = age_df["Age of Universe (Gyr =10^9 yr)"].apply(
    lambda x: x.nominal_value if isinstance(x, UFloat) else sum(x) / len(x)
)
age_errors = age_df["Age of Universe (Gyr =10^9 yr)"].apply(
    lambda x: x.std_dev if isinstance(x, UFloat) else 0
)


# Plot the ages with error bars
plt.figure(figsize=(15, 8))
plt.bar(age_df["Hubble Source"], age_values, yerr=age_errors, capsize=5, color='teal', alpha=0.7)

plt.xlabel('Hubble Sources')
plt.ylabel('Age of Universe (Gyr)')
plt.title('Age of Universe for Various Hubble Constants')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
# Convert all distance columns to nominal values
distance_columns = [col for col in combined_df.columns if col.startswith("Distances")]
for col in distance_columns:
    combined_df[col] = combined_df[col].apply(
        lambda x: x.nominal_value if isinstance(x, UFloat) else x
    )

# Extract RA for labels (all 50 bodies)
ra_all = combined_df['RA']

# Set up the figure
plt.figure(figsize=(20, 10))

# Calculate the bar width to separate bars for each Hubble constant
num_constants = len(distance_columns)
bar_width = 1 / (num_constants + 1)  # Adjust based on the number of constants
indices = np.arange(len(ra_all))  # Indices for RA positions

# Iterate over Hubble constant distances and plot vertical bars
for i, col in enumerate(distance_columns):
    plt.bar(
        indices + i * bar_width,  # Offset bars for each constant
        combined_df[col],
        width=bar_width,
        label=col,
        alpha=0.7
    )

# Formatting the plot
plt.xlabel('Right Ascension (RA)', fontsize=12)
plt.ylabel('Distance (Mpc)', fontsize=12)
plt.title('Distances of 50 Celestial Bodies for Various Hubble Constants', fontsize=16)
plt.xticks(indices + (num_constants - 1) * bar_width / 2, ra_all, rotation=45, fontsize=8)  # Center ticks
plt.legend(loc='upper right', fontsize=8, bbox_to_anchor=(1.15, 1))
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

# Show the plot
plt.show()