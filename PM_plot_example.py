import pandas as pd

# Create a dictionary with the data
data = {
    "return": [0.003421, 0.000508, -0.003321, 0.006753, -0.000416],
    "cost": [0, 0, 0, 0, 0],
    # "cost": [0.000864, 0.000447, 0.000212, 0.000212, 0.000440],
    "bench": [0.011693, 0.000721, -0.004322, 0.006874, -0.003350],
    # "turnover": [0.576325, 0.227882, 0.102765, 0.105864, 0.208396],
    "turnover": [0, 0, 0, 0, 0],
}

# Create a list with the dates
dates = ["2017-01-04", "2017-01-05", "2017-01-06", "2017-01-09", "2017-01-10"]

# Convert the dates to datetime objects
dates = pd.to_datetime(dates)

# Create the DataFrame
df = pd.DataFrame(data, index=dates)

# Set the name of the index
df.index.name = 'date'

print(df)

import qlib
from qlib.contrib.report.analysis_position.report import report_graph
report_graph(df, show_notebook=True)
