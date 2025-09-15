# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 15/09/2025



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

file_path = "Coffe_sales.xlsx"      
data = pd.read_excel(file_path, sheet_name="index_1")

daily_sales = (
    data.groupby("date")["money"]
    .sum()
    .sort_index()
)

plt.rcParams["figure.figsize"] = [12, 6]
plt.plot(daily_sales)
plt.title("Original Daily Coffee Sales")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plot_acf(daily_sales, lags=len(daily_sales)//4, ax=plt.gca())
plt.title("Daily Sales ACF")
plt.subplot(2, 1, 2)
plot_pacf(daily_sales, lags=len(daily_sales)//4, ax=plt.gca())
plt.title("Daily Sales PACF")
plt.tight_layout()
plt.show()

arma11_model = ARIMA(daily_sales, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params["ar.L1"]
theta1_arma11 = arma11_model.params["ma.L1"]

N = 1000
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title("Simulated ARMA(1,1) Process")
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_1)
plt.title("Simulated ARMA(1,1) ACF")
plt.show()

plot_pacf(ARMA_1)
plt.title("Simulated ARMA(1,1) PACF")
plt.show()

arma22_model = ARIMA(daily_sales, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params["ar.L1"]
phi2_arma22 = arma22_model.params["ar.L2"]
theta1_arma22 = arma22_model.params["ma.L1"]
theta2_arma22 = arma22_model.params["ma.L2"]

ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N * 10)

plt.plot(ARMA_2)
plt.title("Simulated ARMA(2,2) Process")
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_2)
plt.title("Simulated ARMA(2,2) ACF")
plt.show()

plot_pacf(ARMA_2)
plt.title("Simulated ARMA(2,2) PACF")
plt.show()

```

### OUTPUT:


<img width="1005" height="547" alt="TS EXP4 PIC1" src="https://github.com/user-attachments/assets/8c8930df-b855-4b8f-a690-a9d53a5cbf5b" />

<img width="1198" height="590" alt="TS EXP4 PIC2" src="https://github.com/user-attachments/assets/34945053-d8cd-4ad6-87ce-725ffdef8283" />
SIMULATED ARMA(1,1) PROCESS:
<img width="993" height="528" alt="TS EXP4 PIC 3" src="https://github.com/user-attachments/assets/facc4bb8-e26c-4318-ac8a-9bf161c4745b" />

Partial Autocorrelation
<img width="1002" height="528" alt="TS EXP4 PIC5" src="https://github.com/user-attachments/assets/1f783e9b-ee60-470c-b046-1bc45d759653" />


Autocorrelation
<img width="1002" height="528" alt="TS EXP4 PIC4" src="https://github.com/user-attachments/assets/9ad37329-68fb-41cc-8202-bec7e51ac18e" />





SIMULATED ARMA(2,2) PROCESS:

<img width="993" height="528" alt="TS EXP4 PIC6" src="https://github.com/user-attachments/assets/d221f8ec-afc8-4248-90b6-e64dae66e757" />


Autocorrelation

<img width="1002" height="528" alt="TS EXP4 PIC7" src="https://github.com/user-attachments/assets/894fb005-48e7-47d9-9dfa-1eba1dbf5cb6" />


Partial Autocorrelation
<img width="1002" height="528" alt="TS EXP4 PIC8" src="https://github.com/user-attachments/assets/3169d229-5858-4088-a25c-9b843c41892d" />


RESULT:
Thus, a python program is created to fir ARMA Model successfully.
