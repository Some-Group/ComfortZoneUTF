from traceback import extract_tb

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import neurokit2 as nk
import os
import warnings

# สวัสดี ทดลอง github จ้าา


def ProcessingHrvFreq(ppg_df, window_size, sampling_rate):
    # จำนวนขนาดเวลา(วินาที) ที่ขยับ shift เพื่อคำนวณค่า hrv freq ต่าง ๆ
    window_samples = window_size * sampling_rate
    if window_size == 300:
        ppg_df["NewLFHF"] = []

    try:
        if device_name == "EmotiBit" :
            ppg_df["Timestamp"] = pd.to_datetime(ppg_df["LocalTimestamp"], unit="s").dt.tz_localize('UTC').dt.tz_convert("Asia/Bangkok")
            ppg_df.drop(columns=["LocalTimestamp"], inplace=True)

            ppg_cleaned = nk.ppg_clean(ppg_df["PI"], sampling_rate=sampling_rate)

            hrv_results = []
            for start in range(0, len(ppg_cleaned) - window_samples, sampling_rate):
                end = start + window_samples
                window = ppg_cleaned[start:end]
                start_time = ppg_df['Timestamp'].iloc[start]
                if start % 30 == 0:
                    print(f"Start Time: {start_time} - End Time: {ppg_df['Timestamp'].iloc[end]}")

                # Filter out specific warning by message substring
                warnings.filterwarnings("ignore", message=".*DFA_alpha2.*")
                warnings.filterwarnings("ignore", message=".*mse = np.trapz(mse)*")

                ppg_processed, info = nk.ppg_process(window, sampling_rate=sampling_rate)
                signals, info = nk.ppg_peaks(ppg_processed["PPG_Clean"], sampling_rate=sampling_rate)

                hrv_metrics = nk.hrv(signals, sampling_rate=sampling_rate)
                hrv_metrics['Timestamp'] = start_time
                hrv_results.append(hrv_metrics)
                # if window_size == 300:
                #     LFHFvalue = float(ppg_df['HRV_LF'] / ppg_df['HRV_HF'])
                #     ppg_df["NewLFHF"].append(LFHFvalue)
            hrv_df = pd.concat(hrv_results, ignore_index=True)
            hrv_df['Timestamp'] = pd.to_datetime(hrv_df['Timestamp'])
            hrv_df = hrv_df.set_index('Timestamp')

            # extracted_df = hrv_df.resample('1s').max()
            if window_size == 60:
                extracted_df = hrv_df.resample('1s').max()
            elif window_size == 120:
                extracted_df = hrv_df.resample('2s').max()
            elif window_size == 300:
                extracted_df = hrv_df.resample('5s').max()

            extracted_df.reset_index(inplace=True)
            extracted_df['HR'] = 60 / (extracted_df['HRV_MeanNN'] / 1000)

        elif device_name == "MAX30102":
            window_samples = window_size * sampling_rate
            ppg_df["Timestamp"] = pd.to_datetime(ppg_df["Timestamp"], dayfirst=True)

            ppg_cleaned = nk.ppg_clean(ppg_df["IR"], sampling_rate=sampling_rate)
            hrv_results = []

            for start in range(0, len(ppg_cleaned) - window_samples, sampling_rate):
                end = start + window_samples
                window = ppg_cleaned[start:end]
                start_time = ppg_df['Timestamp'].iloc[start]

                if start % 30 == 0:
                    print(f"Start Time: {start_time} - End Time: {ppg_df['Timestamp'].iloc[end]}")

                # ปิด warning บางประเภทที่ไม่สำคัญ
                warnings.filterwarnings("ignore", message=".*DFA_alpha2.*")
                warnings.filterwarnings("ignore", message=".*mse = np.trapz(mse)*")
                warnings.filterwarnings("ignore", message=".*invalid value encountered in scalar divide.*")

                # try:
                ppg_processed, info = nk.ppg_process(window, sampling_rate=sampling_rate)
                signals, info = nk.ppg_peaks(ppg_processed["PPG_Clean"], sampling_rate=sampling_rate)

                hrv_metrics = nk.hrv(signals, sampling_rate=sampling_rate)
                # except Exception as e:
                #     print(f"HRV calc failed at {start_time}: {e}")
                #     # สร้างแถวว่างที่มี Timestamp อย่างเดียว
                #     hrv_metrics = pd.DataFrame(columns=nk.hrv(signals.iloc[:0], sampling_rate=sampling_rate).columns)
                #     hrv_metrics.loc[0] = [np.nan] * len(hrv_metrics.columns)
                hrv_metrics['Timestamp'] = start_time
                hrv_results.append(hrv_metrics)

                # if window_size == 300:
                #     LFHFvalue = float(ppg_df['HRV_LF'] / ppg_df['HRV_HF'])
                #     ppg_df["NewLFHF"].append(LFHFvalue)



            hrv_df = pd.concat(hrv_results, ignore_index=True)
            hrv_df['Timestamp'] = pd.to_datetime(hrv_df['Timestamp'])
            hrv_df = hrv_df.set_index('Timestamp')


            # extracted_df = hrv_df.interpolate(method='time')
            if window_size == 60:
                extracted_df = hrv_df.resample('1s').max()
            elif window_size == 120:
                extracted_df = hrv_df.resample('2s').max()
            elif window_size == 300:
                extracted_df = hrv_df.resample('5s').max()

            # extracted_df = hrv_df.reset_index()
            extracted_df.reset_index(inplace=True)
            # hrv_df = hrv_df.set_index('Timestamp')
            # if 'HRV_MeanNN' in hrv_df.columns:
            extracted_df['HR'] = 60 / (extracted_df['HRV_MeanNN'] / 1000)
            # else:
            #     print("Warning: 'HRV_MeanNN' not found. Skipping HR calculation.")

        else:

            print("Device Not Supported")

            return None

    except Exception as e:
        if csv_time < 1:
            print("\nInsufficient time for HF calculation, data length must be at least 1 minutes.\n")
            print(str(e))
        elif csv_time < 2:
            print("\nInsufficient time for LF calculation, data length must be at least 2 minutes.\n")
            print(str(e))
        elif csv_time < 5:
            print("\nInsufficient time for LF/HF calculation, data length must be at least 5 minutes.\n")
            print(str(e))
        else:
            print(f"Can't result to convert Timestamp! Error: {str(e)}")

    return extracted_df


# ---- การทำงานหลัก ----
pathfile = input("Enter the path of file: ").strip('"')
pathfile = pathfile.strip("'")
df = pd.read_csv(rf"{pathfile}")

sampling_rate = 100

# เช็คชื่ออุปกรณ์ที่ฝช้วัดจากคอลัมน์ในไฟล์
try:
    if 'PI' in df.columns:
        device_name = "EmotiBit"
        color = 'green'
        signal_column = "PI"
    elif 'IR' in df.columns:
        device_name = "MAX30102"
        color = 'red'
        signal_column = "IR"
except:
    print(f"ไม่พบคอลัมน์ PI หรือ IR ในไฟล์ csv นี้ {pathfile}")


csv_time = len(df) // 6000

print(f"This csv file has a recording length of ~{csv_time} miniutes. Recorded from {device_name}.")

subject_name = input("\nEnter the subject name: ")

# nowTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# print(hrvData[['Timestamp','HRV_LF','HRV_HF','HRV_LFHF']])
# สำหรับ lf (window_size=120) ค่่าอื่นๆ ก็แปลผันไปตามค่าความเหมาะสมของเวลลาที่ใช้ในการคำนวณ
# now = datetime.now().strftime("%Y%m%d_%H%M%S")
# path = r"D:\1-BukAILab\01-MAX30102\PythonForCSVrecordingMAX30102\01-hrv_freq_result"
# filename = glob.glob(pathfile + "/*.csv")
filename = os.path.splitext(os.path.basename(pathfile))[0]
if device_name == "MAX30102":
    filename = "IR-"+ filename
    path = r"D:\1-BukAILab\01-MAX30102\PythonForCSVrecordingMAX30102\01-hrv_freq_result\MAX30102"
elif device_name == "EmotiBit":
    filename = "PI-"+ filename
    path = r"D:\1-BukAILab\01-MAX30102\PythonForCSVrecordingMAX30102\01-hrv_freq_result\EMOTIBIT"

# HF
print("\nNext Processing in HF")

hrvHFData = ProcessingHrvFreq(ppg_df=df, window_size=60, sampling_rate=sampling_rate)
HFoutput_filename = f"hrv_HF_{device_name.upper()}_{filename}"
hrvHFData.to_csv(f"{path}/{HFoutput_filename}.csv")
print("Finished Processing HF signal")


df = pd.read_csv(rf"{pathfile}")

# LF
print("\nNext Processing in LF")

hrvLFData = ProcessingHrvFreq(ppg_df=df, window_size=120, sampling_rate=sampling_rate)
LFoutput_filename = f"hrv_LF_{device_name.upper()}_{filename}"
hrvLFData.to_csv(f"{path}/{LFoutput_filename}.csv")
print("Finished Processing HF signal")
# รอเติมเลขวันที่เวลา ในชื่อไฟล์


# LFHF
print("\nNext Processing in Ratio LF/HF")
df = pd.read_csv(rf"{pathfile}")
hrvRatioLFHFData = ProcessingHrvFreq(ppg_df=df, window_size=300, sampling_rate=sampling_rate)
RatioLFHFoutput_filename = f"hrv_RatioLFHF_{device_name.upper()}_{filename}"
hrvRatioLFHFData.to_csv(f"{path}/{RatioLFHFoutput_filename}.csv")
print("Finished Processing Ratio LF/HF signal")
# hrvLFHFData.to_csv('LFHFt.csv')


print("\nNext Ploting Graph")
plt.figure(figsize=(16, 8))
plt.plot(hrvHFData["Timestamp"], hrvHFData["HRV_HF"], label="HF (1-min)", alpha=0.7)
plt.plot(hrvLFData["Timestamp"], hrvLFData["HRV_LF"], label="LF (2-min)", alpha=0.7)
if device_name == "MAX30102":
    plt.plot(hrvRatioLFHFData["Timestamp"], hrvRatioLFHFData["NewLFHF"], label="LF/HF Ratio (5-min)", alpha=0.7)
else:
    plt.plot(hrvRatioLFHFData["Timestamp"], hrvRatioLFHFData["HRV_LFHF"], label="LF/HF Ratio (5-min)", alpha=0.7)

plt.title(f"{subject_name}'s HRV Frequency per Segment - {device_name} - {filename}")
plt.xlabel("Time (%M:%S)")
plt.ylabel("Power (ms^2) / Scaled Ratio")
plt.xticks(rotation=45)  # หมุน timestamp ให้อ่านง่าย
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#yrdyfyfyfdydfyfyffdgfg
