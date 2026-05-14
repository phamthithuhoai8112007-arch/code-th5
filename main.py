import pandas as pd
from sklearn.ensemble import IsolationForest

# 1. Đọc dữ liệu từ file bạn đã tải từ Kaggle
# Lưu ý: Đảm bảo file .csv này nằm cùng thư mục với file code
try:
    df = pd.read_csv('financial_anomaly_data.csv')
    print("--- Đã tải dữ liệu thành công ---")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'financial_anomaly_data.csv'. Bạn hãy kiểm tra lại tên file nhé!")
    exit()

# 2. Tiền xử lý dữ liệu (Chuyển thời gian thành con số để máy hiểu)
df['Timestamp'] = pd.to_datetime(
    df['Timestamp'],
    format='%d-%m-%Y %H:%M'
)
df['Hour'] = df['Timestamp'].dt.hour 

# 3. Chọn các đặc trưng để phân tích (Số tiền và Giờ giao dịch)
X = df[['Amount', 'Hour']]

# 4. Sử dụng thuật toán Isolation Forest để phát hiện bất thường
# contamination=0.01 nghĩa là chúng ta nghi ngờ có 1% giao dịch là bất thường
model = IsolationForest(contamination=0.01, random_state=42)

# 5. Huấn luyện và dự đoán
df['is_anomaly'] = model.fit_predict(X)

# 6. Lọc ra các giao dịch bất thường (nhãn -1)
anomalies = df[df['is_anomaly'] == -1]

print(f"\nTìm thấy {len(anomalies)} giao dịch có dấu hiệu bất thường.")
print("\nDanh sách 5 giao dịch nghi vấn nhất:")
print(anomalies[['Timestamp', 'AccountID', 'Amount']].head())

# 7. Xuất kết quả ra file mới để báo cáo
anomalies.to_csv('ket_qua_bat_thuong.csv', index=False)
print("\n--- Đã lưu kết quả vào file 'ket_qua_bat_thuong.csv' ---")