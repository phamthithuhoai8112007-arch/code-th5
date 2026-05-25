import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

print("Đang phân tích dữ liệu và trích xuất lí do rời bỏ...")

# 1. Tải dữ liệu
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn (1).csv')

# 2. Chuẩn bị tập khách hàng đang sử dụng (để dự đoán tháng sau)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
active_customers = df[df['Churn'] == 'No'].copy()

# Lấy lại thông tin gốc để lát nữa ghi lí do cho dễ đọc
active_info = active_customers[['customerID', 'Contract', 'MonthlyCharges', 'tenure']].copy()

# Tiền xử lý dữ liệu để học
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# 3. Huấn luyện mô hình Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X, y)

# 4. Tính toán dự đoán
X_active = active_customers.drop(['customerID', 'Churn'], axis=1)
for column in X_active.select_dtypes(include=['object']).columns:
    X_active[column] = label_encoders[column].transform(X_active[column])

churn_probabilities = model.predict_proba(X_active)[:, 1]

# 5. TÌM LÍ DO RỜI BỎ CHO TỪNG NGƯỜI
reasons = []
median_monthly = df['MonthlyCharges'].median() # Cước trung bình

for idx, row in active_info.iterrows():
    # Lấy xác suất của người hiện tại
    prob = churn_probabilities[active_info.index.get_loc(idx)]
    
    if prob > 0.5: # Nếu rủi ro cao mới đi tìm lí do
        r = []
        if row['Contract'] == 'Month-to-month':
            r.append("Dùng Hợp đồng theo tháng")
        if row['MonthlyCharges'] > median_monthly:
            r.append(f"Cước cao (${row['MonthlyCharges']})")
        if row['tenure'] < 12:
            r.append("Khách hàng mới (< 1 năm)")
        
        # Gộp các lí do lại
        if not r:
            reasons.append("Chi phí tích lũy hoặc phương thức thanh toán rủi ro")
        else:
            reasons.append(" + ".join(r))
    else:
        reasons.append("An toàn")

# 6. Đóng gói kết quả
results_df = pd.DataFrame({
    'Mã khách hàng': active_info['customerID'],
    'Xác suất rời bỏ (%)': (churn_probabilities * 100).round(2),
    'Dự đoán tháng sau': ['Cảnh báo rời đi' if p > 0.5 else 'An toàn' for p in churn_probabilities],
    'Lí do chủ yếu': reasons
})

# Sắp xếp và lưu file
results_df = results_df.sort_values(by='Xác suất rời bỏ (%)', ascending=False)
results_df.to_csv('Bao_Cao_Churn_Co_Li_Do.csv', index=False, encoding='utf-8-sig')

print(f"\nPhân tích hoàn tất!")
print(f"- Tổng số khách hàng đang xét duyệt: {len(results_df)}")
print(f"- Số người bị cảnh báo rời bỏ: {sum(results_df['Dự đoán tháng sau'] == 'Cảnh báo rời đi')}")
print(f"- Tỷ lệ dự báo rời bỏ tháng tới: {(sum(results_df['Dự đoán tháng sau'] == 'Cảnh báo rời đi') / len(results_df) * 100):.2f}%")
print("\nĐã xuất file 'Bao_Cao_Churn_Co_Li_Do.csv' chứa đầy đủ nguyên nhân!")