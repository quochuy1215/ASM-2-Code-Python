import pandas as pd
df = pd.read_csv('sale_data.csv' , sep=';')
import matplotlib.pyplot as plt
sale_df = pd.read_csv('sale_data.csv')


# Kiểm tra giá trị bị thiếu trong DataFrame
missing_values = sale_df.isnull().sum()
print(missing_values)

# Xác định các kiểu dữ liệu không chính xác
data_types = sale_df.dtypes
print(data_types)

# Kiểm tra các bản ghi trùng lặp trong cột SaleID
duplicate_sales = sale_df.duplicated(subset=['SaleID']).sum()
print(f"Số lượng SaleID trùng lặp: {duplicate_sales}")

# Ví dụ: Xóa các hàng có giá trị bị thiếu
sale_df_cleaned = sale_df.dropna()

# Ví dụ: Chuyển đổi kiểu dữ liệu nếu cần thiết
sale_df_cleaned['SaleDate'] = pd.to_datetime(sale_df_cleaned['SaleDate'])

# Ví dụ: Xóa các bản ghi trùng lặp
sale_df_cleaned = sale_df_cleaned.drop_duplicates(subset=['SaleID'])

# Kiểm tra lại giá trị bị thiếu
print(sale_df_cleaned.isnull().sum())

# Kiểm tra lại các bản ghi trùng lặp
print(f"Số lượng SaleID trùng lặp: {sale_df_cleaned.duplicated(subset=['SaleID']).sum()}")


# Hiển thị dữ liệu sau khi làm sạch
print("\nDữ liệu sau khi làm sạch:")
print(sale_df_cleaned.head())


# Visual
# Load the Sale Data CSV file
sale_df = pd.read_csv('sale_data.csv')

# Convert the SaleDate column to datetime format if not already done
sale_df['SaleDate'] = pd.to_datetime(sale_df['SaleDate'])
# Step 3.2.1: Line Chart for Sales Over Time
# Calculate total sales by date
sales_by_date = sale_df.groupby(sale_df['SaleDate'].dt.date)['TotalPrice'].sum()

# Plot the line chart
plt.figure(figsize=(10, 5))
plt.plot(sales_by_date.index, sales_by_date.values, marker='o', linestyle='-')
plt.title('Doanh số bán hàng theo thời gian')
plt.xlabel('Ngày')
plt.ylabel('Tổng doanh số bán hàng')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 3.2.2: Bar Chart for Sales by Payment Method
# Calculate total sales by payment method
sales_by_payment = sale_df.groupby('PaymentMethod')['TotalPrice'].sum()

# Plot the bar chart
plt.figure(figsize=(10, 5))
plt.bar(sales_by_payment.index, sales_by_payment.values, color='skyblue')
plt.title('Doanh số bán hàng theo phương thức thanh toán')
plt.xlabel('Phương thức thanh toán')
plt.ylabel('Tổng doanh số bán hàng')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 3.2.3: Pie Chart for Sales by Status
# Calculate total sales by sale status
sales_by_status = sale_df.groupby('SaleStatus')['TotalPrice'].sum()

# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(sales_by_status.values, labels=sales_by_status.index, autopct='%1.1f%%', startangle=140)
plt.title('Tỷ lệ doanh số bán hàng theo trạng thái')
plt.axis('equal')  # Ensure the pie chart is circular
plt.show()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Tải dữ liệu
data = pd.read_csv('sale_data.csv')

# Giả sử chúng ta muốn dự đoán doanh số (Sales) dựa trên một số đặc trưng
# Chuẩn bị dữ liệu
X = data[['Feature1', 'Feature2']]  # Đặc trưng giả định
y = data['Sales']  # Biến mục tiêu

# Chia tách dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)

# Tạo biểu đồ đường để so sánh giá trị thực tế và giá trị dự đoán
plt.plot(y_test.values, label='Thực tế')
plt.plot(y_pred, label='Dự đoán')
plt.xlabel('Mẫu')
plt.ylabel('Doanh số')
plt.title('So sánh Doanh số Thực tế và Dự đoán')
plt.legend()
plt.show()


from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
# Tải dữ liệu
data = pd.read_csv('path_to_your_file/sale_data.csv')

# Chuẩn bị dữ liệu
X = data[['Feature1', 'Feature2']]  # Các đặc trưng giả định
y = data['Sales']  # Biến mục tiêu

# Chia tách dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse}')
print(f'R-squared: {r2}')

# Tạo biểu đồ để so sánh giá trị thực tế và giá trị dự đoán
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Thực tế', linestyle='-', marker='o')
plt.plot(y_pred, label='Dự đoán', linestyle='--', marker='x')
plt.xlabel('Mẫu')
plt.ylabel('Doanh số')
plt.title('So sánh Doanh số Thực tế và Dự đoán')
plt.legend()
plt.show()


# Tải dữ liệu
data = pd.read_csv('path_to_your_file/sale_data.csv')

# Chuẩn bị dữ liệu
X = data[['Feature1', 'Feature2']]  # Các đặc trưng giả định
y = data['Sales']  # Biến mục tiêu

# Chia tách dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse}')
print(f'R-squared: {r2}')

# Tạo biểu đồ để so sánh giá trị thực tế và dự đoán
plt.plot(y_test.values, label='Thực tế')
plt.plot(y_pred, label='Dự đoán')
plt.xlabel('Mẫu')
plt.ylabel('Doanh số')
plt.title('So sánh Doanh số Thực tế và Dự đoán')
plt.legend()
plt.show()