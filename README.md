<div align="center">

# Shopping Behavior Analysis
### Membership Subscription Prediction with Machine Learning

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![pandas](https://img.shields.io/badge/pandas-Data-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org)

Phân tích hành vi mua sắm của 3,900 khách hàng và xây dựng mô hình dự đoán khả năng đăng ký chương trình thành viên với độ chính xác 84.1%.

[Khám phá EDA](#phan-tich-kham-pha-du-lieu-eda) · [Mô hình](#mo-hinh-hoc-may) · [Kết quả](#ket-qua--so-sanh-mo-hinh) · [Khuyến nghị](#khuyen-nghi-chien-luoc-cho-doanh-nghiep)

</div>

---

## Giới thiệu

Chương trình khách hàng thành viên (loyalty program) là công cụ marketing phòng thủ quan trọng để giữ chân khách hàng hiện tại. Tuy nhiên, không phải mọi khách hàng đều có xu hướng đăng ký — việc tiếp cận dàn trải tốn nguồn lực mà hiệu quả thấp.

**Câu hỏi nghiên cứu:** Dựa vào dữ liệu hành vi mua sắm, liệu có thể dự đoán khách hàng nào sẽ đăng ký chương trình thành viên không? Và những hành vi nào ảnh hưởng quyết định nhất đến quyết định đó?

Dự án này sử dụng các kỹ thuật Machine Learning để:
- Khám phá các mẫu hành vi mua sắm trong dữ liệu thực tế
- Xây dựng và so sánh 4 mô hình phân loại
- Chuyển hóa kết quả mô hình thành chiến lược kinh doanh có thể hành động được

---

## Cấu trúc dự án

```
shopping-behavior-analysis/
├── analysis.ipynb              # Notebook chính — EDA + Modeling
├── data_shopping_trends.csv    # Dataset gốc (3,900 records)
├── images/
│   ├── feature_importance.png
│   ├── roc_curves.png
│   └── confusion_matrices.png
└── README.md
```

---

## Dataset

| Thông số | Giá trị |
|---|---|
| Nguồn | [Kaggle — Customer Shopping Trends](https://www.kaggle.com/datasets/bhadramohit/customer-shopping-latest-trends-dataset) |
| Số quan sát | 3,900 rows |
| Số đặc trưng | 19 columns |
| Biến số (numeric) | 5 |
| Biến phân loại (categorical) | 14 |
| Missing values | Không có |
| Biến mục tiêu | `Subscription Status` (Yes / No) |

<details>
<summary>Xem chi tiết các đặc trưng</summary>

<br/>

**Nhân khẩu học**
| Cột | Kiểu | Mô tả |
|---|---|---|
| `Customer ID` | int | Mã định danh khách hàng |
| `Age` | int | Tuổi (18–70) |
| `Gender` | categorical | Male / Female |
| `Location` | categorical | Khu vực sinh sống |

**Sản phẩm & Giao dịch**
| Cột | Kiểu | Mô tả |
|---|---|---|
| `Item Purchased` | categorical | Tên sản phẩm |
| `Category` | categorical | Danh mục (Clothing, Footwear...) |
| `Purchase Amount (USD)` | float | Giá trị giao dịch (20–100 USD) |
| `Size` | categorical | Kích cỡ sản phẩm |
| `Color` | categorical | Màu sắc |
| `Season` | categorical | Mùa mua hàng |

**Hành vi & Đánh giá**
| Cột | Kiểu | Mô tả |
|---|---|---|
| `Review Rating` | float | Điểm đánh giá (1.0–5.0) |
| `Subscription Status` | binary | **TARGET** — Yes / No |
| `Frequency of Purchases` | categorical | Tần suất mua hàng |
| `Previous Purchases` | int | Số lần mua trước đây (0–50) |

**Thanh toán & Giao hàng**
| Cột | Kiểu | Mô tả |
|---|---|---|
| `Payment Method` | categorical | Phương thức thanh toán |
| `Preferred Payment Method` | categorical | Phương thức ưa thích |
| `Shipping Type` | categorical | Hình thức giao hàng |
| `Discount Applied` | binary | Có áp dụng giảm giá không |
| `Promo Code Used` | binary | Có dùng mã khuyến mãi không |

</details>

---

## Phân tích khám phá dữ liệu (EDA)

### Thống kê mô tả

| Thuộc tính | Mean | Median | Min | Max |
|---|---|---|---|---|
| Purchase Amount (USD) | 59.76 | 60 | 20 | 100 |
| Age | 44.07 | 44 | 18 | 70 |
| Previous Purchases | 25.35 | 25 | 1 | 50 |
| Review Rating | 3.750 | 3.7 | 2.6 | 5.0 |

### Phân phối các biến

- **Purchase Amount**: Phân bố khá đồng đều (20–100 USD); phân khúc chi tiêu cao (90–100 USD) có tần suất nhỉnh hơn, không tập trung rõ vào một mức cụ thể
- **Age**: Nhóm trung niên–cao tuổi (45–70) chiếm ưu thế; nhóm khách hàng trẻ dưới 30 tuổi chiếm tần suất thấp hơn đáng kể
- **Previous Purchases**: Trải rộng 0–50 lần; nhóm khách trung thành (45–50 lần) có tần suất cao hơn hẳn
- **Review Rating**: Tập trung 3.0–5.0; đỉnh tại khoảng 4.5 — mức độ hài lòng chung cao

### Ma trận tương quan

Tất cả cặp biến định lượng đều có **|r| < 0.05** — không có multicollinearity, mỗi biến đo lường một khía cạnh độc lập của hành vi khách hàng và có thể đóng góp riêng vào mô hình.

| Cặp biến | Pearson r |
|---|:---:|
| Age — Previous Purchases | +0.040 |
| Purchase Amount — Review Rating | +0.031 |
| Age — Review Rating | -0.022 |
| Age — Purchase Amount | -0.010 |
| Previous Purchases — Purchase Amount | +0.008 |
| Previous Purchases — Review Rating | +0.004 |

---

## Tiền xử lý dữ liệu

### 1. Chuẩn hóa biến số

```python
from sklearn.preprocessing import StandardScaler

numeric_cols = ['Purchase Amount (USD)', 'Previous Purchases', 'Review Rating']
scaler = StandardScaler()
df_numeric_scaled = pd.DataFrame(
    scaler.fit_transform(df[numeric_cols]),
    columns=numeric_cols
)
# Đưa các features về μ = 0, σ = 1
# Cần thiết để SVM và Logistic Regression không bị lệch bởi scale của từng biến
```

### 2. Mã hóa biến mục tiêu

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(df['Subscription Status'])
# Yes → 1  |  No → 0
```

### 3. One-Hot Encoding

```python
categorical_cols = [
    'Payment Method', 'Shipping Type', 'Promo Code Used',
    'Preferred Payment Method', 'Frequency of Purchases'
]
df_cat_encoded = pd.get_dummies(df[categorical_cols], drop_first=True)
# drop_first=True tránh dummy variable trap (perfect multicollinearity)
```

### 4. Train / Test Split

```python
X = pd.concat([df_numeric_scaled, df_cat_encoded], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y    # Giữ tỷ lệ Yes/No cân bằng trong cả 2 tập
)
# Train: 75% (2,925 records) | Test: 25% (975 records)
```

---

## Mô hình học máy

4 thuật toán được huấn luyện song song để so sánh toàn diện:

```python
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'SVM':                 SVC(probability=True, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'Neural Network':      MLPClassifier(random_state=42, max_iter=500)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        'Model':     name,
        'Accuracy':  accuracy_score(y_test, y_pred),
        'F1':        f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall':    recall_score(y_test, y_pred),
        'AUC-ROC':   roc_auc_score(y_test, y_proba),
        'MCC':       matthews_corrcoef(y_test, y_pred)
    })
```

---

## Đánh giá các mô hình

### Bảng đánh giá tổng thể

| Mô hình | AUC-ROC | Accuracy | F1 Score | Precision | Recall | MCC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | 0.895 | **0.841** | **0.849** | **0.896** | 0.841 | **0.694** |
| SVM | **0.899** | 0.832 | 0.841 | 0.877 | 0.832 | 0.657 |
| Neural Network | 0.896 | 0.809 | 0.809 | 0.810 | 0.809 | 0.514 |
| Decision Tree | 0.773 | 0.785 | 0.784 | 0.783 | 0.785 | 0.445 |

### Phân tích Confusion Matrix

| Mô hình | True Negative | False Positive | False Negative | True Positive |
|---|:---:|:---:|:---:|:---:|
| **Logistic Regression** | 99.3% | 37.0% | **0.7%** | 63.0% |
| SVM | 96.9% | 37.4% | 3.1% | 62.6% |
| Neural Network | 87.2% | 36.1% | 12.8% | 63.9% |
| Decision Tree | 84.8% | 39.7% | 15.2% | 60.3% |

**False Negative** (bỏ sót khách hàng tiềm năng) là chi phí cao nhất về mặt kinh doanh — đồng nghĩa với việc mất cơ hội chuyển đổi khách hàng vào chương trình thành viên. Logistic Regression đạt FN thấp nhất **(0.7%)**, gần như không bỏ sót khách hàng có khả năng đăng ký.

### => Logistic Regression được lựa chọn làm mô hình phù hợp hơn cho bài toán này

---

## Feature Importance

Hệ số hồi quy (coefficients) của Logistic Regression được phân tích để xác định các yếu tố hành vi tác động mạnh nhất:

```python
model_lr = LogisticRegression(random_state=42)
model_lr.fit(X, y)

coefs = model_lr.coef_[0]
sorted_idx = np.argsort(np.abs(coefs))[::-1]

sorted_features = [X.columns[i] for i in sorted_idx]
sorted_coefs    = coefs[sorted_idx]
colors = ['skyblue' if c >= 0 else 'salmon' for c in sorted_coefs]

plt.barh(sorted_features, sorted_coefs, color=colors)
plt.axvline(0, color='grey', linestyle='--')
plt.title('Feature Importance — Logistic Regression')
```

### Ranking các yếu tố ảnh hưởng

| Hạng | Yếu tố hành vi | Tác động | Diễn giải |
|:---:|---|:---:|---|
| 1 | Promo Code Used = Yes | Tích cực mạnh | Dùng mã khuyến mãi phản ánh mức độ gắn kết cao với thương hiệu |
| 2 | Frequency of Purchases (Weekly/Monthly) | Tích cực | Mua thường xuyên tạo thói quen gắn bó — xác suất đăng ký cao hơn |
| 3 | Shipping Type (Express/Next Day Air) | Tích cực | Chọn giao hàng nhanh cho thấy sẵn sàng đầu tư vào trải nghiệm |
| 4 | Previous Purchases (cao) | Tích cực | Nhiều giao dịch cũ đồng nghĩa với niềm tin và gắn bó thương hiệu |
| 5 | Payment Method (Debit/Credit Card) | Tích cực nhẹ | Quen dùng thẻ thanh toán — thiên về các dịch vụ số và đăng ký |
| 6 | Review Rating (cao) | Tích cực yếu | Hài lòng là điều kiện cần nhưng chưa đủ để thúc đẩy đăng ký |
| 7 | Purchase Amount (USD) | Rất yếu | Chi tiêu cao không đồng nghĩa với muốn gắn bó lâu dài |

> **Phát hiện cốt lõi:** Quyết định đăng ký thành viên được quyết định bởi **mức độ gắn kết hành vi** (tần suất mua, sử dụng KM, lịch sử giao dịch) — không phải **giá trị giao dịch đơn lẻ** (số tiền chi tiêu, điểm hài lòng). Khách hàng trung thành về thói quen mới là đối tượng mục tiêu thực sự của loyalty program.

---

## Khuyến nghị chiến lược cho doanh nghiệp

Dựa trên kết quả phân tích, phần này đề xuất hướng đi cụ thể cho doanh nghiệp bán lẻ muốn tối ưu hóa chương trình thành viên.


### 1. Xác định đúng nhóm khách hàng mục tiêu

Thay vì tiếp cận toàn bộ khách hàng, doanh nghiệp nên tập trung nguồn lực vào nhóm có xác suất đăng ký cao nhất. Hồ sơ khách hàng tiềm năng điển hình:

- Có lịch sử sử dụng mã khuyến mãi
- Mua hàng định kỳ hàng tuần hoặc hàng tháng
- Có trên 25 lần mua trước đây
- Ưa thích thanh toán bằng thẻ (Credit/Debit Card)
- Chọn hình thức giao hàng nhanh (Express, Next Day Air)

Việc áp dụng mô hình Logistic Regression để scoring toàn bộ tệp khách hàng sẽ cho phép doanh nghiệp phân tầng ưu tiên — nhóm xác suất cao được tiếp cận trước, nhóm thấp có thể nurture dần bằng chi phí thấp hơn.

---

### 2. Tái thiết kế cơ chế khuyến mãi thành viên

Vì **Promo Code Used** là yếu tố dự báo mạnh nhất, có thể kết luận rằng khách hàng phản hồi tốt với các ưu đãi cụ thể, rõ ràng hơn là các lợi ích trừu tượng. Doanh nghiệp nên:

- Thiết kế mã khuyến mãi đặc quyền chỉ dành cho thành viên — tạo động lực đăng ký trực tiếp
- Tránh các chương trình loyalty mơ hồ (tích điểm không có lộ trình đổi quà rõ ràng) — nhóm khách hàng phản hồi tốt với mã KM thiên về lợi ích ngay lập tức, không phải lợi ích dài hạn không rõ ràng

---

### 3. Khai thác nhóm khách hàng cũ có lịch sử mua cao

**Previous Purchases** có tương quan tích cực với khả năng đăng ký thành viên. Nhóm khách hàng đã có trên 30–40 lần mua nhưng chưa đăng ký là cơ hội chuyển đổi bị bỏ qua lớn nhất, vì:

- Họ đã chứng minh mức độ tin tưởng và gắn bó với thương hiệu qua hành động thực tế
- Rào cản duy nhất thường là thiếu thông tin hoặc chưa được tiếp cận đúng thời điểm
- Chi phí thuyết phục nhóm này thấp hơn nhiều so với việc tìm khách hàng mới

Đề xuất: Chạy chiến dịch remarketing riêng biệt cho nhóm này với offer đặc biệt "dành riêng cho khách hàng thân thiết chưa là thành viên" — đây là thông điệp có tính cá nhân hóa cao và chuyển đổi tốt.

---

## Tài liệu tham khảo

- Sharp, B. et al. (1997). *Loyalty programs and their impact on repeat purchase loyalty patterns*
- Uncles, M. et al. (2003). *Customer loyalty and customer loyalty programs*. Journal of Consumer Marketing
- Dataset: [Customer Shopping Trends — Kaggle](https://www.kaggle.com/datasets/bhadramohit/customer-shopping-latest-trends-dataset)
