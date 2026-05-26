## PHÂN TÍCH HÀNH VI MUA SẮM VÀ DỰ ĐOÁN ĐĂNG KÝ CHƯƠNG TRÌNH THÀNH VIÊN
Dự án ứng dụng phân lớp để phân tích hành vi mua sắm của khách hàng bán lẻ, từ đó dự đoán khả năng đăng ký chương trình khách hàng thành viên.
## Tổng Quan
Nghiên cứu này phân tích hành vi mua sắm của khách hàng trong môi trường bán lẻ thông qua các phương pháp khai phá dữ liệu, tập trung vào việc xác định những hành vi nào có ảnh hưởng đến quyết định đăng ký chương trình thành viên (loyalty program).
Dự án áp dụng học có giám sát (phân lớp) trên bộ dữ liệu Customer Shopping Trends gồm 3.900 quan sát và 19 đặc trưng, nhằm cung cấp cơ sở phân tích hành vi khách hàng dựa trên dữ liệu thực tế.
## Câu Hỏi Nghiên Cứu
Những hành vi mua sắm ảnh hưởng như thế nào đến quyết định đăng ký chương trình thành viên của khách hàng?
## Phương Pháp
Dự án tuân theo quy trình chuẩn KDD (Knowledge Discovery in Databases):
Dữ liệu thô → Lựa chọn → Tiền xử lý → Biến đổi → Khai phá → Đánh giá → Tri thức 
### Tiền Xử Lý
1. Chuẩn hóa biến số bằng StandardScaler (μ = 0, σ² = 1)
2. Mã hóa biến định tính bằng One-Hot Encoding và Label Encoding
3. Kiểm tra dữ liệu cho thấy không có giá trị thiếu, dữ liệu sạch, sẵn sàng cho mô hình hóa
### Chia Dữ Liệu
75% huấn luyện / 25% kiểm tra
### Đánh giá các mô hình




