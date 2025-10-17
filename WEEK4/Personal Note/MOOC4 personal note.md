# IBM – Unsupervised Machine Learning: Full Course Notes

## **MODULE 1: Introduction to Unsupervised Learning and K-Means**

### **1. Tổng quan nội dung**
Module đầu tiên giới thiệu về **học không giám sát (Unsupervised Learning)** – một phương pháp trong Machine Learning nhằm **khám phá cấu trúc tiềm ẩn trong dữ liệu không có nhãn**. Thay vì huấn luyện từ các cặp (x, y) như trong Supervised Learning, Unsupervised Learning tìm kiếm **mối quan hệ giữa các mẫu dữ liệu**.

Thuật toán nổi bật trong phần này là **K-Means Clustering**, một phương pháp phân cụm (clustering) dựa trên khoảng cách.

---

### **2. Thuật ngữ chính (Key Terms)**
- **Unsupervised Learning (Học không giám sát):** Phương pháp học từ dữ liệu không có nhãn, mục tiêu là khám phá cấu trúc tiềm ẩn.
- **Cluster (Cụm):** Nhóm các điểm dữ liệu có đặc điểm tương đồng cao.
- **Centroid (Tâm cụm):** Trung tâm của một cụm, tính bằng trung bình cộng các điểm trong cụm.
- **K-Means Algorithm:** Thuật toán phân cụm phổ biến, lặp lại việc gán điểm dữ liệu vào cụm gần nhất và cập nhật tâm cụm.
- **Iteration (Lặp):** Quá trình cập nhật liên tục tâm cụm cho đến khi hội tụ.

---

### **3. Công thức toán học & Giải thích**
Mục tiêu của K-Means là **tối thiểu hoá tổng bình phương khoảng cách giữa các điểm dữ liệu và tâm cụm tương ứng**:

$$J = \sum_{i=1}^{k}\sum_{x \in C_i} \lVert x - \mu_i \rVert^2$$

Trong đó:
- **J:** hàm mục tiêu (objective function) – thể hiện độ sai lệch tổng.
- **k:** số cụm (clusters).
- **x:** điểm dữ liệu.
- **μᵢ:** tâm cụm thứ i (centroid of cluster i).
- **‖x - μᵢ‖²:** bình phương khoảng cách **Euclidean** giữa điểm x và tâm cụm.

**Khoảng cách Euclidean:**

$$d(x, \mu_i) = \sqrt{(x_1 - \mu_{i1})^2 + (x_2 - \mu_{i2})^2 + \dots + (x_n - \mu_{in})^2}$$

Khoảng cách này đo độ khác nhau giữa hai vector trong không gian n-chiều.

---

### **4. Ví dụ minh hoạ**
Giả sử ta có 6 điểm dữ liệu trên mặt phẳng 2D, cần chia thành **k = 2** cụm:

1. Chọn ngẫu nhiên 2 tâm cụm ban đầu.
2. Gán mỗi điểm vào cụm có tâm gần nhất.
3. Cập nhật tâm cụm bằng trung bình của các điểm trong cụm.
4. Lặp lại đến khi tâm cụm không thay đổi nhiều.

**Ví dụ cụ thể:**
- Điểm: A(1,1), B(2,1), C(4,3), D(5,4), E(1,2), F(5,3)
- Khởi tạo: μ₁ = (1,1), μ₂ = (5,4)
- Iteration 1: Gán A,B,E vào C₁; C,D,F vào C₂
- Cập nhật: μ₁_new = (1.33, 1.33), μ₂_new = (4.67, 3.33)
- Lặp lại đến khi hội tụ

---

### **5. Ghi chú và lưu ý**
- Cần chọn trước số cụm k (thường xác định bằng **Elbow Method**).
- Dữ liệu nên được **chuẩn hoá (Standardization)** để tránh bias do khác thang đo.
- Kết quả phụ thuộc vào **khởi tạo tâm cụm** (initialization); dùng **K-Means++** để tối ưu hoá chọn tâm ban đầu.
- K-Means hoạt động tốt với cụm hình cầu, kém hiệu quả với cụm hình dạng phức tạp.

---

### **6. Key Takeaways**
 Unsupervised Learning không cần nhãn, tìm cấu trúc ẩn trong dữ liệu  
 K-Means tối ưu hóa khoảng cách điểm đến tâm cụm  
 Cần chuẩn hóa dữ liệu và chọn k hợp lý (Elbow Method)  
 K-Means++ giúp khởi tạo tốt hơn, tránh local minima  
 Phù hợp với dữ liệu có cụm hình cầu và phân bố đều  

---

## **MODULE 2: Distance Metrics & Computational Hurdles**

### **1. Tổng quan nội dung**
Module này tập trung vào **các độ đo khoảng cách (Distance Metrics)** – công cụ cốt lõi để đánh giá độ tương tự giữa các điểm dữ liệu trong clustering. Việc chọn metric phù hợp ảnh hưởng trực tiếp đến chất lượng phân cụm.

Ngoài ra, module đề cập đến **các thách thức tính toán** khi xử lý dữ liệu lớn và cách giải quyết.

---

### **2. Thuật ngữ chính (Key Terms)**
- **Euclidean Distance (Khoảng cách Euclid):** Khoảng cách đường thẳng giữa hai điểm trong không gian.
- **Manhattan Distance (Khoảng cách Manhattan):** Tổng độ chênh lệch theo từng trục tọa độ (như đi trên lưới ô vuông).
- **Minkowski Distance:** Tổng quát hóa của Euclidean và Manhattan với tham số p.
- **Cosine Similarity (Độ tương tự Cosine):** Đo góc giữa hai vector, phù hợp với dữ liệu văn bản.
- **Normalization/Standardization:** Chuẩn hóa dữ liệu về cùng thang đo trước khi tính khoảng cách.
- **Computational Complexity (Độ phức tạp tính toán):** Chi phí tính toán tăng theo kích thước dữ liệu.

---

### **3. Công thức toán học & Giải thích**

#### **3.1. Euclidean Distance**
$$d(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$

Đo khoảng cách đường thẳng trong không gian n-chiều. Nhạy cảm với outliers và khác biệt về thang đo.

#### **3.2. Manhattan Distance**
$$d(x, y) = \sum_{i=1}^{n}|x_i - y_i|$$

Tổng giá trị tuyệt đối của chênh lệch trên từng chiều. Ít bị ảnh hưởng bởi outliers hơn Euclidean.

#### **3.3. Minkowski Distance**
$$d(x, y) = \left(\sum_{i=1}^{n}|x_i - y_i|^p\right)^{1/p}$$

- **p = 1:** Manhattan Distance
- **p = 2:** Euclidean Distance
- **p → ∞:** Chebyshev Distance (max|xᵢ - yᵢ|)

#### **3.4. Cosine Similarity**
$$\text{cosine}(x,y) = \frac{x \cdot y}{\lVert x \rVert \lVert y \rVert} = \frac{\sum_{i=1}^{n} x_i y_i}{\sqrt{\sum_{i=1}^{n} x_i^2} \cdot \sqrt{\sum_{i=1}^{n} y_i^2}}$$

**Cosine Distance:**
$$d_{\text{cosine}}(x,y) = 1 - \text{cosine}(x,y)$$

Đo độ tương đồng về hướng, không quan tâm độ lớn. Phù hợp với text mining, recommendation systems.

---

### **4. Ví dụ minh hoạ**

Cho hai điểm: **x = (1, 2)**, **y = (4, 6)**

#### **Euclidean:**
$$d = \sqrt{(1-4)^2 + (2-6)^2} = \sqrt{9+16} = \sqrt{25} = 5$$

#### **Manhattan:**
$$d = |1-4| + |2-6| = 3 + 4 = 7$$

#### **Cosine Similarity:**
$$\text{cosine} = \frac{1×4 + 2×6}{\sqrt{1^2+2^2} \cdot \sqrt{4^2+6^2}} = \frac{16}{\sqrt{5} \cdot \sqrt{52}} = \frac{16}{16.12} \approx 0.993$$

$$d_{\text{cosine}} = 1 - 0.993 = 0.007$$

**Nhận xét:** Cosine distance rất nhỏ → hai vector gần như cùng hướng.

---

### **5. Ghi chú và lưu ý**

#### **5.1. Computational Hurdles (Thách thức tính toán)**
- Với n điểm dữ liệu, tính tất cả khoảng cách cặp: **O(n²)** – không khả thi với dữ liệu lớn.
- K-Means chuẩn: mỗi iteration tính khoảng cách n×k lần → **O(n×k×t×d)** với t iterations, d dimensions.

#### **5.2. Giải pháp tối ưu**
- **MiniBatch K-Means:** Chỉ dùng một phần dữ liệu mỗi iteration.
- **Approximate Nearest Neighbor (ANN):** Dùng cấu trúc dữ liệu như KD-Tree, Ball Tree.
- **Dimensionality Reduction:** Dùng PCA giảm số chiều trước khi clustering.
- **Parallel Computing:** Tận dụng GPU, distributed computing.

#### **5.3. Lựa chọn metric phù hợp**
- **Euclidean:** Dữ liệu số liệu, cụm hình cầu, đã chuẩn hóa.
- **Manhattan:** Dữ liệu có outliers, grid-based structures.
- **Cosine:** Text data, sparse vectors, quan tâm hướng hơn độ lớn.

---

### **6. Key Takeaways**
 Euclidean phù hợp với dữ liệu số đã chuẩn hóa, cụm compact  
 Manhattan ít nhạy với outliers, tốt cho dữ liệu grid  
 Cosine đo độ tương đồng hướng, lý tưởng cho text/sparse data  
 Luôn chuẩn hóa dữ liệu trước khi tính khoảng cách  
 Với dữ liệu lớn: dùng MiniBatch K-Means, ANN, hoặc PCA  

---

## **MODULE 3: Selecting a Clustering Algorithm**

### **1. Tổng quan nội dung**
Module này hướng dẫn cách **lựa chọn thuật toán phân cụm phù hợp** dựa trên đặc điểm dữ liệu: kích thước, hình dạng cụm, nhiễu, và mục tiêu phân tích. Mỗi thuật toán có ưu nhược điểm riêng, không có "best algorithm for all".

---

### **2. Thuật ngữ chính (Key Terms)**
- **Hierarchical Clustering (Phân cụm phân cấp):** Xây dựng cây phân cấp (dendrogram) thể hiện mối quan hệ giữa các cụm.
- **DBSCAN (Density-Based Spatial Clustering):** Phân cụm dựa trên mật độ điểm.
- **Gaussian Mixture Model (GMM):** Mô hình hỗn hợp Gaussian, gán xác suất cho mỗi điểm thuộc cụm.
- **Dendrogram:** Biểu đồ cây thể hiện quá trình gộp/chia cụm trong hierarchical clustering.
- **Soft Clustering:** Cho phép một điểm thuộc nhiều cụm với xác suất khác nhau (GMM).
- **Hard Clustering:** Mỗi điểm chỉ thuộc đúng 1 cụm (K-Means, DBSCAN).

---

### **3. Công thức toán học & Giải thích**

#### **3.1. Hierarchical Clustering**

**Agglomerative (Bottom-up):**
1. Mỗi điểm là một cụm
2. Gộp 2 cụm gần nhất
3. Lặp lại đến khi còn 1 cụm

**Linkage criteria:**
- **Single Linkage:** min d(a,b) với a ∈ C₁, b ∈ C₂
- **Complete Linkage:** max d(a,b)
- **Average Linkage:** trung bình tất cả d(a,b)
- **Ward's Method:** tối thiểu variance khi gộp cụm

#### **3.2. Gaussian Mixture Model (GMM)**

$$P(x) = \sum_{i=1}^{k} \pi_i \cdot \mathcal{N}(x | \mu_i, \Sigma_i)$$

Trong đó:
- **πᵢ:** Trọng số cụm i (mixing coefficient), ∑πᵢ = 1
- **𝒩(x|μᵢ, Σᵢ):** Phân phối Gaussian với mean μᵢ và covariance Σᵢ
- **μᵢ:** Vector trung bình của cụm i
- **Σᵢ:** Ma trận hiệp phương sai của cụm i

**Phân phối Gaussian đa chiều:**

$$\mathcal{N}(x|\mu,\Sigma) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)$$

**Thuật toán EM (Expectation-Maximization):**
- **E-step:** Tính xác suất điểm x thuộc cụm i
- **M-step:** Cập nhật πᵢ, μᵢ, Σᵢ

---

### **4. Ví dụ minh hoạ**

#### **So sánh 4 thuật toán trên cùng dataset:**

**Dataset:** Dữ liệu 2D với 3 cụm không đều, có outliers.

| Thuật toán | Kết quả |
|-----------|---------|
| **K-Means** | Tạo 3 cụm hình cầu, outliers bị gán sai |
| **Hierarchical** | Dendrogram cho thấy cấu trúc phân cấp, nhưng outliers vẫn trong cụm |
| **DBSCAN** | Nhận diện đúng 3 cụm + đánh dấu outliers là noise |
| **GMM** | Tạo 3 cụm với xác suất, một số điểm biên có xác suất trung gian |

---

### **5. Ghi chú và lưu ý**

#### **5.1. Bảng so sánh chi tiết**

| Tiêu chí | K-Means | Hierarchical | DBSCAN | GMM |
|---------|---------|--------------|--------|-----|
| **Số cụm** | Phải chọn trước | Cắt dendrogram | Tự động | Phải chọn trước |
| **Hình dạng cụm** | Hình cầu | Linh hoạt | Bất kỳ | Ellipsoid |
| **Outliers** | Nhạy cảm | Nhạy cảm | Xử lý tốt | Trung bình |
| **Complexity** | O(nkt) | O(n²logn) | O(nlogn) | O(nk²t) |
| **Soft/Hard** | Hard | Hard | Hard | Soft |
| **Scale** | Tốt (n lớn) | Kém (n nhỏ) | Trung bình | Trung bình |

#### **5.2. Decision Tree để chọn thuật toán**

```
Dữ liệu có nhiễu/outliers?
├─ YES → DBSCAN
└─ NO → Biết trước số cụm k?
    ├─ YES → Cụm hình cầu?
    │   ├─ YES → K-Means
    │   └─ NO → GMM
    └─ NO → Hierarchical
```

#### **5.3. Lưu ý khi áp dụng**
- **K-Means:** Chạy nhiều lần với khởi tạo khác nhau, dùng K-Means++.
- **Hierarchical:** Chọn linkage method phù hợp (Ward thường tốt nhất).
- **DBSCAN:** Tuning epsilon (ε) và MinPts quan trọng, dùng k-distance graph.
- **GMM:** Dễ overfit, cần regularization hoặc BIC/AIC để chọn k.

---

### **6. Key Takeaways**
 Không có thuật toán tốt nhất cho mọi trường hợp  
 K-Means: nhanh, đơn giản, nhưng giả định cụm hình cầu  
 Hierarchical: không cần chọn k, nhưng tốn bộ nhớ  
 DBSCAN: xử lý outliers tốt, tìm cụm hình dạng bất kỳ  
 GMM: soft clustering, linh hoạt nhưng giả định Gaussian  
 Chọn thuật toán dựa trên: hình dạng cụm, outliers, quy mô dữ liệu  

---

## **MODULE 4: Clustering Evaluation Metrics**

### **1. Tổng quan nội dung**
Sau khi phân cụm, cần **đánh giá chất lượng** kết quả. Module này giới thiệu các metrics để:
- Đo độ compact (chặt chẽ) trong cụm
- Đo độ separation (tách biệt) giữa các cụm
- So sánh các thuật toán/tham số khác nhau

---

### **2. Thuật ngữ chính (Key Terms)**
- **Inertia/Within-Cluster Sum of Squares (WCSS):** Tổng bình phương khoảng cách trong cụm.
- **Silhouette Score:** Đo độ tách biệt giữa các cụm, từ -1 đến 1.
- **Davies-Bouldin Index (DBI):** Đo tỷ lệ giữa phân tán trong cụm và khoảng cách giữa các cụm.
- **Calinski-Harabasz Index:** Tỷ lệ giữa phân tán giữa các cụm và trong cụm.
- **Elbow Method:** Phương pháp chọn k bằng cách tìm điểm gấp khúc trên đồ thị Inertia.

---

### **3. Công thức toán học & Giải thích**

#### **3.1. Inertia (WCSS)**
$$J = \sum_{i=1}^{k}\sum_{x \in C_i} \lVert x - \mu_i \rVert^2$$

- Càng nhỏ càng tốt (cụm càng compact)
- Giảm khi k tăng → không dùng riêng để chọn k
- Dùng Elbow Method để tìm k tối ưu

#### **3.2. Silhouette Score**

**Cho từng điểm x:**
$$s(x) = \frac{b(x) - a(x)}{\max(a(x), b(x))}$$

Trong đó:
- **a(x):** Khoảng cách trung bình từ x đến các điểm khác trong cùng cụm (cohesion)
- **b(x):** Khoảng cách trung bình từ x đến các điểm trong cụm gần nhất khác (separation)

**Silhouette Score tổng thể:**
$$S = \frac{1}{n}\sum_{i=1}^{n} s(x_i)$$

**Diễn giải:**
- **s(x) ≈ 1:** Điểm được phân cụm tốt
- **s(x) ≈ 0:** Điểm nằm giữa 2 cụm
- **s(x) < 0:** Điểm có thể bị phân cụm sai

#### **3.3. Davies-Bouldin Index (DBI)**
$$DBI = \frac{1}{k}\sum_{i=1}^{k} \max_{j \neq i} \left( \frac{\sigma_i + \sigma_j}{d(c_i, c_j)} \right)$$

Trong đó:
- **σᵢ:** Độ phân tán trung bình trong cụm i
- **d(cᵢ, cⱼ):** Khoảng cách giữa tâm cụm i và j

**Diễn giải:** Càng nhỏ càng tốt (cụm compact và tách biệt).

#### **3.4. Calinski-Harabasz Index (CH)**
$$CH = \frac{SS_B/(k-1)}{SS_W/(n-k)}$$

Trong đó:
- **SS_B:** Between-cluster sum of squares
- **SS_W:** Within-cluster sum of squares
- **n:** Số điểm, **k:** Số cụm

**Diễn giải:** Càng lớn càng tốt (phân tán giữa cụm lớn, trong cụm nhỏ).

---

### **4. Ví dụ minh hoạ**

#### **4.1. Elbow Method**

Chạy K-Means với k = 1, 2, 3, ..., 10, tính Inertia:

| k | Inertia |
|---|---------|
| 1 | 3500 |
| 2 | 1800 |
| 3 | 950 |
| 4 | 850 |
| 5 | 820 |

**Đồ thị:** Gấp khúc rõ tại k=3 → chọn k=3.

#### **4.2. Silhouette Score**

Dataset với 100 điểm, chạy K-Means k=3:

- Cụm 1: 30 điểm, S_avg = 0.75
- Cụm 2: 45 điểm, S_avg = 0.82
- Cụm 3: 25 điểm, S_avg = 0.68

**Tổng:** S = (30×0.75 + 45×0.82 + 25×0.68)/100 = 0.77 → Tốt!

---

### **5. Ghi chú và lưu ý**

#### **5.1. So sánh các metrics**

| Metric | Ưu điểm | Nhược điểm | Giá trị tốt |
|--------|---------|------------|-------------|
| **Inertia** | Đơn giản, nhanh | Không dùng riêng để chọn k | Càng nhỏ |
| **Silhouette** | Trực quan, [-1,1] | Chậm với n lớn | Gần 1 |
| **DBI** | Xét cả cohesion & separation | Nhạy với outliers | Càng nhỏ |
| **CH** | Hiệu quả với n lớn | Giả định cụm convex | Càng lớn |

#### **5.2. Quy trình đánh giá**

```python
# Pseudo-code
for k in range(2, 11):
    model = KMeans(n_clusters=k)
    labels = model.fit_predict(X)
    
    inertia[k] = model.inertia_
    silhouette[k] = silhouette_score(X, labels)
    dbi[k] = davies_bouldin_score(X, labels)
    ch[k] = calinski_harabasz_score(X, labels)

# Vẽ đồ thị và chọn k tối ưu
```

#### **5.3. Lưu ý quan trọng**
- **Elbow Method:** Đôi khi không có điểm gấp khúc rõ ràng.
- **Silhouette:** Tính toán O(n²) → chậm với dữ liệu lớn.
- **Kết hợp nhiều metrics:** Không nên chỉ dựa vào 1 chỉ số.
- **Ground truth:** Nếu có nhãn thật, dùng Adjusted Rand Index (ARI), Normalized Mutual Information (NMI).

---

### **6. Key Takeaways**
 Inertia giảm khi k tăng, dùng Elbow Method để chọn k  
 Silhouette Score [-1,1]: gần 1 là tốt, <0 là phân cụm sai  
 DBI và CH đánh giá cả cohesion và separation  
 Nên kết hợp nhiều metrics để đánh giá toàn diện  
 Với ground truth: dùng ARI, NMI thay vì internal metrics  

---

## **MODULE 5: Density-Based Clustering and DBSCAN**

### **1. Tổng quan nội dung**
**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** là thuật toán phân cụm dựa trên **mật độ điểm** thay vì khoảng cách đến tâm cụm. Ưu điểm lớn: **nhận diện cụm hình dạng bất kỳ** và **tự động phát hiện outliers**.

Không cần chọn số cụm k trước, phù hợp với dữ liệu có nhiễu.

---

### **2. Thuật ngữ chính (Key Terms)**
- **Density (Mật độ):** Số điểm trong một vùng lân cận.
- **ε (epsilon):** Bán kính vùng lân cận.
- **MinPts:** Số điểm tối thiểu trong vùng ε để tạo cụm.
- **Core Point (Điểm lõi):** Điểm có ≥ MinPts điểm trong vùng ε.
- **Border Point (Điểm biên):** Nằm trong vùng ε của core point nhưng không đủ MinPts.
- **Noise Point (Điểm nhiễu):** Không thuộc cụm nào.
- **Directly Density-Reachable:** Điểm q trong vùng ε của core point p.
- **Density-Reachable:** Có chuỗi core points nối từ p đến q.

---

### **3. Công thức toán học & Giải thích**

#### **3.1. Định nghĩa lân cận ε**

$N_{\varepsilon}(p) = \{q \in D \mid d(p,q) \leq \varepsilon\}$

Tập hợp các điểm q có khoảng cách đến p không quá ε.

#### **3.2. Core Point Condition**

Điểm p là core point nếu:
$|N_{\varepsilon}(p)| \geq MinPts$

#### **3.3. Directly Density-Reachable**

Điểm q directly density-reachable từ p nếu:
- p là core point
- q ∈ N_ε(p)

#### **3.4. Density-Reachable**

q density-reachable từ p nếu tồn tại chuỗi: p = p₁, p₂, ..., pₙ = q

Sao cho pᵢ₊₁ directly density-reachable từ pᵢ.

#### **3.5. Density-Connected**

p và q density-connected nếu tồn tại điểm o sao cho cả p và q đều density-reachable từ o.

**Một cụm trong DBSCAN** là tập tất cả các điểm density-connected với nhau.

---

### **4. Ví dụ minh hoạ**

#### **4.1. Thuật toán DBSCAN từng bước**

**Dataset:** 12 điểm, ε = 1.5, MinPts = 3

**Bước 1:** Chọn điểm chưa thăm (A)
- Tìm N_ε(A) = {A, B, C, D} → |N_ε(A)| = 4 ≥ 3 → A là core point
- Tạo Cluster 1 = {A, B, C, D}

**Bước 2:** Mở rộng từ B (trong Cluster 1)
- N_ε(B) = {A, B, E} → B là core point
- Thêm E vào Cluster 1

**Bước 3:** Tiếp tục với C, D, E...

**Bước 4:** Điểm X không đến được từ core point nào → X là noise

**Kết quả:**
- Cluster 1: {A, B, C, D, E, F}
- Cluster 2: {G, H, I}
- Noise: {X, Y}

---

### **5. Ghi chú và lưu ý**

#### **5.1. Chọn tham số ε và MinPts**

**MinPts:**
- Thường chọn: MinPts = 2×dim (dim là số chiều)
- Tối thiểu: MinPts = 3 (cho dữ liệu 2D)
- Dữ liệu nhiều chiều hoặc nhiễu: tăng MinPts

**ε (epsilon):**
- Dùng **k-distance graph:**
  1. Tính khoảng cách đến điểm thứ k gần nhất (k = MinPts)
  2. Sắp xếp tăng dần
  3. Vẽ đồ thị, tìm điểm "gấp khúc" → chọn ε

```python
# Pseudo-code k-distance
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=MinPts).fit(X)
distances, indices = nbrs.kneighbors(X)
distances = np.sort(distances[:, -1])
plt.plot(distances)  # Tìm elbow
```

#### **5.2. Ưu và Nhược điểm**

**Ưu điểm:**
 Nhận diện cụm hình dạng bất kỳ (không chỉ hình cầu)
 Tự động phát hiện và loại bỏ outliers
 Không cần chọn số cụm k trước
 Robust với nhiễu

**Nhược điểm:**
 Nhạy cảm với tham số ε và MinPts
 Không hiệu quả nếu mật độ cụm thay đổi nhiều
 Khó xử lý dữ liệu cao chiều (curse of dimensionality)
 Độ phức tạp O(n²) → chậm với dữ liệu lớn (cải thiện bằng spatial index)

#### **5.3. Biến thể của DBSCAN**

- **HDBSCAN:** Hierarchical DBSCAN, tự động chọn ε cho từng vùng
- **OPTICS:** Ordering Points To Identify Clustering Structure, không cần chọn ε cố định
- **DBSCAN++:** Tối ưu hóa hiệu suất với spatial indexing

---

### **6. Key Takeaways**
 DBSCAN phân cụm dựa mật độ, tìm cụm hình dạng bất kỳ  
 Tự động phát hiện outliers (noise points)  
 Không cần chọn số cụm k trước  
 Tham số quan trọng: ε (bán kính) và MinPts (ngưỡng mật độ)  
 Dùng k-distance graph để chọn ε hợp lý  
 Hạn chế: nhạy với tham số, kém hiệu quả nếu mật độ không đều  

---

## **MODULE 6: Dimensionality Reduction Techniques (PCA & t-SNE)**

### **1. Tổng quan nội dung**
**Giảm chiều dữ liệu (Dimensionality Reduction)** là quá trình biến đổi dữ liệu từ không gian nhiều chiều sang không gian ít chiều hơn, trong khi vẫn **giữ lại thông tin quan trọng**. Mục tiêu:
- Visualization (dữ liệu 2D/3D)
- Giảm chi phí tính toán
- Loại bỏ nhiễu và multicollinearity
- Tiền xử lý trước clustering

Hai kỹ thuật chính: **PCA (linear)** và **t-SNE (non-linear)**.

---

### **2. Thuật ngữ chính (Key Terms)**
- **Principal Component Analysis (PCA):** Tìm các trục chính (principal components) giữ lại phương sai lớn nhất.
- **Eigenvalue (Trị riêng):** Đo lượng phương sai giải thích bởi mỗi eigenvector.
- **Eigenvector (Vector riêng):** Hướng của principal component.
- **Covariance Matrix (Ma trận hiệp phương sai):** Ma trận đo mối quan hệ tuyến tính giữa các features.
- **Explained Variance Ratio:** Tỷ lệ phương sai được giữ lại bởi từng PC.
- **t-SNE (t-Distributed Stochastic Neighbor Embedding):** Kỹ thuật giảm chiều phi tuyến, giữ cấu trúc cục bộ.
- **Perplexity:** Tham số t-SNE, cân bằng giữa cấu trúc local và global.

---

### **3. Công thức toán học & Giải thích**

### **3.1. PCA (Principal Component Analysis)**

#### **Bước 1: Chuẩn hóa dữ liệu**
$X_{std} = \frac{X - \mu}{\sigma}$

#### **Bước 2: Tính ma trận hiệp phương sai**
$\Sigma = \frac{1}{n-1}X^T X$

Với X đã được chuẩn hóa (mean = 0).

#### **Bước 3: Tìm eigenvalues và eigenvectors**

Giải phương trình:
$\Sigma v = \lambda v$

Trong đó:
- **λ (lambda):** Eigenvalue (phương sai dọc theo PC)
- **v:** Eigenvector (hướng của PC)

#### **Bước 4: Chọn k eigenvectors lớn nhất**

Sắp xếp eigenvalues: λ₁ ≥ λ₂ ≥ ... ≥ λₙ

Tạo ma trận projection:
$W = [v_1, v_2, ..., v_k]$

#### **Bước 5: Transform dữ liệu**
$X_{new} = X \cdot W$

X_new có k chiều thay vì n chiều ban đầu.

#### **Explained Variance Ratio:**
$\text{EVR}_i = \frac{\lambda_i}{\sum_{j=1}^{n}\lambda_j}$

Tổng EVR của k PCs đầu tiên cho biết % thông tin giữ lại.

---

### **3.2. t-SNE**

#### **Bước 1: Tính conditional probability trong không gian ban đầu**

Xác suất điểm j là neighbor của i:
$p_{j|i} = \frac{\exp(-\lVert x_i - x_j \rVert^2 / 2\sigma_i^2)}{\sum_{k \neq i}\exp(-\lVert x_i - x_k \rVert^2 / 2\sigma_i^2)}$

Symmetric probability:
$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$

#### **Bước 2: Tính probability trong không gian thấp chiều (2D/3D)**

Dùng t-distribution (heavy tail):
$q_{ij} = \frac{(1 + \lVert y_i - y_j \rVert^2)^{-1}}{\sum_{k \neq l}(1 + \lVert y_k - y_l \rVert^2)^{-1}}$

#### **Bước 3: Minimize Kullback-Leibler divergence**

$KL(P||Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$

Dùng gradient descent để tối ưu vị trí các điểm y trong không gian 2D/3D.

---

### **4. Ví dụ minh hoạ**

#### **4.1. PCA - Dataset 3D → 2D**

**Dữ liệu ban đầu:** X có 3 features (x₁, x₂, x₃), 100 samples.

**Sau chuẩn hóa và tính eigenvalues:**
- λ₁ = 5.2 → EVR₁ = 65%
- λ₂ = 2.1 → EVR₂ = 26%
- λ₃ = 0.7 → EVR₃ = 9%

**Chọn k=2:** Giữ lại 91% thông tin (65% + 26%).

**Kết quả:** X_new có 2 chiều (PC1, PC2).

#### **4.2. t-SNE - MNIST digits**

**Dataset:** 784 features (28×28 pixels), 10 classes (digits 0-9).

**PCA vs t-SNE:**
- **PCA 2D:** Các class chồng lên nhau, khó phân biệt
- **t-SNE 2D (perplexity=30):** 10 cụm rõ ràng tương ứng 10 chữ số

**Lý do:** t-SNE giữ cấu trúc non-linear tốt hơn PCA.

---

### **5. Ghi chú và lưu ý**

#### **5.1. So sánh PCA vs t-SNE**

| Tiêu chí | PCA | t-SNE |
|---------|-----|-------|
| **Tính chất** | Linear | Non-linear |
| **Mục tiêu** | Maximize variance | Preserve local structure |
| **Tốc độ** | Nhanh O(n×d²) | Chậm O(n²) |
| **Interpretability** | PCs có ý nghĩa | Không interpret được |
| **Deterministic** | Yes | No (random init) |
| **Use case** | Preprocessing, feature extraction | Visualization |

#### **5.2. Khi nào dùng PCA?**
 Dữ liệu có cấu trúc tuyến tính
 Cần giảm chiều nhanh (preprocessing cho ML)
 Muốn giữ lại phương sai lớn nhất
 Cần interpret được các PC

#### **5.3. Khi nào dùng t-SNE?**
 Visualization dữ liệu cao chiều
 Dữ liệu có cấu trúc phi tuyến phức tạp
 Muốn thấy rõ clusters trong 2D/3D
 Không cần interpret các trục

#### **5.4. Lưu ý khi dùng t-SNE**
- **Perplexity:** Thường chọn 5-50, dataset lớn dùng 30-50
- **Learning rate:** Thường 10-1000, điều chỉnh nếu không hội tụ
- **Random:** Chạy nhiều lần với seed khác nhau
- **Không dùng để transform data mới** (chỉ dùng visualization)
- **Scale dữ liệu trước:** Standardization quan trọng

#### **5.5. Biến thể và mở rộng**
- **Incremental PCA:** Xử lý dữ liệu lớn không vừa RAM
- **Kernel PCA:** PCA phi tuyến dùng kernel trick
- **UMAP:** Tương tự t-SNE nhưng nhanh hơn, preserve global structure
- **Autoencoders:** Deep learning approach cho giảm chiều

---

### **6. Key Takeaways**
 PCA: giảm chiều tuyến tính, giữ phương sai lớn nhất  
 Chọn k PCs dựa trên Explained Variance Ratio (thường ≥85%)  
 t-SNE: giảm chiều phi tuyến, giữ cấu trúc local  
 t-SNE chỉ dùng visualization, không transform data mới  
 PCA nhanh và interpret được, t-SNE chậm nhưng visualization tốt hơn  
 Luôn chuẩn hóa dữ liệu trước khi giảm chiều  

---

