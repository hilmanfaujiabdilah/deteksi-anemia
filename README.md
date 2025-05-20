# Laporan Proyek Machine Learning - Hilman Fauji Abdilah

## Domain Proyek

Pada tahun 2019, prevalensi anak usia 6-59 bulan yang terkana anemia mencapai sekitar 39,8% dan 269 juta anak yang ikut terpengaruh. Secara global, masalah anemia pada ini menjadi masalah yang kesehatan yang besar [1]. Angka ini mungkin bahkan lebih tinggi di negara-negara dengan pendapatan rendah dan menengah, seperti Bangladesh. Ini disebabkan oleh masalah gizi yang beragam, infeksi, dan keterbatasan akses layanan kesehatan [2]. Jika anemia anak tidak diobati, dapat menyebabkan gangguan perkembangan fisik dan kognitif yang tidak dapat diperbaiki, yang berdampak pada produktivitas di masa depan [1]. Oleh karena itu, pengembangan model prediksi berbasis indikator hematologi (Hb, RBC, PCV, MCV, MCH, dan MCHC) sangat penting untuk deteksi dini dan intervensi cepat dalam lingkungan klinis dan masyarakat.

**Alasan dan urgensi penyelesaian masalah**:
Anemia kronis pada masa kanak-kanak berisiko menimbulkan keterlambatan perkembangan motorik dan kognitif yang bersifat ireversibel, memengaruhi performa akademik dan produktivitas dewasa [1]. Gangguan perkembangan ini menimbulkan beban ekonomi dan sosial yang besar bagi keluarga dan sistem kesehatan. Akan tetapi pemeriksaan lengkap (CBC) memerlukan peralatan laboratorium yang tidak selalu tersedia di daerah terpencil atau klinik primer. Keterlambatan diagnosis berpotensi memperparah kondisi sebelum intervensi dapat diberikan. Berdasarkan permasalahan tersebut, perlu dibuat sebuah solusi praktis untuk membantu masyarakat dalam mendeteksi gejala anemia sejak dini yang dapat di akses dengan mudah dan tidak menimbulkan biaya yang besar. oleh karena itu dibuatlah proyek pengembangan machine learning ini untuk memprediksi anemia berdasrakan indikator hemagtologikal.

**Hasil riset terkait:**
1. Zemariam et al. (2024)
   - Menerapkan berbagai algoritma supervised ML pada DHS Ethiopia untuk memprediksi anemia pada remaja putri, membandingkan kinerja model dan menyoroti potensi Random Forest sebagai yang paling efektif [3]
2. Prakriti Dhakal et al. (2022)
   - Riset di Nepal menggunakan 700 rekam data CBC, Random Forest terbaik (akurasi 98,4 %), diikuti dengan metode ensemble (voting, stacking, boosting) yang memoles performa lebih jauh [4]
3. Birchak dkk. (2024)
   - Model berbasis CNN, k‑NN, Naïve Bayes, Decision Tree, dan SVM untuk deteksi anemia anak, menyimpulkan kombinasi ensemble learning meningkatkan sensitivitas dan spesifisitas [5]

## Business Understanding

Anemia pediatrik memiliki prevalensi tinggi mencapai hampir 40% pada anak usia 6 hingga 59 bulan. Namun, karena ketergantungan pada pemeriksaan laboratorium lengkap (CBC) yang tidak selalu tersedia di klinik primer, terutama di daerah berpendapatan rendah seperti Bangladesh, belum ada solusi prediksi terintegrasi dan mudah digunakan untuk membantu tenaga kesehatan non-spesialis melakukan skrining dini berdasarkan parameter hematologis dasar.

### Problem Statements
berikut adalah problem statement dari proyek ini:
- Tingginya Prevalensi Anemia Pediatrik dan Keterlambatan Deteksi
  Prevalensi anemia pada anak usia 6–59 bulan mencapai hampir 40% secara global, dengan angka lebih tinggi di wilayah berpendapatan rendah seperti Bangladesh. Keterlambatan deteksi sering terjadi karena screening konvensional yang bergantung pada pemeriksaan laboratorium lengkap (CBC), sehingga intervensi dini menjadi terhambat.
- Keterbatasan Akses dan Sumber Daya Laboratorium di Klinik Primer
  Banyak pusat pelayanan kesehatan primer di daerah terpencil belum dilengkapi peralatan laboratorium yang memadai untuk melakukan pemeriksaan CBC secara rutin. Hal ini menimbulkan disparitas diagnosis dan perawatan, terutama pada populasi anak di pedesaan.

### Goals

Berdasarkan masalah di atas, berikut adalah tujuan dari adanya proyek ini:
- Meminimalkan Waktu Deteksi Dini Anemia
  Mengembangkan model prediksi berbasis parameter hematologis (Hb, RBC, PCV, MCV, MCH, MCHC) sehingga dapat mendeteksi risiko anemia pada anak lebih awal, sebelum hasil laboratorium lengkap tersedia, dengan akurasi minimal 90%.
- Menghadirkan Solusi Ringkas untuk Klinik Primer
  Merancang prototipe aplikasi atau dashboard yang mudah digunakan, yang hanya memerlukan input parameter hematologis dasar untuk memprediksi status anemia. Solusi ini harus ramah sumber daya (low‑compute) agar dapat diadopsi di klinik dengan keterbatasan infrastruktur.
  

### Solution Statement:
- Pengujian Multi-Algoritma dengan Evaluasi Terukur
  Membangun dan membandingkan tiga model prediksi Logistic Regression (LR), Gradient Boosting Classifier (Boosting), dan Support Vector Machine (SVM, kernel RBF). Pada data pelatihan dan uji untuk mendapatkan gambaran awal kinerja masing‑masing model. Ukur dan bandingkan `accuracy_train` dan `accuracy_test` sebagai metrik pertama, serta analisis confusion matrix untuk setiap model agar dapat menilai kesalahan klasifikasi pada kelas anemia vs non‑anemia.
- Selain accuracy, menghitung nilai precision, recall, dan F1‑score pada data latih dan data uji untuk setiap model ter‑tuning. Hal ini memastikan model tidak hanya akurat secara keseluruhan, tetapi juga sensitif dalam mendeteksi kasus anemia (high recall) dan spesifik pada non‑anemia (high precision).

## Data Understanding

Dataset ini berisi informasi klinis pasien yang digunakan untuk mengklasifikasikan status anemia. Data dikumpulkan dari pemeriksaan laboratorium dan mencakup parameter hematologi utama.

- **Jumlah Sampel**: 1000 record data
- **Jumlah Fitur** : 8 (7 fitur prediktor + 1 target)
- **Sumber Data**   : [Link Dataset](https://data.mendeley.com/datasets/y7v7ff3wpj/1)

### Variabel-variabel pada Anemia Dataset adalah sebagai berikut:
1. **Age**            : Usia pasien dalam tahun (numerik, rentang 18-96)
2. **Gender**         : Jenis kelamin pasien (kategorikal: "Male" atau "Female")
3. **Hb**             : Hemoglobin (g/dL), indikator utama jumlah hemoglobin dalam darah
4. **RBC**            : Red Blood Cell Count (×10^6 sel/µL), jumlah sel darah merah
5. **MCV**            : Mean Corpuscular Volume (fL), rata-rata volume sel darah merah
6. **MCH**            : Mean Corpuscular Hemoglobin (pg), rata-rata berat hemoglobin per sel
7. **MCHC**           : Mean Corpuscular Hemoglobin Concentration (g/dL), konsentrasi hemoglobin per volume sel
8. **PCV**            : Packed Cell Volume (%), proporsi volume darah yang ditempati sel darah merah
9. **Decision_Class** : Label target (0 = Non-anemia, 1 = Anemia)

### Proses Exploratory Data Analysis:

Terdapat beberapa tahap yang dilakuakn dalam eksploratory data analysis diantaranya:

1. Mengecek tipe data dan statistik deskriptif
   Dengan menggunakan fungsi `df.info()` untuk mengecek tipe data tiap kolom dan jumlah entri non-null. Ini berguna untuk mengidentifikasi apakah ada data kosong atau kolom bertipe data yang tidak sesuai.
2. Mengecek analisis deskriptif
   Menggunakan `df.describe(include='all')` untuk mendapatkan nilai-nilai statistik seperti rata-rata (mean), median, nilai minimum dan maksimum, serta kuartil. Statistik ini membantu memahami skala data, sebaran nilai, dan mendeteksi ketidakwajaran (anomali awal).
3. Analysis Outliers
   Menggunakan boxplot per fitur numerik untuk mengidentifikasi nilai outlier. Menghitung IQR (Q3–Q1) dan menghapus baris yang berada di luar [Q1–1.5×IQR, Q3+1.5×IQR] pada kolom numerik. Hal ini dilakukan karena nilai ekstream dapat mempengaruhi rata-rata dan variansi, serta mengarahkan model untuk overfit pada data unik.
3. Melakukan EDA Univariate Analysis
   - Distribusi numerik: Menggunakan histogram untuk semua fitur numerik seperti Age, Hb, RBC, dll., guna memahami distribusi dan skewness. Mayoritas fitur menunjukkan distribusi normal atau hampir normal.
   - Distribusi kategorikal: Gender ditampilkan dalam bentuk bar plot untuk melihat proporsi laki-laki dan perempuan. Persentase distribusi gender juga dihitung.
4. Melakukan EDA Multivariate Analysis
   - Pairplot: Digunakan untuk mengeksplorasi relasi antar fitur numerik dan potensi klasterisasi antar kelas target (anemia vs non-anemia).
   - Heatmap korelasi: Menggunakan Pearson correlation untuk mengukur kekuatan hubungan linier antar fitur. Teridentifikasi bahwa PCV dan MCV sangat berkorelasi dengan fitur lain, sehingga dihapus untuk mengurangi multikolinearitas.

## Data Preparation
Dalam tahapan data preparation, langkah-langkah berikut dilakukan secara berurutan:

1. Encoding Kategorikal
   Pada langkah ini dilakukan untuk mengubah Gender menjadi format numerik (0/1) dengan LabelEncoder. Hal tersebut dilakukan karena algoritma ML memerlukan input numerik agar dapat menghitung jarak (SVM) atau pembagian (tree-based).
2. Feature Selection
   Proses ini menghapus PCV dan MCV karena korelasi tinggi (>0.9) dengan Hb dan RBC. alasannya karena fitur yang sangat berkorelasi menyebabkan multikollinearitas, mengurangi stabilitas koefisien pada model linier dan ensembling.
3. Train-Test Split
   pada tahap ini memisahkan data dengan rasio 80% training dan 20% testing (random_state=42). rasio ini memberikan cukup data untuk melatih model sekaligus menyisakan cukup sampel yang representatif untuk mengevaluasi performa. Rasio 80:20 menjaga keseimbangan antara kebutuhan data belajar dan kebutuhan validasi model secara generalisasi.
4. Feature Scaling
   StandardScaler dilakukan pada kolom Age, Hb, RBC, MCH, MCHC (yang akan merubah data sehingga memiliki skala mean=0 dan std=1). Hal ini dilakukan untuk menghindari dominasi fitur berskala besar pada algoritma yang peka skala (Logistic Regression, SVM).

## Modeling
Algoritma yang digunakan dalam proyek ini diantaranya
1. Logistic Regression (LR)
   - Kelebihan: Sederhana, cepat, interpretatif.
   - Kekurangan: Lemah menangani non-linearitas.
2. Gradient Boosting Classifier (GBC)
   - Kelebihan: Akurat, tangguh terhadap noise.
   - Kekurangan: Butuh tuning, komputasi berat.
3. Support Vector Machine (SVM)
   - Kelebihan: Baik untuk data berdimensi tinggi.
   - Kekurangan: Sensitif terhadap parameter, lambat jika data besar.
 
Berikut adalah parameter yang digunakan dalam ketiga model
|  Model | Parameter | 
| ------ | ------------- |
| LR     |    default    |
|  GBC   |    default    |
| SVM    | C=1.0, gamma='scale' |


Berdasarkan evaluasi terhadap tiga algoritma yang digunakan (Logistic Regression, Gradient Boosting, dan SVM), model terbaik yang dipilih adalah Gradient Boosting Classifier. Hal ini didasarkan pada hasil evaluasi pada data uji, di mana Gradient Boosting mencapai skor sempurna (100%) untuk semua metrik utama, yaitu accuracy, precision, recall, dan F1-score. Tidak hanya unggul dalam performa prediksi, model ini juga menunjukkan generalisasi yang sangat baik tanpa indikasi overfitting berlebih, sebagaimana terlihat dari konsistensi performa antara data latih dan data uji. Oleh karena itu, Gradient Boosting dipilih sebagai solusi utama dalam mendeteksi anemia pada dataset ini karena memberikan hasil yang paling andal dan stabil dibandingkan model lainnya.

## Evaluation

Pada tahap evaluasi, metrik utama yang digunakan adalah accuracy, precision, recall, dan F1-score. Keempat metrik ini dirumuskan sebagai:

- **Accuracy:** Proporsi prediksi yang benar terhadap total sampel.

  ![image](https://github.com/user-attachments/assets/183dbd64-62f9-494f-a8d0-e30de96c939d)

- **Precision:** Proporsi prediksi positif yang benar (exactness).

  ![image](https://github.com/user-attachments/assets/2ecc9dd5-e562-4bc9-8587-cb67b6f157c5)


- **Recall (Sensitivity):** Proporsi sampel positif yang berhasil terdeteksi (completeness) 

  ![image](https://github.com/user-attachments/assets/7aac35bb-f476-40ce-975c-78db54859aa2)

- **F1-score:** Harmonis antara precision dan recall, menyajikan satu angka performa positif. 

  ![image](https://github.com/user-attachments/assets/475d09fb-f8cc-41ff-b8c9-1430db53325f)

**Keterangan**:  
- TP = True Positive  
- TN = True Negative  
- FP = False Positive  
- FN = False Negative

Selain itu, confusion matrix digunakan untuk meninjau distribusi true positive (TP), false positive (FP), true negative (TN), dan false negative (FN) pada data uji.

### Hasil evaluasi

| Model        | Train Acc | Test Acc | Train Prec | Test Prec | Train Rec | Test Rec | Train F1 | Test F1 |
| ------------ | --------- | -------- | ---------- | --------- | --------- | -------- | -------- | ------- |
| **LR**       | 0.994     | 0.982    | 0.998      | 1.000     | 0.993     | 0.974    | 0.995    | 0.987   |
| **Boosting** | 1.000     | 1.000    | 1.000      | 1.000     | 1.000     | 1.000    | 1.000    | 1.000   |
| **SVM**      | 0.988     | 0.965    | 0.995      | 0.991     | 0.986     | 0.957    | 0.991    | 0.974   |

- Accuracy: Boosting sempurna (100%) pada data uji, diikuti LR 98.2% dan SVM 96.5%.
- Precision: Semua model sangat tepat; Boosting 100%.
- Recall: Boosting mendeteksi semua kasus anemia (100%), sedangkan LR dan SVM masing‑masing 97.4% dan 95.7%.
- F1-score: Boosting memimpin dengan 1.000, menunjukkan keseimbangan sempurna antara precision dan recall.

### Analisis confusion matrix
Berikut adalah hasil dari confusion matrix:
- Logistic Regression:
  - TN = 55, FP = 0, FN = 3, TP = 113
  - Terdapat 3 kasus anemia yang gagal terdeteksi.
- Boosting:
  - TN = 55, FP = 0, FN = 0, TP = 116
  - Tanpa kesalahan prediksi.
- SVM:
  - TN = 54, FP = 1, FN = 5, TP = 111
  - Terdapat 1 FP dan 5 FN.

### Kesimpulan evaluasi
Model Gradient Boosting menunjukkan performa terbaik dan paling konsisten di semua metrik, dengan accuracy, precision, recall, dan F1-score = 1.000 pada data uji. Konsistensi performa antara train dan test tanpa penurunan signifikan menandakan minimnya overfitting, sehingga model ini direkomendasikan sebagai solusi utama untuk prediksi anemia pediatrik.

**Catatan**: 

Visualisasi distribusi data, heatmap korelasi, dan confusion matrix tersedia pada notebook untuk mendukung interpretasi data dan performa model.


**Referensi:**

[1] 	V. Martinez-Torres, N. Torres, J. A. Davis dan F. F. Corrales-Medina, “Anemia and   Associated Risk Factors in Pediatric Patients,” Pediatric Health, Medicine and Therapeutics, pp. 267-280, 2023. 
[2] 	G. A. Stevens, C. J. Paciorek, M. C. Flores-Urrutia, E. Borghi, S. Namaste, J. P. Wirth, P. S. Suchdev, M. Ezzati, F. Rohner, S. R. Flaxman dan L. M. Rogers, “National, regional, and global estimates of anaemia by severity in women and children for 2000–19: a pooled analysis of population-representative data,” 23 April 2022. [Online]. Available: https://www.thelancet.com/action/showPdf?pii=S2214-109X%2822%2900084-5. [Diakses 20 Mei 2025].
[3] 	A. B. Zemariam, A. Yimer, G. K. Abebe, W. T. Wondie, B. B. Abate, A. W. Alamaw, G. Yilak, T. M. Melaku dan H. S. Ngusie, “Employing supervised machine learning algorithms for classification and prediction of anemia among youth girls in Ethiopia,” Scientific Reports, 20 April 2024. [Online]. Available: https://www.nature.com/articles/s41598-024-60027-4?. [Diakses 20 Mei 2025].
[4] 	P. Dhakal, S. Khanal dan R. Bista, “Prediction of Anemia Using Machine Learning Algorithms,” International Journal of Computer Science & Information Technology (IJCSIT), vol. XV, no. 1, pp. 15-30, 2023. 
[5] 	J. W. Asare, W. L. Brown-Acquaye, M. M. Ujakpa, E. Freeman dan P. Appiahene, “Application of machine learning approach for iron deficiency anaemia detection in children using conjunctiva images,” Informatics in Medicine Unlocked , pp. 1-14, 2024. 
