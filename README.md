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
- Mengembangkan Model Prediksi Anemia yang Akurat
  Membangun sebuah model klasifikasi machine learning yang mampu memprediksi status anemia pada anak (anemia atau non-anemia) berdasarkan indikator hematologis (Hb, RBC, MCH, MCHC, dan Age, serta Gender) dengan target akurasi minimal 90% pada data uji.
- Mengidentifikasi Algoritma Klasifikasi Terbaik
  Melakukan evaluasi komparatif terhadap beberapa algoritma machine learning (seperti Logistic Regression, Gradient Boosting, dan SVM) untuk menentukan model dengan performa terbaik dalam hal akurasi, presisi, recall, dan F1-score, khususnya dengan sensitivitas tinggi (recall) untuk mendeteksi kasus anemia.
- Menghasilkan Model yang Dapat Diimplementasikan
  Mengembangkan model prediktif yang tidak hanya akurat tetapi juga cukup efisien untuk potensi implementasi pada sistem atau aplikasi sederhana, guna mendukung deteksi dini anemia di fasilitas dengan sumber daya terbatas.

### Solution Statement:
Untuk mencapai tujuan-tujuan tersebut, solusi yang diajukan adalah pengembangan sistem prediksi anemia melalui tahapan berikut:
- Pengembangan dan Evaluasi Model
  - Akan dibangun dan dilatih tiga model klasifikasi utama: Logistic Regression (LR), Gradient Boosting Classifier (Boosting), dan Support Vector Machine (SVM) menggunakan data hematologis yang telah diproses.
  - Kinerja setiap model akan dievaluasi secara komprehensif pada data pelatihan dan data pengujian menggunakan metrik standar: akurasi, presisi, recall, dan F1-score. Analisis confusion matrix juga akan dilakukan untuk memahami secara detail bagaimana masing-masing model mengklasifikasikan kasus anemia dan non-anemia, termasuk identifikasi jenis kesalahan yang dibuat.
- Pemilihan Model Optimal
   - Model terbaik akan dipilih berdasarkan perbandingan skor metrik evaluasi, dengan penekanan khusus pada test_accuracy yang tinggi dan skor recall yang optimal untuk meminimalkan risiko kasus anemia yang tidak terdeteksi. Pertimbangan juga akan diberikan pada keseimbangan antara presisi dan recall (F1-score).
   - Model terpilih diharapkan dapat memberikan prediksi status anemia secara efektif, menjadi dasar untuk solusi deteksi dini yang praktis.

## Data Understanding

Tahap ini bertujuan untuk memahami karakteristik dataset yang digunakan dalam proyek ini sebelum dilakukan pemrosesan lebih lanjut. Pemahaman ini meliputi struktur data, konten, kualitas awal, serta analisis eksploratif untuk mendapatkan wawasan awal.

### Sumber Data
Dataset yang digunakan dalam proyek ini adalah "Anemia Prediction Dataset" yang bersumber dari Mendeley Data. Dataset ini berisi data klinis dan parameter hematologis pasien yang relevan untuk prediksi anemia.
Tautan sumber data : [Link](https://data.mendeley.com/datasets/y7v7ff3wpj/1)

### Deskripsi Data
- Jumlah Baris dan Kolom: Dataset awal terdiri dari 1000 baris (sampel) dan 9 kolom (fitur). Kolom-kolom ini mencakup 8 fitur prediktor dan 1 fitur target ('Decision_Class').
- Uraian Seluruh Fitur pada Data:
  - Gender: Jenis kelamin pasien (kategorikal: nilai asli 'm' untuk Male, 'f' untuk Female).
  - Age: Usia pasien dalam tahun (numerik).
  - Hb: Kadar Hemoglobin dalam darah (g/dL), indikator utama anemia (numerik).
  - RBC: Jumlah Sel Darah Merah (Red Blood Cell Count) (×10^6 sel/µL) (numerik).
  - PCV: Packed Cell Volume atau Hematokrit (%), proporsi volume darah yang ditempati sel darah merah (numerik).
  - MCV: Mean Corpuscular Volume (fL), volume rata-rata sel darah merah (numerik).
  - MCH: Mean Corpuscular Hemoglobin (pg), berat rata-rata hemoglobin per sel darah merah (numerik).
  - MCHC: Mean Corpuscular Hemoglobin Concentration (g/dL), konsentrasi hemoglobin rata-rata per unit volume sel darah merah (numerik).
  - Decision_Class: Fitur target yang mengindikasikan status anemia pasien (biner: 0 = Non-Anemia, 1 = Anemia).

### Kondisi Data Awal

Pemeriksaan awal terhadap kualitas data dilakukan untuk mengidentifikasi potensi masalah:
1. Missing Values: Pemeriksaan menggunakan df.info() menunjukkan bahwa tidak terdapat nilai yang hilang (missing values) pada semua kolom dalam dataset. Setiap kolom memiliki 1000 entri non-null.
2. Data Duplikat: Pemeriksaan menggunakan df.duplicated().sum() menunjukkan adanya data duplikat dalam dataset sebanyak 28 baris.
3. Outlier (Observasi Awal):

   ![outlier](https://github.com/user-attachments/assets/62c35be7-5379-425b-85a2-68928e39c6b7)
   Visualisasi awal menggunakan boxplot untuk setiap fitur numerik mengindikasikan adanya beberapa titik data yang berpotensi menjadi outlier. Fitur seperti 'Age', 'Hb', 'RBC', 'PCV', 'MCV', 'MCH', dan 'MCHC' menunjukkan adanya nilai-nilai yang terletak jauh dari sebagian besar distribusi datanya. Penanganan lebih lanjut terhadap outlier ini akan dibahas pada tahap Data Preparation.

### Exploratory Data Analysis (EDA)
Analisis data eksploratif dilakukan untuk lebih memahami karakteristik dan pola dalam data:
1. Statistik Deskriptif

   ![image](https://github.com/user-attachments/assets/3938f955-155f-4b76-8836-21858b4b20bc)
   Menggunakan `df.describe()` untuk mendapatkan ringkasan statistik (seperti mean, median, standar deviasi, min, max, kuartil) untuk fitur numerik. memberikan gambaran mengenai sebaran dan tendensi sentral masing-masing fitur. Sebagai contoh, rentang usia pasien teridentifikasi antara 18 hingga 96 tahun.
3. Analisis Univariat
   - Distribusi Fitur Numerik

     ![univar num](https://github.com/user-attachments/assets/88a98385-8fca-428a-9945-08ba8513279d)
     Observasi menunjukkan bahwa sebagian besar fitur hematologis seperti 'RBC', 'PCV', 'MCV', dan 'MCHC' memiliki distribusi yang mendekati normal, meskipun beberapa menunjukkan sedikit kemiringan (skewness). Fitur 'Age' menunjukkan distribusi yang lebih tersebar, sementara 'Hb' tampak sedikit miring ke kiri.
   - Distribusi Fitur Kategorikal

     ![univar gender](https://github.com/user-attachments/assets/c38f4c5f-3d27-495e-8f2a-829c9eb7c4ee)
     Observasi menunjukkan bahwa sebagian besar fitur hematologis seperti 'RBC', 'PCV', 'MCV', dan 'MCHC' memiliki distribusi yang mendekati normal, meskipun beberapa menunjukkan sedikit kemiringan (skewness). Fitur 'Age' menunjukkan distribusi yang lebih tersebar, sementara 'Hb' tampak sedikit miring ke kiri.
4. Analisis Multivariat
   - Pairplot

     ![multi pair](https://github.com/user-attachments/assets/73b7a86d-f2a5-4b27-9eb7-c898091204d8)
     `sns.pairplot(df, diag_kind='kde')` digunakan untuk memvisualisasikan hubungan antar pasangan fitur numerik dan distribusi individual setiap fitur. Dari scatter plot, teridentifikasi adanya korelasi positif yang kuat antara beberapa pasang fitur hematologis, misalnya antara 'Hb' dengan 'RBC' dan 'PCV', serta antara 'RBC' dan 'PCV'. Korelasi yang sangat kuat juga terlihat antara 'MCV' dan 'MCH'.
   - Heatmap Korelasi

     ![image](https://github.com/user-attachments/assets/baaa6b90-7987-4abd-bb7f-e1d6b62b7a3e)
      Hasil heatmap mengonfirmasi korelasi tinggi (nilai mendekati +1) antara 'PCV' dan 'Hb', 'PCV' dan 'RBC', serta 'MCV' dan 'MCH'. Observasi ini menjadi dasar pertimbangan untuk tahap feature selection guna mengurangi multikolinearitas.
   
## Data Preparation
Setelah memahami karakteristik data, tahap data preparation dilakukan untuk membersihkan, mentransformasi, dan menyiapkan data agar sesuai untuk proses pemodelan machine learning. Langkah-langkah yang dilakukan secara berurutan adalah sebagai berikut:

1. Penanganan Outlier
   - Berdasarkan identifikasi outlier pada tahap EDA menggunakan boxplot, dilakukan penanganan outlier dengan metode IQR (Interquartile Range) pada fitur-fitur numerik.
   - Penghapusan outlier bertujuan untuk meningkatkan robustisitas model dan mencegah pengaruh yang tidak proporsional dari nilai-nilai ekstrem pada hasil analisis dan kinerja model.
   - Setelah proses ini, jumlah sampel dalam dataset berkurang dari 1000 menjadi 827 sampel.
2. Feature Selection
   - Berdasarkan analisis korelasi pada tahap EDA yang menunjukkan multikolinearitas tinggi, fitur 'MCV' dan 'PCV' dihapus dari dataset.
   - Kedua fitur ini memiliki korelasi yang sangat kuat dengan fitur hematologis lainnya (seperti 'MCV' dengan 'MCH', dan 'PCV' dengan 'Hb' dan 'RBC'). Penghapusan ini bertujuan untuk mengurangi redundansi informasi, menyederhanakan model, dan meningkatkan stabilitasnya.
   - Dataset kini memiliki fitur prediktor: 'Gender', 'Age', 'Hb', 'RBC', 'MCH', dan 'MCHC'.
3. Encoding Variabel Kategorikal
   - Fitur 'Gender', yang merupakan variabel kategorikal, diubah menjadi representasi numerik menggunakan `LabelEncoder` dari scikit-learn. Dimana, 'f' di encode menjadi 0 dan 'm' menjadi 1
4. Pembagian Data (Train-Test Split)
   - Dataset dibagi menjadi fitur independen (X) dan fitur target (y, yaitu 'Decision_Class').
   - Data dibagi lagi menjadi set pelatihan (`X_train`, `y_train`) dan set pengujian (`X_test`, `y_test`) dengan rasio 80% untuk data pelatihan dan 20% untuk data pengujian. Parameter `random_state=42` digunakan untuk memastikan reproduktifitas pembagian.
   - Dari 827 sampel (setelah penanganan outlier), 661 sampel dialokasikan untuk data pelatihan dan 166 sampel untuk data pengujian.
5. Penskalaan Fitur (Feature Scaling)
   - Standarisasi diterapkan pada fitur-fitur numerik dalam set pelatihan (`X_train`), yaitu 'Age', 'Hb', 'RBC', 'MCH', dan 'MCHC', menggunakan `StandardScaler` dari scikit-learn.
   - Kemudian, transformasi menggunakan scaler yang telah di-fit tersebut diterapkan pada `X_train` dan juga pada `X_test`. Sehingga fitur-fitur numerik yang ditentukan dalam X_train dan X_test dapat memiliki skala yang seragam.

## Modeling
Pada tahap ini, dikembangkan tiga model machine learning untuk prediksi anemia, yaitu Logistic Regression, Gradient Boosting Classifier, dan Support Vector Machine.
1. Logistic Regression (LR)
   - Logistic Regression adalah model linear yang memprediksi probabilitas kelas biner menggunakan fungsi sigmoid. Model ini menentukan batas keputusan linear berdasarkan kombinasi fitur input.
   - Model ini menggunakan parameter default dari scikit-learn, yang umumnya mencakup regularisasi L2 (`penalty='l2'`) dan solver seperti 'lbfgs'
   - Kelebihan: Sederhana, cepat dilatih, mudah diinterpretasikan.
   - Kekurangan: Kurang efektif untuk hubungan non-linear, rentan underfitting pada data kompleks.
2. Gradient Boosting Classifier (GBC)
   - Gradient Boosting adalah teknik ensemble yang membangun model (biasanya decision tree) secara sekuensial. Setiap model baru berfokus pada perbaikan kesalahan yang dibuat oleh model-model sebelumnya, menggunakan optimasi gradien.
   - Model ini menggunakan parameter default dari scikit-learn
   - Kelebihan: Akurasi tinggi, dapat menangani hubungan kompleks.
   - Kekurangan: Membutuhkan tuning, komputasi bisa lebih berat, kurang interpretatif.
3. Support Vector Machine (SVM)
   - SVM mencari hyperplane optimal yang memaksimalkan margin (jarak) antara kelas data. Untuk data non-linear, SVM menggunakan fungsi kernel (seperti RBF) untuk memetakan data ke ruang dimensi yang lebih tinggi agar pemisahan linear dapat dilakukan.
   - Parameter yang digunakan adalah 'C=1.0' yaitu parameter regularisasi yang mengontrol trade-off antara margin dan kesalahan klasifikasi. gamma='scale' yaitu koefisien kernel (untuk RBF) yang menentukan jangkauan pengaruh satu sampel pelatihan. Kemudian diasumsikan menggunakan `kernel='rbf'` (default).
   - Kelebihan: Efektif di ruang dimensi tinggi, hemat memori (menggunakan support vectors).
   - Kekurangan: Sensitif terhadap parameter, bisa lambat pada data besar, interpretasi lebih sulit dengan kernel non-linear.

## Evaluation

Evaluasi model merupakan tahap krusial untuk mengukur sejauh mana model yang dikembangkan mampu menjawab permasalahan dan mencapai tujuan proyek. Pada tahap ini, metrik utama yang digunakan adalah accuracy, precision, recall, dan F1-score.

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

![image](https://github.com/user-attachments/assets/e6eb869e-4d12-47d8-bcfb-638834db0ccb)

Ringkasan Performa pada Data Uji:
- Akurasi: Model Gradient Boosting mencapai akurasi sempurna (1.000), diikuti oleh LR (0.982) dan SVM (0.970).
- Presisi: Semua model menunjukkan presisi yang sangat tinggi. LR dan Gradient Boosting mencapai presisi sempurna (1.000), sementara SVM mencapai 0.981.
- Recall: Gradient Boosting kembali unggul dengan recall sempurna (1.000), yang berarti berhasil mendeteksi semua kasus anemia aktual. LR mencapai recall 0.971, dan SVM 0.971.
- F1-score: Gradient Boosting memimpin dengan skor 1.000, menunjukkan keseimbangan sempurna antara presisi dan recall. LR dan SVM masing-masing mencatatkan F1-score 0.985 dan 0.976.
Secara keseluruhan, model Gradient Boosting Classifier (Boosting) menunjukkan superioritas di semua metrik evaluasi pada data uji.

### Analisis confusion matrix
Berikut adalah hasil dari confusion matrix:

![conf m](https://github.com/user-attachments/assets/96b06795-eff7-4090-bd3b-32d85d751587)

- Logistic Regression:
  - TN = 55, FP = 0, FN = 3, TP = 113
  - Terdapat 3 kasus anemia yang gagal terdeteksi.
- Boosting:
  - TN = 55, FP = 0, FN = 0, TP = 116
  - Tanpa kesalahan prediksi.
- SVM:
  - TN = 54, FP = 1, FN = 5, TP = 111
  - Terdapat 1 FP dan 5 FN.

### Hasil Evaluasi dan Dampaknya terhadap Tujuan Proyek
Evaluasi kinerja model tidak hanya berhenti pada angka, tetapi juga bagaimana hasil ini menjawab kebutuhan dan tujuan awal proyek.

1. Pencapaian Tujuan (Goals) Proyek
   - Tujuan 1: Mengembangkan Model Prediksi Anemia yang Akurat (Akurasi min 90%)
     Tercapai. Ketiga model yang dikembangkan berhasil melampaui target akurasi minimal 90% pada data uji. Model Gradient Boosting bahkan mencapai akurasi 100%, Logistic Regression 98.2%, dan SVM 96.5%. Ini menunjukkan bahwa model prediksi yang dikembangkan sangat akurat dalam mengklasifikasikan status anemia.
   - Tujuan 2: Mengidentifikasi Algoritma Klasifikasi Terbaik
     Tercapai. Melalui evaluasi komparatif, Gradient Boosting Classifier berhasil diidentifikasi sebagai algoritma terbaik. Model ini unggul secara signifikan di semua metrik evaluasi (akurasi, presisi, recall, dan F1-score) pada data uji, dengan skor sempurna (1.000). Tingginya skor recall (1.000) sangat krusial, menandakan tidak ada kasus anemia yang terlewat.
   - Tujuan 3: Menghasilkan Model yang Dapat Diimplementasikan
     Berpotensi Tercapai. Model Gradient Boosting, meskipun mungkin memerlukan sumber daya komputasi yang lebih besar saat pelatihan dibandingkan LR, umumnya memiliki waktu prediksi yang cepat setelah model dilatih. Kinerja prediksinya yang superior menjadikannya kandidat kuat untuk implementasi. Efisiensi implementasi akhir akan bergantung pada platform target, namun akurasi dan reliabilitasnya yang tinggi sangat mendukung potensinya untuk digunakan di lingkungan klinis, termasuk yang memiliki sumber daya terbatas, jika infrastruktur inferensi dapat dioptimalkan.
   
2. Jawaban Terhadap Problem Statements
   - Problem Statement 1: Tingginya Prevalensi Anemia Pediatrik dan Keterlambatan Deteksi
     Model prediksi yang dikembangkan, khususnya Gradient Boosting dengan akurasi dan recall 100%, secara langsung menjawab masalah keterlambatan deteksi. Dengan menggunakan parameter hematologis dasar, model ini berpotensi memungkinkan skrining awal yang cepat dan akurat, bahkan sebelum hasil pemeriksaan laboratorium lengkap (CBC) tersedia atau jika CBC tidak dapat segera dilakukan. Ini membuka jalan untuk intervensi yang lebih dini.
   - Problem Statement 2: Keterbatasan Akses dan Sumber Daya Laboratorium di Klinik Primer
     Model ini bergantung pada beberapa parameter hematologis yang mungkin dapat diperoleh dari alat pemeriksaan yang lebih sederhana atau sebagai bagian dari pemeriksaan darah parsial, dibandingkan dengan kebutuhan CBC lengkap. Meskipun pengumpulan parameter input tetap memerlukan fasilitas laboratorium dasar, model ini dapat membantu tenaga kesehatan di klinik primer membuat penilaian risiko anemia dengan lebih percaya diri dan cepat, mengoptimalkan penggunaan sumber daya yang terbatas, dan memprioritaskan kasus yang memerlukan rujukan atau pemeriksaan lebih lanjut.

3. Dampak Implementasi Solusi yang Direncanakan (Solution Statements)
   - Solusi 1: Pengembangan dan Evaluasi Model (LR, GBC, SVM dengan metrik komprehensif)
     Berhasil Diimplementasikan dan Berdampak Positif. Pendekatan untuk membangun, melatih, dan mengevaluasi ketiga model (LR, GBC, SVM) menggunakan serangkaian metrik yang komprehensif (akurasi, presisi, recall, F1-score, dan confusion matrix) telah berhasil dilaksanakan. Dampaknya adalah diperolehnya pemahaman yang mendalam tentang kekuatan dan kelemahan masing-masing model. Penggunaan metrik yang beragam, terutama recall, memastikan bahwa aspek kritis dari deteksi penyakit (meminimalkan kasus terlewat) menjadi pertimbangan utama.
   - Solusi 2: Pemilihan Model Optimal (berdasarkan akurasi uji dan recall tinggi)
     Berhasil Diimplementasikan dan Berdampak Signifikan. Proses pemilihan model optimal berhasil mengidentifikasi Gradient Boosting sebagai model superior. Keputusan ini didasarkan pada kriteria yang ditetapkan, yaitu performa terbaik pada data uji dengan penekanan pada akurasi dan recall. Dampaknya adalah terpilihnya model yang paling andal dan paling sesuai dengan kebutuhan domain masalah, yaitu model yang tidak hanya akurat secara keseluruhan tetapi juga sangat sensitif dalam mendeteksi kasus anemia (recall 100% pada data uji).

4. Dampak Model terhadap Pemahaman Bisnis (Business Understanding)
   Pengembangan model machine learning dalam proyek ini, khususnya dengan performa luar biasa dari model Gradient Boosting, memberikan kontribusi signifikan terhadap pemahaman dan solusi masalah anemia pediatrik.
   - Validasi Potensi Deteksi Dini. Hasil ini secara kuat memvalidasi bahwa pendekatan berbasis machine learning dengan menggunakan indikator hematologis dasar memiliki potensi besar untuk deteksi dini anemia pada anak. Ini menguatkan argumen bahwa ketergantungan pada CBC lengkap sebagai satu-satunya metode skrining dapat dikurangi atau didukung oleh alat prediksi seperti ini.
   - Potensi Pengurangan Dampak Negatif Anemia. Dengan deteksi dini yang difasilitasi oleh model ini, intervensi medis dapat dilakukan lebih cepat, sehingga berpotensi mengurangi dampak negatif jangka panjang dari anemia pada perkembangan fisik dan kognitif anak, yang pada akhirnya dapat meringankan beban ekonomi dan sosial.
   - Dasar untuk Pengembangan Lebih Lanjut. Keberhasilan model ini menjadi dasar yang kuat untuk pengembangan lebih lanjut, seperti integrasi ke dalam sistem informasi kesehatan, pengembangan aplikasi mobile health (mHealth) untuk tenaga kesehatan, atau penelitian lebih lanjut untuk validasi pada populasi yang lebih luas dan beragam.

### Kesimpulan evaluasi
Proyek ini berhasil mengembangkan dan mengevaluasi model machine learning untuk prediksi dini anemia pada anak berdasarkan indikator hematologis, sebagai respons terhadap tingginya prevalensi anemia dan keterbatasan akses diagnosis di banyak wilayah. Melalui serangkaian tahapan mulai dari pemahaman data, persiapan data yang cermat (termasuk penanganan outlier dan seleksi fitur), hingga pemodelan komparatif menggunakan algoritma Logistic Regression, Gradient Boosting Classifier, dan SVM, ditemukan bahwa model Gradient Boosting Classifier menunjukkan performa superior. Model ini mencapai akurasi, presisi, recall, dan F1-score sempurna (100%) pada data uji, mengindikasikan kemampuannya untuk mengidentifikasi kasus anemia dengan sangat andal tanpa kesalahan prediksi dan potensi overfitting yang minimal. Keberhasilan ini menegaskan potensi besar machine learning dalam menyediakan alat bantu skrining anemia yang efektif, akurat, dan dapat diimplementasikan untuk mendukung deteksi dini serta intervensi yang lebih cepat, terutama di lingkungan dengan sumber daya terbatas, sehingga dapat berkontribusi pada peningkatan kesehatan anak.

**Referensi:**

[1] 	V. Martinez-Torres, N. Torres, J. A. Davis dan F. F. Corrales-Medina, “Anemia and   Associated Risk Factors in Pediatric Patients,” Pediatric Health, Medicine and Therapeutics, pp. 267-280, 2023. 
[2] 	G. A. Stevens, C. J. Paciorek, M. C. Flores-Urrutia, E. Borghi, S. Namaste, J. P. Wirth, P. S. Suchdev, M. Ezzati, F. Rohner, S. R. Flaxman dan L. M. Rogers, “National, regional, and global estimates of anaemia by severity in women and children for 2000–19: a pooled analysis of population-representative data,” 23 April 2022. [Online]. Available: https://www.thelancet.com/action/showPdf?pii=S2214-109X%2822%2900084-5. [Diakses 20 Mei 2025].
[3] 	A. B. Zemariam, A. Yimer, G. K. Abebe, W. T. Wondie, B. B. Abate, A. W. Alamaw, G. Yilak, T. M. Melaku dan H. S. Ngusie, “Employing supervised machine learning algorithms for classification and prediction of anemia among youth girls in Ethiopia,” Scientific Reports, 20 April 2024. [Online]. Available: https://www.nature.com/articles/s41598-024-60027-4?. [Diakses 20 Mei 2025].
[4] 	P. Dhakal, S. Khanal dan R. Bista, “Prediction of Anemia Using Machine Learning Algorithms,” International Journal of Computer Science & Information Technology (IJCSIT), vol. XV, no. 1, pp. 15-30, 2023. 
[5] 	J. W. Asare, W. L. Brown-Acquaye, M. M. Ujakpa, E. Freeman dan P. Appiahene, “Application of machine learning approach for iron deficiency anaemia detection in children using conjunctiva images,” Informatics in Medicine Unlocked , pp. 1-14, 2024. 
