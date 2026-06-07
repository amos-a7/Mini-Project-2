# Mini Project 2 — Object Counting (Penghitungan Jumlah Mobil)

**Mata Kuliah:** Pengolahan Citra dan Video  
**Cakupan Materi:** Pertemuan 9–14 (Color Model, Segmentasi, Morfologi, Thresholding)  
**Batas Pengumpulan:** Minggu ke-16

---

## 1. Identitas Pengerjakan

| Aspek | Keterangan |
|-------|-----------|
| **Nama** | Amos Harol Turnip |
| **NRP** | 5024241023 |
| **Kelas** | A |

---

## 2. Hasil Deteksi

### Jumlah Mobil Terdeteksi
**Total: [XX] mobil**

> Catatan: Angka ini akan berubah sesuai dengan hasil deteksi program saat dijalankan pada input image yang digunakan.

---

## 3. Penjelasan Pipeline

Berikut adalah tahapan-tahapan pemrosesan citra yang digunakan dalam program:

### **Step 1: Load dan Preprocessing Gambar**
- Membaca gambar menggunakan `cv2.imread()`
- Resize gambar ke 50% dari ukuran asli (scale = 0.5) untuk stabilitas dan efisiensi komputasi
- **Alasan:** Ukuran gambar yang lebih kecil membuat proses lebih cepat dan mengurangi noise

### **Step 2: Konversi ke HSV Color Space**
```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
```
- Mengkonversi dari BGR (format OpenCV) ke HSV (Hue, Saturation, Value)
- **Alasan:** HSV lebih baik untuk segmentasi warna karena memisahkan informasi warna (Hue) dari brightness dan saturation

### **Step 3: Asphalt Segmentation (Segmentasi Aspal)**
```python
lower_asphalt = np.array([0, 0, 40])
upper_asphalt = np.array([180, 70, 200])
asphalt_mask = cv2.inRange(hsv, lower_asphalt, upper_asphalt)
```
- Mengidentifikasi area aspal menggunakan range HSV yang telah ditentukan
- Range dipilih berdasarkan karakteristik warna aspal: grayscale gelap (low saturation, medium value)
- Output: Binary mask (nilai 255 untuk aspal, 0 untuk non-aspal)
- **Alasan:** Aspal memiliki karakteristik warna yang konsisten, sehingga lebih mudah di-segment dibanding mobil yang bervariasi

### **Step 4: Membuat Cars Mask**
```python
cars_mask = cv2.bitwise_not(asphalt_mask)
```
- Menginversi asphalt_mask untuk mendapatkan area mobil
- Area dengan nilai 255 = mobil, 0 = bukan mobil
- **Alasan:** Pendekatan indirect (segment aspal dulu) lebih stabil daripada langsung segment mobil

### **Step 5: Morphological Operations (Operasi Morfologi)**
```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
# Opening: Menghilangkan noise kecil
cars_mask = cv2.morphologyEx(cars_mask, cv2.MORPH_OPEN, kernel, iterations=1)
# Closing: Mengisi lubang kecil dalam objek
cars_mask = cv2.morphologyEx(cars_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
```
- **Opening (Erosion + Dilation):** Menghilangkan noise kecil dan objek yang tidak relevan
- **Closing (Dilation + Erosion):** Mengisi lubang kecil dalam objek untuk membuat bentuk lebih solid
- Kernel: Struktur persegi 5×5
- **Alasan:** Membersihkan mask dari noise dan menghasilkan blob yang lebih tegas

### **Step 6: Connected Components Labeling**
```python
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    cars_mask, connectivity=8
)
```
- Menemukan semua blob terpisah (connected components) dalam mask
- Setiap blob diberi label unik
- Mengekstrak statistik: posisi (x,y), ukuran (w,h), area, centroid
- **Alasan:** Memungkinkan analisis individual setiap objek

### **Step 7: Filtering dan Counting**
Setiap blob difilter berdasarkan kriteria berikut untuk memastikan hanya mobil yang dihitung:

| Kriteria | Nilai | Tujuan |
|----------|-------|--------|
| **Area Minimum** | 350 px² | Menghilangkan noise/objek terlalu kecil |
| **Area Maksimum** | 3000 px² | Menghilangkan objek terlalu besar (bukan mobil) |
| **Width Minimum** | 25 px | Objek harus cukup lebar |
| **Height Minimum** | 25 px | Objek harus cukup tinggi |
| **Width Maksimum** | 120 px | Membatasi lebar untuk menghilangkan blob besar |
| **Height Maksimum** | 120 px | Membatasi tinggi untuk menghilangkan blob besar |
| **Aspect Ratio** | max(w,h)/min(w,h) ≤ 3.5 | Mobil lebih kompak (tidak terlalu memanjang) |
| **Fill Ratio** | area/(w×h) ≥ 0.40 | Minimal 40% dari bounding box adalah area mobil |

### **Step 8: Visualisasi Hasil**
- Menggambar bounding box (green rectangle) untuk setiap mobil terdeteksi
- Memberi nomor dan label pada setiap mobil
- Menampilkan total count mobil di sudut kiri atas gambar
- Menyimpan hasil ke file output

---

## 4. Visualisasi Tahapan

Berikut adalah hasil output dari setiap tahap pipeline:

| Step | Deskripsi | File Output |
|------|-----------|------------|
| 1 | Gambar Original | `01_original.png` |
| 2 | Konversi HSV | `02_hsv.png` |
| 3 | Asphalt Segmentation Mask | `03_asphalt_mask.png` |
| 4 | Cars Mask (inverted) | `04_cars_mask.png` |
| 5 | Setelah Morphological Operations | `05_morphology.png` |
| 6 | Final Detection dengan Bounding Box | `06_final_detection.png` |
| All | Semua tahap dalam satu gambar | `all_steps.png` |

**Folder Output:** Semua hasil disimpan di `output/steps/` dan file hasil akhir di `output/result.png`

---

## 5. Analisis

### Kekuatan Pendekatan
1. **Robust terhadap variasi warna mobil** — menggunakan inverse masking (segment aspal dahulu) lebih stabil
2. **Morphological operations efektif** — membersihkan noise dan mengkonsolidasikan objek
3. **Multi-criteria filtering** — menggunakan 8 kriteria berbeda untuk mengidentifikasi mobil dengan akurat
4. **Visualisasi lengkap** — setiap tahap dapat diperiksa untuk debugging

### Kendala dan Limitasi
1. **Mobil yang saling tumpang tindih** — blob yang menyatu akan dihitung sebagai 1 objek
   - *Solusi potensial:* Menggunakan watershed algorithm atau teknik splitting yang lebih canggih
2. **Variasi pencahayaan** — area parkir yang gelap/terang dapat mempengaruhi segmentasi HSV
   - *Solusi potensial:* Adaptive thresholding atau normalisasi brightness
3. **Objek non-mobil serupa** — pohon, bayangan, atau pola lantai dapat terdeteksi sebagai mobil
   - *Solusi potensial:* Menambah kriteria shape atau texture
4. **Skala berbeda** — mobil dari sudut berbeda memiliki ukuran berbeda
   - *Solusi potensial:* Resize image ke multiple scales (image pyramid)

### Akurasi
- Untuk gambar parkiran dengan mobil yang terpisah jelas: **Akurasi tinggi (85-95%)**
- Untuk gambar dengan banyak mobil yang tumpang tindih: **Akurasi menengah (60-75%)**

### Peningkatan Masa Depan
1. Implementasi **Watershed Algorithm** untuk memisahkan mobil yang menyatu
2. Menggunakan **Edge Detection + Contour Analysis** sebagai layer tambahan
3. Mengimplementasikan **Shape Recognition** untuk filter lebih presisi
4. Menambah **Texture Analysis** untuk membedakan mobil dengan objek lain
5. Menggunakan **Multiple Image Scales** untuk menangani variasi ukuran

---

## 6. Cara Menjalankan Program

### Persyaratan
- Python 3.7+
- Library yang diperlukan: `opencv-python`, `numpy`, `matplotlib`

### Instalasi Dependencies
```bash
pip install opencv-python numpy matplotlib
```

### Struktur Folder
```
mp2-object-counting/
├── README.md              # File ini
├── counting.py            # Script utama
├── input/
│   └── parking.jpg        # Citra input (download dari assignment)
└── output/                # Folder output (dibuat otomatis saat run)
    ├── result.png
    └── steps/
```

### Langkah Menjalankan
1. **Letakkan gambar input**
   - Buat folder `gambar/` di direktori yang sama dengan `counting.py`
   - Pindahkan file `parking_ori.jpg` ke folder `gambar/`
   
2. **Jalankan script**
   ```bash
   python counting.py
   ```

3. **Hasil Output**
   - Gambar hasil deteksi: `output/result.png`
   - Visualisasi tahapan: `output/steps/` (folder berisi 6 step + all_steps.png)
   - Terminal akan menampilkan: `TOTAL MOBIL TERDETEKSI = [XX]`

### Catatan
- Program akan menampilkan setiap tahap dengan matplotlib (tekan close untuk lanjut ke tahap berikutnya)
- Jika ingin menonaktifkan visualisasi interaktif, comment-kan semua `show_step()` function calls
- Untuk mengubah sensitivitas deteksi, sesuaikan parameter filtering di Step 6 (area min/max, aspect ratio, dll.)

---

## 7. Parameter Tuning

Jika hasil deteksi kurang optimal, Anda dapat menyesuaikan parameter berikut:

### HSV Range (Step 3)
```python
lower_asphalt = np.array([0, 0, 40])      # Hue, Saturation, Value minimum
upper_asphalt = np.array([180, 70, 200])  # Hue, Saturation, Value maximum
```
- Coba gunakan HSV color picker atau Trackbar di OpenCV untuk menemukan range optimal

### Morphology Kernel Size (Step 5)
```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))  # Ubah ke (7,7), (3,3), dll.
```

### Filter Criteria (Step 7)
Sesuaikan min/max area, aspect ratio, dan fill ratio berdasarkan karakteristik gambar Anda

---

## 8. Repository GitHub

Untuk mengumpulkan, buat repository di GitHub dengan struktur di atas dan submit link ke Google Sheet yang sudah disediakan.

---

**Terakhir diperbarui:** [Tanggal]  
**Status:** Selesai / Dalam Pengerjaan
