# Staywise — Hotel Booking Cancellation Predictor

Web prototipe akademik untuk memperkirakan peluang pembatalan reservasi hotel. Frontend menggunakan HTML, CSS, dan JavaScript Fetch API; backend menggunakan FastAPI dan Pydantic.

## Menjalankan aplikasi

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Buka `http://127.0.0.1:8000`. Status model dapat dilihat di `/health`; dokumentasi API tersedia di `/docs`.

## Integrasi model

Simpan pipeline scikit-learn lengkap dari kelompok sebagai `model/model_kelompok_4.joblib`, menggantikan file placeholder. Pipeline harus menerima DataFrame dengan fitur dan nama kolom berikut, berurutan seperti payload yang dibentuk API:

`hotel`, `lead_time`, `arrival_date_month`, `stays_in_weekend_nights`, `stays_in_week_nights`, `adults`, `children`, `babies`, `meal`, `market_segment`, `distribution_channel`, `is_repeated_guest`, `previous_cancellations`, `deposit_type`, `customer_type`.

Target klasifikasi diharapkan memakai `1` untuk reservasi dibatalkan dan `0` untuk tidak dibatalkan. Pipeline sebaiknya menyediakan `predict_proba` agar antarmuka menampilkan probabilitas. Sesuaikan form dan `BookingInput` di `app.py` jika kolom model final berbeda.

Model yang sekarang berada di folder tersebut adalah pipeline Random Forest final. Pipeline dilatih dengan scikit-learn 1.6.1. Jika memakai versi lebih baru, aplikasi menambahkan shim kompatibilitas untuk perubahan struktur pickle `ColumnTransformer` dan `SimpleImputer`; tetap gunakan versi scikit-learn yang sama saat memungkinkan.
