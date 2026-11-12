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

`lead_time`, `arrival_date_month`, `arrival_date_week_number`, `arrival_date_day_of_month`, `stays_in_weekend_nights`, `stays_in_week_nights`, `adults`, `children`, `meal`, `country`, `market_segment`, `distribution_channel`, `is_repeated_guest`, `previous_cancellations`, `booking_changes`, `deposit_type`, `customer_type`, `adr`, `required_car_parking_spaces`, `total_of_special_requests`.

Target klasifikasi diharapkan memakai `1` untuk reservasi dibatalkan dan `0` untuk tidak dibatalkan. Pipeline sebaiknya menyediakan `predict_proba` agar antarmuka menampilkan probabilitas. Sesuaikan form dan `BookingInput` di `app.py` jika kolom model final berbeda.

Model yang sekarang berada di folder tersebut adalah placeholder, bukan file Joblib yang bisa dibaca. Sampai model final tersedia, `/health` akan menyatakan bahwa model belum siap dan `/predict` akan mengembalikan HTTP 503.
