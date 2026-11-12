const form = document.querySelector("#booking-form");
const button = document.querySelector("#predict-button");
const emptyResult = document.querySelector("#empty-result");
const predictionResult = document.querySelector("#prediction-result");
const errorResult = document.querySelector("#error-result");
const errorMessage = document.querySelector("#error-message");

function showPanel(panel) {
  emptyResult.hidden = panel !== "empty";
  predictionResult.hidden = panel !== "prediction";
  errorResult.hidden = panel !== "error";
}

async function updateHealth() {
  const statusText = document.querySelector("#status-text");
  const statusDot = document.querySelector("#status-dot");
  try {
    const response = await fetch("/health");
    const health = await response.json();
    statusDot.classList.toggle("is-ready", health.model_loaded);
    statusText.textContent = health.model_loaded ? "Model siap" : "Model belum tersedia";
    statusDot.title = health.model_message || "";
  } catch {
    statusDot.classList.remove("is-ready");
    statusText.textContent = "API tidak terhubung";
  }
}

function payloadFromForm() {
  const values = Object.fromEntries(new FormData(form).entries());
  const numericFields = [
    "lead_time", "stays_in_weekend_nights", "stays_in_week_nights", "adults",
    "children", "babies", "is_repeated_guest", "previous_cancellations",
  ];
  for (const name of numericFields) values[name] = Number(values[name]);
  return values;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!form.reportValidity()) return;

  const label = button.querySelector("span:first-child");
  button.disabled = true;
  button.classList.add("is-loading");
  label.textContent = "Menganalisis reservasi…";
  errorResult.hidden = true;

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payloadFromForm()),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Terjadi kesalahan pada layanan prediksi.");

    if (result.cancellation_probability == null) {
      throw new Error("Pipeline model tidak menyediakan estimasi probabilitas (predict_proba).");
    }
    const percent = Math.round(result.cancellation_probability * 100);
    document.querySelector("#risk-percent").textContent = `${percent}%`;
    document.querySelector("#risk-fill").style.width = `${Math.min(100, Math.max(0, percent))}%`;
    document.querySelector("#confidence-value").textContent = result.confidence == null ? "Tidak tersedia" : `${Math.round(result.confidence * 100)}%`;

    const isCanceled = result.prediction === "cancelled";
    const verdict = document.querySelector("#verdict");
    verdict.classList.toggle("is-risk", isCanceled);
    document.querySelector("#verdict-icon").textContent = isCanceled ? "!" : "✓";
    document.querySelector("#verdict-title").textContent = result.label;
    document.querySelector("#verdict-copy").textContent = isCanceled
      ? "Model mengenali pola yang berisiko dibatalkan."
      : "Model memperkirakan reservasi tetap aktif.";
    showPanel("prediction");
    document.querySelector("#result-card").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    errorMessage.textContent = error.message || "Periksa kembali koneksi API Anda.";
    showPanel("error");
  } finally {
    button.disabled = false;
    button.classList.remove("is-loading");
    label.textContent = "Hitung peluang pembatalan";
  }
});

document.querySelector("#retry-button").addEventListener("click", () => form.requestSubmit());
updateHealth();
