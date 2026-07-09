/**
 * RetinAI — Frontend Logic
 * Handles file upload, API calls, and results rendering.
 */

(function () {
  "use strict";

  // ── DOM Refs ──────────────────────────────────────────────
  const dropZone       = document.getElementById("drop-zone");
  const fileInput      = document.getElementById("file-input");
  const previewOverlay = document.getElementById("preview-overlay");
  const previewImg     = document.getElementById("preview-img");
  const previewName    = document.getElementById("preview-name");
  const clearBtn       = document.getElementById("clear-btn");
  const analyzeBtn     = document.getElementById("analyze-btn");

  const uploadSection  = document.getElementById("upload-section");
  const loadingSection = document.getElementById("loading-section");
  const resultsSection = document.getElementById("results-section");
  const errorSection   = document.getElementById("error-section");
  const errorMessage   = document.getElementById("error-message");
  const errorRetryBtn  = document.getElementById("error-retry-btn");
  const resetBtn       = document.getElementById("reset-btn");

  let selectedFile = null;

  // ── File Selection ────────────────────────────────────────
  dropZone.addEventListener("click", (e) => {
    if (e.target.closest("#clear-btn") || e.target.closest(".btn-icon")) return;
    fileInput.click();
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
  });

  // Drag & Drop
  ["dragenter", "dragover"].forEach((evt) =>
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.add("drag-over");
    })
  );

  ["dragleave", "drop"].forEach((evt) =>
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.remove("drag-over");
    })
  );

  dropZone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length) handleFile(files[0]);
  });

  function handleFile(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please select an image file (PNG, JPG, JPEG).");
      return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewName.textContent = file.name;
      previewOverlay.classList.remove("hidden");
      analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  // Clear
  clearBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    resetUpload();
  });

  function resetUpload() {
    selectedFile = null;
    fileInput.value = "";
    previewImg.src = "";
    previewOverlay.classList.add("hidden");
    analyzeBtn.disabled = true;
  }

  // ── Analyze ───────────────────────────────────────────────
  analyzeBtn.addEventListener("click", () => {
    if (!selectedFile) return;
    runInference(selectedFile);
  });

  async function runInference(file) {
    showSection("loading");

    const formData = new FormData();
    formData.append("image", file);

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok || data.error) {
        throw new Error(data.error || `Server returned ${res.status}`);
      }

      renderResults(data);
      showSection("results");
    } catch (err) {
      errorMessage.textContent = err.message;
      showSection("error");
    }
  }

  // ── Show Section ──────────────────────────────────────────
  function showSection(name) {
    uploadSection.classList.toggle("hidden", name !== "upload");
    loadingSection.classList.toggle("hidden", name !== "loading");
    resultsSection.classList.toggle("hidden", name !== "results");
    errorSection.classList.toggle("hidden", name !== "error");
  }

  // ── Render Results ────────────────────────────────────────
  function renderResults(data) {
    const isAnomaly = data.is_anomalous;

    // Verdict
    const verdictCard = document.getElementById("verdict-card");
    verdictCard.className = `verdict-card glass-card ${isAnomaly ? "anomalous" : "healthy"}`;

    const verdictIcon = document.getElementById("verdict-icon");
    verdictIcon.innerHTML = isAnomaly ? "⚠" : "✓";

    document.getElementById("verdict-label").textContent = isAnomaly
      ? "⚠ DR SUSPECTED — Anomaly Detected"
      : "✓ HEALTHY — No Anomaly Detected";

    document.getElementById("verdict-action").textContent = data.action;

    // Score
    const scoreVal = document.getElementById("score-value");
    scoreVal.textContent = data.score.toFixed(6);

    // Animate score bar — cap visual at 1.0 for display
    const maxVisual = Math.max(data.threshold * 2.5, 1.0);
    const barPct = Math.min((data.score / maxVisual) * 100, 100);
    const threshPct = Math.min((data.threshold / maxVisual) * 100, 100);

    requestAnimationFrame(() => {
      document.getElementById("score-bar").style.width = barPct + "%";
      document.getElementById("score-threshold-marker").style.left = threshPct + "%";
    });

    document.getElementById("threshold-label").textContent =
      `Threshold: ${data.threshold.toFixed(4)}`;

    // Confidence
    document.getElementById("confidence-value").textContent = data.confidence.toFixed(1) + "%";

    const circumference = 2 * Math.PI * 52; // r=52
    const offset = circumference * (1 - data.confidence / 100);
    const ringFill = document.getElementById("confidence-ring-fill");

    // Add gradient for confidence ring
    ensureRingGradient(isAnomaly);

    requestAnimationFrame(() => {
      ringFill.style.strokeDashoffset = offset;
    });
    document.getElementById("confidence-ring-pct").textContent =
      data.confidence.toFixed(0) + "%";

    // Device
    document.getElementById("device-value").textContent = data.device.toUpperCase();

    // Images
    document.getElementById("img-original").src =
      "data:image/png;base64," + data.original_b64;
    document.getElementById("img-reconstructed").src =
      "data:image/png;base64," + data.reconstructed_b64;
    document.getElementById("img-error").src =
      "data:image/png;base64," + data.error_map_b64;
  }

  function ensureRingGradient(isAnomaly) {
    // Inject SVG gradient for the confidence ring
    let svg = document.querySelector(".confidence-ring");
    let defs = svg.querySelector("defs");
    if (!defs) {
      defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
      svg.prepend(defs);
    }

    const color1 = isAnomaly ? "#ef4444" : "#10b981";
    const color2 = isAnomaly ? "#f59e0b" : "#06b6d4";

    defs.innerHTML = `
      <linearGradient id="ring-gradient" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="${color1}"/>
        <stop offset="100%" stop-color="${color2}"/>
      </linearGradient>
    `;

    document.getElementById("confidence-ring-fill").setAttribute("stroke", "url(#ring-gradient)");
  }

  // ── Reset / Retry ─────────────────────────────────────────
  resetBtn.addEventListener("click", () => {
    resetUpload();
    // Reset animations
    document.getElementById("score-bar").style.width = "0%";
    document.getElementById("confidence-ring-fill").style.strokeDashoffset = "326.73";
    showSection("upload");
  });

  errorRetryBtn.addEventListener("click", () => {
    showSection("upload");
  });
})();
