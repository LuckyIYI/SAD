// Entry point for the SAD trainer web page.
// Hugging-Face-ish UI: target-bpp + target-image in top bar, advanced options
// (mask, init-sites, densify/prune/lr…) in the collapsible left panel, and a
// side-by-side target vs reconstruction view, both aspect-fit.

import { loadShaders } from "./src/shader_loader.js?v=viewer-sync-20260423i";
import { Trainer, buildArgs } from "./src/trainer.js?v=viewer-sync-20260423i";
import { loadImageToFloat32, loadMaskRgba32Float, whiteMaskRgba32Float } from "./src/textures.js?v=viewer-sync-20260423i";
import { loadSitesFromFile, serializeSitesTxt, triggerDownload, floatImageToRgba8 } from "./src/io.js?v=viewer-sync-20260423i";
import { SITE_FLOATS } from "./src/params.js?v=viewer-sync-20260423i";

const CONFIG_URL = "../../training_config.json";
const DEMO_MAX_IMAGE_SIDE = 2048;
const DEMO_VIEWER_FREQ = 250;
const MIN_TARGET_BPP = 0.2;
const MAX_TARGET_BPP = 16.0;

// Fields that live in the advanced panel. Target BPP and target image are in
// the top bar; init-sites / mask are pickers inside the panel (not listed here).
const UI_FIELDS = [
  ["iters", "p-iters"],
  ["sites", "p-sites"],
  ["maxSites", "p-maxSites"],
  ["logFreq", "p-logFreq"],
  ["viewerFreq", "p-viewerFreq"],
  ["densify", "p-densify"],
  ["densifyStart", "p-densifyStart"],
  ["densifyEnd", "p-densifyEnd"],
  ["densifyFreq", "p-densifyFreq"],
  ["densifyPercentile", "p-densifyPercentile"],
  ["densifyScoreAlpha", "p-densifyScoreAlpha"],
  ["pruneStart", "p-pruneStart"],
  ["pruneEnd", "p-pruneEnd"],
  ["pruneFreq", "p-pruneFreq"],
  ["prunePercentile", "p-prunePercentile"],
  ["pruneDuringDensify", "p-pruneDuringDensify"],
  ["candFreq", "p-candFreq"],
  ["candUpdatePasses", "p-candUpdatePasses"],
  ["candDownscale", "p-candDownscale"],
  ["candRadiusScale", "p-candRadiusScale"],
  ["candRadiusProbes", "p-candRadiusProbes"],
  ["candInjectCount", "p-candInjectCount"],
  ["candHilbertWindow", "p-candHilbertWindow"],
  ["candHilbertProbes", "p-candHilbertProbes"],
  ["lrPosBase", "p-lrPosBase"],
  ["lrTauBase", "p-lrTauBase"],
  ["lrRadiusBase", "p-lrRadiusBase"],
  ["lrColorBase", "p-lrColorBase"],
  ["lrDirBase", "p-lrDirBase"],
  ["lrAnisoBase", "p-lrAnisoBase"],
];

const state = {
  device: null,
  shaders: null,
  config: null,
  targetImage: null,     // { data, width, height }
  maskData: null,        // { mask, rgba, maskSum } or null
  initSites: null,       // { data, width, height, count } or null
  running: false,
  abortCtrl: null,
  trainer: null,
  lastRender: null,      // { rgba, width, height }
  lastHashed: null,
  lastHeatmap: null,
  lastSites: null,
  lastSiteCount: 0,
  lastPsnr: null,        // persists between log ticks so preview updates don't blank the PSNR
};

const ui = {
  status: document.getElementById("status"),
  imageInput: document.getElementById("imageInput"),
  imageLabel: document.getElementById("imageLabel"),
  maskInput: document.getElementById("maskInput"),
  maskLabel: document.getElementById("maskLabel"),
  initSitesInput: document.getElementById("initSitesInput"),
  initLabel: document.getElementById("initLabel"),
  bppInput: document.getElementById("bppInput"),
  bppValue: document.getElementById("bppValue"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  toggleParamsBtn: document.getElementById("toggleParamsBtn"),
  saveBtn: document.getElementById("saveBtn"),
  savePopover: document.getElementById("savePopover"),
  savePng: document.getElementById("savePng"),
  saveTxt: document.getElementById("saveTxt"),
  paramsPanel: document.getElementById("paramsPanel"),
  viewMode: document.getElementById("viewMode"),
  previewLabel: document.getElementById("previewLabel"),
  previewCanvas: document.getElementById("previewCanvas"),
  overlayCanvas: document.getElementById("overlayCanvas"),
  targetCanvas: document.getElementById("targetCanvas"),
  metricRow: document.getElementById("metricRow"),
  progressFill: document.getElementById("progressFill"),
  progressText: document.getElementById("progressText"),
  log: document.getElementById("logPanel"),
};

function setStatus(msg) { ui.status.textContent = msg; }
function appendLog(line) {
  ui.log.textContent += line + "\n";
  ui.log.scrollTop = ui.log.scrollHeight;
}

function clampTargetBpp(value) {
  return Number.isFinite(value)
    ? Math.min(MAX_TARGET_BPP, Math.max(MIN_TARGET_BPP, value))
    : MIN_TARGET_BPP;
}

function setTargetBpp(value) {
  const targetBpp = clampTargetBpp(Number(value));
  ui.bppInput.value = targetBpp.toFixed(2);
  ui.bppValue.value = targetBpp.toFixed(2);
  ui.bppValue.textContent = targetBpp.toFixed(2);
  return targetBpp;
}

function applyConfigToForm(config) {
  const defaults = buildArgs(config, {});
  for (const [key, elId] of UI_FIELDS) {
    const el = document.getElementById(elId);
    if (!el) continue;
    const v = defaults[key];
    if (el.type === "checkbox") el.checked = Boolean(v);
    else if (v === undefined) el.value = "";
    else el.value = v;
  }
  document.getElementById("p-viewerFreq").value = DEMO_VIEWER_FREQ;
  if (Number.isFinite(defaults.targetBpp) && defaults.targetBpp > 0) {
    setTargetBpp(defaults.targetBpp);
  } else {
    setTargetBpp(MIN_TARGET_BPP);
  }
}

function readFormArgs() {
  const overrides = {};
  for (const [key, elId] of UI_FIELDS) {
    const el = document.getElementById(elId);
    if (!el) continue;
    if (el.type === "checkbox") overrides[key] = el.checked;
    else if (el.value === "") continue;
    else overrides[key] = Number(el.value);
  }
  const bpp = Number(ui.bppInput.value);
  const targetBpp = setTargetBpp(bpp);
  overrides.targetBpp = targetBpp;
  if (overrides.viewerFreq === undefined || !Number.isFinite(overrides.viewerFreq)) {
    overrides.viewerFreq = DEMO_VIEWER_FREQ;
  } else {
    overrides.viewerFreq = Math.max(1, overrides.viewerFreq);
  }
  return buildArgs(state.config, overrides);
}

async function initDevice() {
  if (!navigator.gpu) throw new Error("WebGPU not available.");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No WebGPU adapter.");
  const want = Math.max(12, adapter.limits.maxStorageBuffersPerShaderStage || 8);
  const required = {
    maxStorageBuffersPerShaderStage: Math.min(want, adapter.limits.maxStorageBuffersPerShaderStage),
    maxBufferSize: Math.min(1 << 30, adapter.limits.maxBufferSize),
    maxStorageBufferBindingSize: Math.min(1 << 30, adapter.limits.maxStorageBufferBindingSize),
  };
  const device = await adapter.requestDevice({ requiredLimits: required });
  device.lost.then((info) => appendLog(`Device lost: ${info.reason} ${info.message}`));
  device.addEventListener?.("uncapturederror", (e) => {
    appendLog(`GPU error: ${e.error?.message || e.error}`);
    console.error(e.error);
  });
  return device;
}

async function loadConfig() {
  const url = new URL(CONFIG_URL, import.meta.url);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load training_config.json: ${res.status}`);
  return await res.json();
}

function drawToCanvas(canvas, rgbaF, width, height) {
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  fitCanvasToCard(canvas, width, height);
  const ctx = canvas.getContext("2d");
  ctx.putImageData(new ImageData(floatImageToRgba8(rgbaF, width, height), width, height), 0, 0);
}

function fitCanvasToCard(canvas, width = canvas.width, height = canvas.height) {
  if (!canvas || !width || !height) return;
  const card = canvas.parentElement;
  if (!card) return;
  const rect = card.getBoundingClientRect();
  const cardWidth = card.clientWidth || rect.width;
  const cardHeight = card.clientHeight || rect.height;
  if (cardWidth <= 0 || cardHeight <= 0) return;
  const scale = Math.min(cardWidth / width, cardHeight / height);
  canvas.style.width = `${Math.max(1, width * scale)}px`;
  canvas.style.height = `${Math.max(1, height * scale)}px`;
}

function updateCanvasLayout(width = state.targetImage?.width, height = state.targetImage?.height) {
  const cards = [ui.targetCanvas.parentElement, ui.previewCanvas.parentElement];
  for (const card of cards) {
    card.style.removeProperty("aspect-ratio");
    card.style.removeProperty("width");
    card.style.removeProperty("height");
  }
  fitCanvasToCard(ui.targetCanvas, width || ui.targetCanvas.width, height || ui.targetCanvas.height);
  fitCanvasToCard(ui.previewCanvas, width || ui.previewCanvas.width, height || ui.previewCanvas.height);
  positionOverlay();
}

function drawTarget() {
  if (!state.targetImage) return;
  const { data, width, height } = state.targetImage;
  updateCanvasLayout(width, height);
  drawToCanvas(ui.targetCanvas, data, width, height);
}

async function handleImageInput(ev) {
  const file = ev.target.files[0];
  if (!file) return;
  const maxDim = DEMO_MAX_IMAGE_SIDE;
  setStatus(`Loading ${file.name}…`);
  ui.imageLabel.textContent = file.name;
  try {
    const img = await loadImageToFloat32(file, maxDim);
    const scaled = img.sourceWidth && img.sourceHeight &&
      (img.sourceWidth !== img.width || img.sourceHeight !== img.height);
    state.targetImage = img;
    drawTarget();
    // Blank the preview canvas to same size so layout matches immediately.
    ui.previewCanvas.width = img.width;
    ui.previewCanvas.height = img.height;
    const ctx = ui.previewCanvas.getContext("2d");
    ctx.fillStyle = "#0a0a0f";
    ctx.fillRect(0, 0, img.width, img.height);
    ui.overlayCanvas.width = img.width;
    ui.overlayCanvas.height = img.height;
    fitCanvasToCard(ui.previewCanvas, img.width, img.height);
    updateCanvasLayout(img.width, img.height);
    setStatus(scaled
      ? `Target ${img.width}×${img.height} (downscaled from ${img.sourceWidth}×${img.sourceHeight})`
      : `Target ${img.width}×${img.height}`);
    maybeEnableStart();
  } catch (err) {
    setStatus(`Failed: ${err.message}`);
    console.error(err);
  }
}

async function handleMaskInput(ev) {
  const file = ev.target.files[0];
  if (!file || !state.targetImage) return;
  try {
    state.maskData = await loadMaskRgba32Float(file, state.targetImage.width, state.targetImage.height);
    ui.maskLabel.textContent = file.name;
    appendLog(`Mask loaded (sum=${state.maskData.maskSum}).`);
  } catch (err) {
    appendLog(`Mask load failed: ${err.message}`);
  }
}

async function handleInitSitesInput(ev) {
  const file = ev.target.files[0];
  if (!file) return;
  try {
    state.initSites = await loadSitesFromFile(file);
    ui.initLabel.textContent = `${file.name} (${state.initSites.count})`;
    appendLog(`Init sites: ${state.initSites.count}` +
      (state.initSites.width ? ` (${state.initSites.width}×${state.initSites.height})` : ""));
  } catch (err) {
    appendLog(`Init sites load failed: ${err.message}`);
  }
}

function maybeEnableStart() {
  ui.startBtn.disabled = !(state.device && state.targetImage);
}

function scaleInitSitesIfNeeded(init, width, height) {
  if (!init) return null;
  if (!init.width || !init.height || (init.width === width && init.height === height)) return init;
  const sx = width / init.width;
  const sy = height / init.height;
  const data = new Float32Array(init.data);
  for (let i = 0; i < init.count; i += 1) {
    data[i * SITE_FLOATS + 0] *= sx;
    data[i * SITE_FLOATS + 1] *= sy;
  }
  return { ...init, data, width, height };
}

function overrideInitLogTau(init, initLogTau) {
  if (!init) return null;
  const data = new Float32Array(init.data);
  for (let i = 0; i < init.count; i += 1) {
    data[i * SITE_FLOATS + 2] = initLogTau;
  }
  return { ...init, data };
}

async function startTraining() {
  if (state.running) return;
  if (!state.device || !state.targetImage) return;

  ui.saveBtn.disabled = true;
  ui.savePopover.classList.add("hidden");
  ui.startBtn.disabled = true;
  ui.stopBtn.disabled = false;
  state.running = true;
  state.abortCtrl = new AbortController();
  state.lastHashed = null;
  state.lastHeatmap = null;

  const args = readFormArgs();
  const { width, height } = state.targetImage;

  let maskRgba = null;
  let maskSum = width * height;
  if (state.maskData) {
    maskRgba = state.maskData.rgba;
    maskSum = state.maskData.maskSum;
  } else {
    maskRgba = whiteMaskRgba32Float(width, height).rgba;
  }

  let init = scaleInitSitesIfNeeded(state.initSites, width, height);
  if (init) init = overrideInitLogTau(init, args.initLogTau);
  const initSiteCount = init ? init.count : args.sites;

  appendLog(`Target: ${width}×${height} | sites: ${initSiteCount} | iters: ${args.iters} | bpp: ${args.targetBpp > 0 ? args.targetBpp : "off"} | log: ${args.logFreq} | view: ${args.viewerFreq}`);
  updateProgress(0, args.iters, "Preparing…");

  const trainer = new Trainer(state.device, state.shaders, args, {
    signal: state.abortCtrl.signal,
    onLog: async (info) => {
      if (info.type === "msg") {
        appendLog(info.message);
      } else if (info.type === "iter") {
        const line = `Iter ${String(info.it).padStart(4)} | PSNR: ${info.psnr.toFixed(2)} dB | ` +
                     `Active: ${info.activeSites}/${info.totalSites} | ${info.itsPerSec.toFixed(1)} it/s | ${info.elapsedSec.toFixed(1)}s`;
        appendLog(line);
        state.lastRender = { rgba: info.renderRgba, width: info.width, height: info.height };
        state.lastSites = info.sitesData;
        state.lastSiteCount = info.totalSites;
        state.lastPsnr = info.psnr;
        state.lastHashed = null;
        state.lastHeatmap = null;
        updateProgress(info.it + 1, info.iters,
          `log iter ${info.it + 1}/${info.iters} · ${info.itsPerSec.toFixed(1)} it/s · ${info.elapsedSec.toFixed(1)}s`);
        renderMetrics(info);
        await drawView();
      } else if (info.type === "preview") {
        // Full image preview. Site overlays arrive through a separate lighter
        // event so they are not blocked by full-image readback.
        state.lastRender = { rgba: info.renderRgba, width: info.width, height: info.height };
        if (info.sitesData) {
          state.lastSites = info.sitesData;
          state.lastSiteCount = info.totalSites;
        }
        state.lastHashed = null;
        state.lastHeatmap = null;
        updateProgress(info.it + 1, info.iters,
          `image iter ${info.it + 1}/${info.iters} · ${info.itsPerSec.toFixed(1)} it/s · ${info.elapsedSec.toFixed(1)}s`);
        renderMetrics(info);
        await drawView();
      } else if (info.type === "sites") {
        state.lastSites = info.sitesData;
        state.lastSiteCount = info.totalSites;
        state.lastHashed = null;
        state.lastHeatmap = null;
        updateProgress(info.it + 1, info.iters,
          `sites iter ${info.it + 1}/${info.iters} · ${info.itsPerSec.toFixed(1)} it/s · ${info.elapsedSec.toFixed(1)}s`);
        renderMetrics(info);
        drawCentroidOverlay();
      }
    },
  });
  state.trainer = trainer;

  try {
    await trainer.prepare({
      width, height,
      targetRgba32f: state.targetImage.data,
      maskRgba32f: maskRgba,
      maskSum,
      initSites: init ? init.data : null,
      initSiteCount,
    });
    await trainer.run();
    const final = await trainer.finalize();
    state.lastRender = { rgba: final.renderRgba, width: final.width, height: final.height };
    state.lastSites = final.sites;
    state.lastSiteCount = final.count;
    state.lastHashed = null;
    state.lastHeatmap = null;
    await drawView();
    appendLog(`Done. Best PSNR: ${final.bestPsnr.toFixed(2)} dB | time: ${final.trainSeconds.toFixed(2)} s | active: ${final.count}`);
    updateProgress(args.iters, args.iters,
      `done · best ${final.bestPsnr.toFixed(2)} dB · ${final.trainSeconds.toFixed(1)}s`);
    ui.saveBtn.disabled = false;
  } catch (err) {
    appendLog(`Error: ${err.message}`);
    console.error(err);
  } finally {
    state.running = false;
    ui.startBtn.disabled = false;
    ui.stopBtn.disabled = true;
  }
}

// Unified metric renderer — PSNR sticks between log ticks so preview updates
// don't momentarily blank it.
function renderMetrics(info) {
  const psnr = Number.isFinite(info.psnr) ? info.psnr : state.lastPsnr;
  const psnrText = psnr != null ? `${psnr.toFixed(2)} dB` : "—";
  ui.metricRow.innerHTML =
    `<span>iter <strong>${info.it + 1}/${info.iters}</strong></span>` +
    `<span>PSNR <strong>${psnrText}</strong></span>` +
    `<span>active <strong>${info.activeSites}</strong>/${info.totalSites}</span>` +
    `<span><strong>${info.itsPerSec.toFixed(1)}</strong> it/s</span>`;
}

function updateProgress(cur, total, text) {
  const pct = total > 0 ? (cur / total) * 100 : 0;
  ui.progressFill.style.width = `${Math.min(100, pct).toFixed(1)}%`;
  ui.progressText.textContent = text;
}

function clearOverlay() {
  const ctx = ui.overlayCanvas.getContext("2d");
  ctx.clearRect(0, 0, ui.overlayCanvas.width, ui.overlayCanvas.height);
}

// The overlay canvas sits inside the same card as the preview but is absolutely
// positioned, so we need to match its drawing buffer and CSS box to the
// preview's actual displayed rect (which is aspect-fit inside the card).
function positionOverlay() {
  const parent = ui.previewCanvas.parentElement;
  const prect = ui.previewCanvas.getBoundingClientRect();
  const crect = parent.getBoundingClientRect();
  ui.overlayCanvas.style.left = `${prect.left - crect.left}px`;
  ui.overlayCanvas.style.top = `${prect.top - crect.top}px`;
  ui.overlayCanvas.style.width = `${prect.width}px`;
  ui.overlayCanvas.style.height = `${prect.height}px`;
}

function prepareOverlay(width, height) {
  if (ui.overlayCanvas.width !== width || ui.overlayCanvas.height !== height) {
    ui.overlayCanvas.width = width;
    ui.overlayCanvas.height = height;
  }
  positionOverlay();
  clearOverlay();
  return ui.overlayCanvas.getContext("2d");
}

function drawCentroidOverlay() {
  if (ui.viewMode.value !== "centroids") return;
  if (!state.lastRender || !state.lastSites) return;
  const ctx = prepareOverlay(state.lastRender.width, state.lastRender.height);
  drawCentroids(ctx);
}

async function drawView() {
  if (!state.lastRender) return;
  const mode = ui.viewMode.value;
  const { rgba, width, height } = state.lastRender;
  ui.previewLabel.textContent = {
    render: "Reconstruction",
    centroids: "Reconstruction + centroids",
    diagram: "Site IDs",
    heatmap: "Tau heatmap",
  }[mode] || "Reconstruction";

  if (mode === "render") {
    drawToCanvas(ui.previewCanvas, rgba, width, height);
    positionOverlay();
    clearOverlay();
    return;
  }
  if (mode === "centroids") {
    drawToCanvas(ui.previewCanvas, rgba, width, height);
    drawCentroidOverlay();
    return;
  }
  if (mode === "diagram") {
    const img = await ensureRender("hashed");
    if (img) {
      drawToCanvas(ui.previewCanvas, img.rgba, img.width, img.height);
      positionOverlay();
      clearOverlay();
    }
    return;
  }
  if (mode === "heatmap") {
    const img = await ensureRender("heatmap");
    if (img) {
      drawToCanvas(ui.previewCanvas, img.rgba, img.width, img.height);
      positionOverlay();
      clearOverlay();
    }
    return;
  }
}

async function ensureRender(mode) {
  if (!state.trainer) return null;
  if (mode === "hashed" && state.lastHashed) return state.lastHashed;
  if (mode === "heatmap" && state.lastHeatmap) return state.lastHeatmap;
  try {
    if (state.lastSites) state.trainer.setLastSites(state.lastSites);
    const rgba = await state.trainer.renderView(mode);
    const out = { rgba, width: state.lastRender.width, height: state.lastRender.height };
    if (mode === "hashed") state.lastHashed = out;
    else if (mode === "heatmap") state.lastHeatmap = out;
    return out;
  } catch (err) {
    appendLog(`View render failed: ${err.message}`);
    return null;
  }
}

function drawCentroids(ctx) {
  if (!state.lastSites) return;
  ctx.save();
  ctx.fillStyle = "rgba(255, 196, 64, 0.82)";
  for (let i = 0; i < state.lastSiteCount; i += 1) {
    const base = i * SITE_FLOATS;
    const x = state.lastSites[base + 0];
    const y = state.lastSites[base + 1];
    if (x < 0) continue;
    ctx.beginPath();
    ctx.arc(x, y, 1.4, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function stopTraining() {
  if (state.abortCtrl) state.abortCtrl.abort();
  appendLog("Stop requested.");
}

function exportPng() {
  if (!state.lastRender) return;
  ui.previewCanvas.toBlob((blob) => {
    if (blob) triggerDownload(blob, `sad_${Date.now()}.png`);
  }, "image/png");
  ui.savePopover.classList.add("hidden");
}

function exportSites() {
  if (!state.lastSites) return;
  const text = serializeSitesTxt(state.lastSites, state.lastSiteCount,
    state.lastRender.width, state.lastRender.height);
  triggerDownload(new Blob([text], { type: "text/plain" }), `sad_${Date.now()}_sites.txt`);
  ui.savePopover.classList.add("hidden");
}

function toggleSaveMenu(ev) {
  ev.stopPropagation();
  if (ui.saveBtn.disabled) return;
  ui.savePopover.classList.toggle("hidden");
}

function toggleParams() {
  ui.paramsPanel.classList.toggle("hidden");
  // Positioning depends on the preview's bounding rect, which changes when the
  // panel collapses/expands.
  requestAnimationFrame(() => {
    if (state.targetImage) updateCanvasLayout(state.targetImage.width, state.targetImage.height);
    else positionOverlay();
  });
}

async function main() {
  try {
    state.config = await loadConfig();
    applyConfigToForm(state.config);
    state.device = await initDevice();
    state.shaders = await loadShaders();
    setStatus("Ready");
    maybeEnableStart();
  } catch (err) {
    setStatus(`Setup failed: ${err.message}`);
    console.error(err);
  }

  ui.imageInput.addEventListener("change", handleImageInput);
  ui.maskInput.addEventListener("change", handleMaskInput);
  ui.initSitesInput.addEventListener("change", handleInitSitesInput);
  ui.bppInput.addEventListener("input", () => setTargetBpp(ui.bppInput.value));
  ui.startBtn.addEventListener("click", startTraining);
  ui.stopBtn.addEventListener("click", stopTraining);
  ui.toggleParamsBtn.addEventListener("click", toggleParams);
  ui.saveBtn.addEventListener("click", toggleSaveMenu);
  ui.savePng.addEventListener("click", exportPng);
  ui.saveTxt.addEventListener("click", exportSites);
  document.addEventListener("click", (ev) => {
    if (!ui.savePopover.contains(ev.target) && ev.target !== ui.saveBtn) {
      ui.savePopover.classList.add("hidden");
    }
  });
  ui.viewMode.addEventListener("change", drawView);
  window.addEventListener("resize", () => {
    requestAnimationFrame(() => {
      if (state.targetImage) updateCanvasLayout(state.targetImage.width, state.targetImage.height);
      else positionOverlay();
    });
  });
}

main();
