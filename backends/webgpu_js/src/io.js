// Sites .txt parsing + serialization. PNG download helper.
// Direct port of the corresponding functions in backends/webgpu_py/train_wgpu.py:
// load_sites_txt, write_sites_txt.

import { SITE_FLOATS } from "./params.js";

// Returns { data: Float32Array of length count*SITE_FLOATS, width, height, count }.
// Layout per site row: [pos_x, pos_y, log_tau, radius, color_r, color_g, color_b,
// aniso_dir_x, aniso_dir_y, log_aniso]. The .txt file format written by
// write_sites_txt orders the columns differently on disk — see below.
export function parseSitesTxt(text) {
  const sizeRe = /image size\s*:?\s*(\d+)\s+(\d+)/i;
  let width = null;
  let height = null;
  const rows = [];
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) continue;
    if (line.startsWith("#")) {
      const m = sizeRe.exec(line);
      if (m) {
        width = Number(m[1]);
        height = Number(m[2]);
      }
      continue;
    }
    const parts = line.split(/\s+/);
    if (parts.length !== 7 && parts.length !== 10) continue;
    const vals = parts.map(Number);
    if (vals.some(Number.isNaN)) continue;
    if (vals.length === 7) vals.push(1.0, 0.0, 0.0);
    rows.push(vals);
  }
  if (rows.length === 0) throw new Error("No sites found in TXT input");

  const data = new Float32Array(rows.length * SITE_FLOATS);
  for (let i = 0; i < rows.length; i += 1) {
    const s = rows[i];
    // TXT on-disk order: x y r g b log_tau radius dir_x dir_y log_aniso
    const base = i * SITE_FLOATS;
    data[base + 0] = s[0]; // pos_x
    data[base + 1] = s[1]; // pos_y
    data[base + 2] = s[5]; // log_tau
    data[base + 3] = s[6]; // radius_sq
    data[base + 4] = s[2]; // color_r
    data[base + 5] = s[3]; // color_g
    data[base + 6] = s[4]; // color_b
    data[base + 7] = s[7]; // aniso_dir_x
    data[base + 8] = s[8]; // aniso_dir_y
    data[base + 9] = s[9]; // log_aniso
  }
  return { data, width, height, count: rows.length };
}

export async function loadSitesFromFile(file) {
  const text = await file.text();
  return parseSitesTxt(text);
}

// Serialize Float32Array (count*SITE_FLOATS) to the on-disk .txt format.
// Only active sites (pos_x >= 0) are written, matching write_sites_txt().
export function serializeSitesTxt(data, count, width, height) {
  const active = [];
  for (let i = 0; i < count; i += 1) {
    const base = i * SITE_FLOATS;
    if (data[base + 0] >= 0.0) active.push(base);
  }
  const lines = [];
  lines.push(
    "# SAD Sites (position_x, position_y, color_r, color_g, color_b, log_tau, radius, aniso_dir_x, aniso_dir_y, log_aniso)",
  );
  lines.push(`# Image size: ${width} ${height}`);
  lines.push(`# Total sites: ${active.length}`);
  lines.push(`# Active sites: ${active.length}`);
  for (const base of active) {
    lines.push(
      [
        data[base + 0], // pos_x
        data[base + 1], // pos_y
        data[base + 4], // color_r
        data[base + 5], // color_g
        data[base + 6], // color_b
        data[base + 2], // log_tau
        data[base + 3], // radius
        data[base + 7], // dir_x
        data[base + 8], // dir_y
        data[base + 9], // log_aniso
      ].map((v) => formatNumber(v)).join(" "),
    );
  }
  return lines.join("\n") + "\n";
}

function formatNumber(v) {
  // 9 significant digits, trimmed — emulates Python "{:.9g}".
  if (!Number.isFinite(v)) return "0";
  const s = v.toPrecision(9);
  // Drop trailing zeros / unnecessary trailing decimal point.
  return s.includes("e") ? s : s.replace(/\.?0+$/, "").replace(/^$/, "0");
}

// Compute PSNR between two Float32Array RGBA buffers (length W*H*4). Mask is
// optional; only pixels with mask[i] > 0 contribute. Returns dB, with 100.0 as
// a sentinel when MSE is 0.
export function computePsnr(render, target, width, height, mask = null) {
  const n = width * height;
  let count = 0;
  let acc = 0;
  for (let i = 0; i < n; i += 1) {
    if (mask && mask[i] <= 0) continue;
    const r = render[i * 4] - target[i * 4];
    const g = render[i * 4 + 1] - target[i * 4 + 1];
    const b = render[i * 4 + 2] - target[i * 4 + 2];
    acc += r * r + g * g + b * b;
    count += 3;
  }
  if (count === 0) return 0.0;
  const mse = acc / count;
  if (mse <= 0) return 100.0;
  return 20.0 * Math.log10(1.0 / Math.sqrt(mse));
}

// Sanitize floating-point output (NaN/Inf → clamped) and convert rgba32float
// Float32Array (length W*H*4) into a Uint8ClampedArray suitable for ImageData.
export function floatImageToRgba8(rgbaF, width, height) {
  const n = width * height;
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i += 1) {
    for (let c = 0; c < 3; c += 1) {
      let v = rgbaF[i * 4 + c];
      if (!Number.isFinite(v)) v = 0;
      v = Math.max(0, Math.min(1, v));
      out[i * 4 + c] = Math.round(v * 255);
    }
    out[i * 4 + 3] = 255;
  }
  return out;
}

// Kick off a browser download of a Blob with a given filename.
export function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
