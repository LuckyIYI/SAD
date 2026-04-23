// Texture creation + image ↔ rgba32float helpers.
// Equivalent of the inline texture calls and sad_shared.py's
// write_texture_rgba32float / read_texture_rgba32float.

export function createRgba32FloatTexture(device, width, height, { storage = true, copyDst = true, copySrc = false } = {}) {
  let usage = GPUTextureUsage.TEXTURE_BINDING;
  if (storage) usage |= GPUTextureUsage.STORAGE_BINDING;
  if (copyDst) usage |= GPUTextureUsage.COPY_DST;
  if (copySrc) usage |= GPUTextureUsage.COPY_SRC;
  return device.createTexture({
    size: [width, height, 1],
    format: "rgba32float",
    usage,
  });
}

export function createCandidateTexture(device, width, height) {
  return device.createTexture({
    size: [width, height, 1],
    format: "rgba32uint",
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.COPY_DST,
  });
}

export function writeFloat32ToRgba32FloatTexture(device, texture, data, width, height) {
  // data: Float32Array of length width*height*4 (row-major, rgba per pixel).
  const bytesPerPixel = 16;
  const bytesPerRow = width * bytesPerPixel;
  const aligned = Math.floor((bytesPerRow + 255) / 256) * 256;

  if (aligned === bytesPerRow) {
    device.queue.writeTexture(
      { texture },
      data,
      { bytesPerRow, rowsPerImage: height },
      [width, height, 1],
    );
    return;
  }

  // Row-pad to 256-byte alignment.
  const padded = new Uint8Array(aligned * height);
  const srcU8 = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  for (let y = 0; y < height; y += 1) {
    padded.set(srcU8.subarray(y * bytesPerRow, (y + 1) * bytesPerRow), y * aligned);
  }
  device.queue.writeTexture(
    { texture },
    padded,
    { bytesPerRow: aligned, rowsPerImage: height },
    [width, height, 1],
  );
}

export async function readRgba32FloatTexture(device, texture, width, height) {
  const bytesPerPixel = 16;
  const bytesPerRow = width * bytesPerPixel;
  const aligned = Math.floor((bytesPerRow + 255) / 256) * 256;

  const staging = device.createBuffer({
    size: aligned * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer(
    { texture },
    { buffer: staging, bytesPerRow: aligned, rowsPerImage: height },
    [width, height, 1],
  );
  device.queue.submit([encoder.finish()]);

  await staging.mapAsync(GPUMapMode.READ);
  const mapped = new Uint8Array(staging.getMappedRange()).slice();
  staging.unmap();
  staging.destroy();

  const out = new Float32Array(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    const rowU8 = mapped.subarray(y * aligned, y * aligned + bytesPerRow);
    out.set(new Float32Array(rowU8.buffer, rowU8.byteOffset, width * 4), y * width * 4);
  }
  return out;
}

// Load an ImageBitmap to an rgba32float Float32Array in [0, 1] (row-major).
// If maxDim is set and the image's longest side exceeds it, returns a scaled
// version (mirrors load_image() in train_wgpu.py).
export async function loadImageToFloat32(file, maxDim = 0) {
  const bitmap = await createImageBitmap(file, { colorSpaceConversion: "default" });
  let width = bitmap.width;
  let height = bitmap.height;
  const sourceWidth = width;
  const sourceHeight = height;
  let drawBitmap = bitmap;
  if (maxDim > 0 && Math.max(width, height) > maxDim) {
    const scale = maxDim / Math.max(width, height);
    const newW = Math.max(1, Math.floor(width * scale));
    const newH = Math.max(1, Math.floor(height * scale));
    drawBitmap = await createImageBitmap(bitmap, {
      resizeWidth: newW,
      resizeHeight: newH,
      resizeQuality: "high",
    });
    width = newW;
    height = newH;
  }

  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(drawBitmap, 0, 0);
  const image = ctx.getImageData(0, 0, width, height);
  const rgba = image.data; // Uint8ClampedArray
  const out = new Float32Array(width * height * 4);
  for (let i = 0; i < rgba.length; i += 1) {
    out[i] = rgba[i] / 255.0;
  }
  bitmap.close();
  if (drawBitmap !== bitmap) drawBitmap.close();
  return { data: out, width, height, sourceWidth, sourceHeight };
}

// Load a mask image to a grayscale float mask (Float32Array of length W*H) and
// an rgba32float mirror (for the mask texture). Mirrors load_mask() in train_wgpu.py
// plus the RGBA expansion done in main().
export async function loadMaskRgba32Float(file, width, height) {
  const bitmap = await createImageBitmap(file);
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0, width, height);
  const image = ctx.getImageData(0, 0, width, height);
  const rgba = image.data;
  const mask = new Float32Array(width * height);
  const out = new Float32Array(width * height * 4);
  let maskSum = 0;
  for (let i = 0; i < width * height; i += 1) {
    // Luminance via simple average of RGB (PIL convert("L") uses ITU-R 601-2, but
    // we only threshold > 0 so any channel mix works).
    const gray = (rgba[i * 4] + rgba[i * 4 + 1] + rgba[i * 4 + 2]) / 3.0 / 255.0;
    const m = gray > 0.0 ? 1.0 : 0.0;
    mask[i] = m;
    out[i * 4] = m;
    out[i * 4 + 1] = m;
    out[i * 4 + 2] = m;
    out[i * 4 + 3] = 1.0;
    if (m > 0.0) maskSum += 1;
  }
  bitmap.close();
  return { mask, rgba: out, maskSum };
}

// Build an all-white mask as rgba32float bytes for the fallback path.
export function whiteMaskRgba32Float(width, height) {
  const mask = new Float32Array(width * height).fill(1.0);
  const rgba = new Float32Array(width * height * 4);
  for (let i = 0; i < width * height; i += 1) {
    rgba[i * 4] = 1.0;
    rgba[i * 4 + 1] = 1.0;
    rgba[i * 4 + 2] = 1.0;
    rgba[i * 4 + 3] = 1.0;
  }
  return { mask, rgba, maskSum: width * height };
}
