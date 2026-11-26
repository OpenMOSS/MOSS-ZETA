const sharp = require("sharp");
const { saveCapture } = require("./captureStore");

function finiteNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function normalizedCrop(crop, width, height) {
  return {
    left: finiteNumber(crop.left) * width,
    top: finiteNumber(crop.top) * height,
    width: finiteNumber(crop.width) * width,
    height: finiteNumber(crop.height) * height
  };
}

function clampCrop(crop, width, height) {
  const left = Math.max(0, Math.min(width, Math.trunc(crop.left)));
  const top = Math.max(0, Math.min(height, Math.trunc(crop.top)));
  const right = Math.max(left, Math.min(width, Math.trunc(crop.left + crop.width)));
  const bottom = Math.max(top, Math.min(height, Math.trunc(crop.top + crop.height)));
  return {
    left,
    top,
    width: right - left,
    height: bottom - top
  };
}

function resolveMinimapCrop(crop, width, height) {
  const coordinateSpace = String(crop?.coordinateSpace || "image").trim().toLowerCase();
  const raw = coordinateSpace === "normalized"
    ? normalizedCrop(crop || {}, width, height)
    : {
        left: finiteNumber(crop?.left),
        top: finiteNumber(crop?.top),
        width: finiteNumber(crop?.width),
        height: finiteNumber(crop?.height)
      };
  const resolved = clampCrop(raw, width, height);
  if (resolved.width <= 0 || resolved.height <= 0) {
    throw new Error("minimap crop must have a positive area inside the screenshot.");
  }
  return {
    ...resolved,
    coordinateSpace,
    sourceWidth: width,
    sourceHeight: height
  };
}

async function attachMinimapCrop(observation, config = {}, options = {}) {
  if (!config.navigationMinimap?.enabled) {
    return observation;
  }
  if (!Buffer.isBuffer(observation?.buffer)) {
    throw new Error("minimap crop requires the observation image buffer.");
  }

  const metadata = await sharp(observation.buffer).metadata();
  const sourceWidth = Number(metadata.width || observation.width);
  const sourceHeight = Number(metadata.height || observation.height);
  const crop = resolveMinimapCrop(config.navigationMinimap.crop, sourceWidth, sourceHeight);
  const buffer = await sharp(observation.buffer)
    .extract({
      left: crop.left,
      top: crop.top,
      width: crop.width,
      height: crop.height
    })
    .jpeg({ quality: Number(options.quality) || 90 })
    .toBuffer();

  const saved = await saveCapture(buffer, {
    sessionId: options.sessionId || "manual",
    label: options.label || "minimap-crop",
    mimeType: "image/jpeg",
    captureRoot: options.captureRoot
  });

  return {
    ...observation,
    navigation: {
      ...(observation.navigation || {}),
      minimap: {
        enabled: true,
        coordinateSpace: crop.coordinateSpace,
        crop: {
          left: crop.left,
          top: crop.top,
          width: crop.width,
          height: crop.height,
          sourceWidth: crop.sourceWidth,
          sourceHeight: crop.sourceHeight
        },
        buffer,
        imagePath: saved.filePath,
        imageUrl: saved.publicUrl,
        mimeType: saved.mimeType,
        width: crop.width,
        height: crop.height
      }
    }
  };
}

function minimapHistoryMetadata(observation) {
  const minimap = observation?.navigation?.minimap;
  if (!minimap) {
    return null;
  }
  return {
    enabled: true,
    coordinateSpace: minimap.coordinateSpace,
    crop: minimap.crop,
    imagePath: minimap.imagePath || null,
    imageUrl: minimap.imageUrl || null,
    mimeType: minimap.mimeType || "image/jpeg",
    width: minimap.width,
    height: minimap.height
  };
}

module.exports = {
  attachMinimapCrop,
  minimapHistoryMetadata,
  resolveMinimapCrop
};
