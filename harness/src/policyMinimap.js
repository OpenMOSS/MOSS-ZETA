const sharp = require('sharp');
const fs = require('node:fs/promises');
const path = require('node:path');
const { imagePathToBuffer } = require('./captureStore');

async function policyMinimap(observation) {
  if (observation.policyMinimap) return observation.policyMinimap;
  const bytes = observation.buffer || await imagePathToBuffer(observation.imagePath);
  const { width, height } = await sharp(bytes).metadata();
  if (!width || !height || Math.abs(width / height - 16 / 9) > 0.02) throw Error('Paired minimap requires a 16:9 main observation');
  const left = 0, top = Math.round(height * 528 / 720);
  const buffer = await sharp(bytes).extract({ left, top, width: Math.round(width * 256 / 1280), height: height - top })
    .resize(512, 384, { kernel: sharp.kernel.cubic }).png().toBuffer();
  const imagePath = path.join(path.dirname(observation.imagePath), `${path.parse(observation.imagePath).name}-policy-minimap.png`);
  await fs.mkdir(path.dirname(imagePath), { recursive: true });
  await fs.writeFile(imagePath, buffer);
  observation.policyMinimap = { imagePath, mimeType: 'image/png', width: 512, height: 384, capturedAt: observation.capturedAt };
  return observation.policyMinimap;
}
module.exports = { policyMinimap };
