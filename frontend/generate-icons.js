/**
 * Regenerate logo192.png and logo512.png from app-icon.svg
 * Run: node generate-icons.js
 * Requires: npm install sharp (one-time)
 */
const sharp = require('sharp');
const path = require('path');
const fs = require('fs');

const svgPath = path.join(__dirname, 'public', 'app-icon.svg');
const svg = fs.readFileSync(svgPath);

const sizes = [
  { size: 192, out: 'logo192.png' },
  { size: 512, out: 'logo512.png' },
  { size: 512, out: 'app-icon.png' },
];

(async () => {
  for (const { size, out } of sizes) {
    const outPath = path.join(__dirname, 'public', out);
    await sharp(svg)
      .resize(size, size)
      .png({ quality: 100, compressionLevel: 6 })
      .toFile(outPath);
    console.log(`✅ Generated ${out} (${size}x${size})`);
  }
})();
