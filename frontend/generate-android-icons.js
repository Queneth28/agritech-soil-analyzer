/**
 * Generate Android mipmap icons from app-icon.svg
 * Run: node generate-android-icons.js
 * Requires: npm install sharp (already installed)
 */
const sharp = require('sharp');
const path = require('path');
const fs = require('fs');

const svgPath = path.join(__dirname, 'public', 'app-icon.svg');
const svg = fs.readFileSync(svgPath);
const resDir = path.join(__dirname, 'android', 'app', 'src', 'main', 'res');

const densities = [
  { dir: 'mipmap-mdpi',    size: 48  },
  { dir: 'mipmap-hdpi',    size: 72  },
  { dir: 'mipmap-xhdpi',   size: 96  },
  { dir: 'mipmap-xxhdpi',  size: 144 },
  { dir: 'mipmap-xxxhdpi', size: 192 },
];

const variants = ['ic_launcher', 'ic_launcher_round'];

(async () => {
  for (const { dir, size } of densities) {
    for (const variant of variants) {
      const outPath = path.join(resDir, dir, `${variant}.webp`);
      const roundedSvg = variant === 'ic_launcher_round'
        ? Buffer.from(svg.toString().replace(
            '<rect width="512" height="512" fill="url(#bg)"/>',
            '<clipPath id="circle"><circle cx="256" cy="256" r="256"/></clipPath><rect width="512" height="512" fill="url(#bg)" clip-path="url(#circle)"/>'
          ))
        : svg;

      await sharp(variant === 'ic_launcher_round' ? roundedSvg : svg)
        .resize(size, size)
        .webp({ quality: 100 })
        .toFile(outPath);
      console.log(`✅ ${dir}/${variant}.webp (${size}x${size})`);
    }

    // foreground (same icon, slightly smaller for adaptive icon safe zone)
    const fgPath = path.join(resDir, dir, 'ic_launcher_foreground.webp');
    await sharp(svg)
      .resize(Math.round(size * 1.08), Math.round(size * 1.08))
      .resize(size, size, { fit: 'contain', background: { r: 0, g: 0, b: 0, alpha: 0 } })
      .webp({ quality: 100 })
      .toFile(fgPath);
    console.log(`✅ ${dir}/ic_launcher_foreground.webp`);
  }

  // Playstore icon
  const playstorePath = path.join(__dirname, 'android', 'app', 'src', 'main', 'ic_launcher-playstore.png');
  await sharp(svg).resize(512, 512).png({ quality: 100 }).toFile(playstorePath);
  console.log('✅ ic_launcher-playstore.png (512x512)');
})();
