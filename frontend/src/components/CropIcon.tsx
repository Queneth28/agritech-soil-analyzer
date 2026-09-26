import React from 'react';
import { Wheat, Bean, Sprout, Flower2, Droplets } from 'lucide-react';

// Icon per crop family (SVG, not emoji, so it renders the same on every device)
const ICONS: Record<string, typeof Wheat> = {
  'Sorgho': Wheat, 'Mil': Wheat, 'Maïs': Wheat,
  'Riz': Droplets,
  'Niébé': Bean, 'Arachide': Bean, 'Soja': Bean,
  'Coton': Flower2, 'Sésame': Sprout,
};

const CropIcon = ({ name, size = 'md' }: { name: string; size?: 'sm' | 'md' | 'lg' }) => {
  const Icon = ICONS[name] || Sprout;
  const px = size === 'lg' ? 34 : size === 'sm' ? 15 : 20;
  return (
    <span className={`crop-icon ${size === 'md' ? '' : size}`} aria-hidden="true">
      <Icon size={px} />
    </span>
  );
};

export default CropIcon;
