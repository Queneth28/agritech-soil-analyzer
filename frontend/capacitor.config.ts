import { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.agritech.soilanalyzer',
  appName: 'AgriTech Soil Analyzer',
  webDir: 'build',
  server: {
    androidScheme: 'https',
    allowNavigation: ['10.0.2.2']
  },
  android: {
    backgroundColor: '#0A1929',
    allowMixedContent: true,
    webContentsDebuggingEnabled: true
  },
  plugins: {
    SplashScreen: {
      launchShowDuration: 0
    }
  }
};

export default config;