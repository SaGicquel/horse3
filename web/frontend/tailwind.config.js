/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    screens: {
      'xs': '475px',
      'sm': '640px',
      'md': '768px',
      'lg': '1024px',
      'xl': '1280px',
      '2xl': '1536px',
    },
    extend: {
      fontFamily: {
        display: ['Bebas Neue', 'sans-serif'],
        sans: ['Inter', 'sans-serif'],
        mono: ['Roboto Mono', 'monospace'],
      },
      colors: {
        // Palette principale - Premium & Vibrant
        primary: {
          50: '#fdf2f8',
          100: '#fce7f3',
          200: '#fbcfe8',
          300: '#f9a8d4',
          400: '#f472b6',
          500: '#ec4899', // Pink/Magenta vibrant
          600: '#db2777',
          700: '#be185d',
          800: '#9d174d',
          900: '#831843',
          950: '#500724',
        },
        secondary: {
          50: '#f5f3ff',
          100: '#ede9fe',
          200: '#ddd6fe',
          300: '#c4b5fd',
          400: '#a78bfa',
          500: '#8b5cf6', // Violet vibrant
          600: '#7c3aed',
          700: '#6d28d9',
          800: '#5b21b6',
          900: '#4c1d95',
          950: '#2e1065',
        },
        accent: {
          500: '#06b6d4', // Cyan
          600: '#0891b2',
        },
        success: '#10B981', // Emerald 500
        warning: '#F59E0B', // Amber 500
        error: '#EF4444',   // Red 500
        info: '#3B82F6',    // Blue 500

        // Neutral palette - Slate based for premium feel
        neutral: {
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
          700: '#334155',
          800: '#1e293b',
          900: '#0f172a',
          950: '#020617', // Deep dark background
        },

        // Anciennes couleurs (support temporaire)
        'midnight-900': '#0b1120',
        'midnight-800': '#111a37',
        'midnight-700': '#182347',
        'card-800': '#141c34',
        'card-700': '#192443',
        'card-border': '#253153',
        'horse-primary': '#AA325D',
        'horse-dark': '#971747',
        'horse-light': '#CA6384',
        'horse-accent': '#D84A78',
        'horse-glow': '#E75B8C',
        'brand-green': '#34d399',
        'brand-yellow': '#fbbf24',
        'brand-red': '#f87171',
      },
      boxShadow: {
        'card-glow': '0 20px 45px rgba(11, 15, 26, 0.5)',
        'p-glow': '0 20px 45px rgba(157, 54, 86, 0.3)',
        'p-glow-strong': '0 10px 30px rgba(157, 54, 86, 0.5)',
        // Anciennes ombres (à retirer progressivement)
        'horse-glow': '0 20px 45px rgba(170, 50, 93, 0.3)',
        'horse-glow-strong': '0 10px 30px rgba(170, 50, 93, 0.5)',
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
}
