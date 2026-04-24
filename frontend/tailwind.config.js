/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        background: '#0a0a0a',
        sidebar: '#121212',
        card: '#1a1a1a',
        accent: {
          blue: '#3b82f6',
          yellow: '#eab308',
          red: '#ef4444',
        }
      },
    },
  },
  plugins: [],
}
