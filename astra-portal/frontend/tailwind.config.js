/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        space: {
          bg: "#0B0F19",
          dark: "#141B2D",
          card: "#1F2940",
          border: "#2A3655",
        },
        cyan: {
          accent: "#00E5FF",
          dim: "#00838F",
        },
        emerald: {
          accent: "#00E676",
        },
        amber: {
          accent: "#FFD600",
        },
      },
      fontFamily: {
        mono: ["Consolas", "Courier New", "monospace"],
        sans: ["Inter", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
