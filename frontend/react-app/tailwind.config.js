/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "SF Pro Text", "system-ui", "sans-serif"],
        display: ["Poppins", "Inter", "sans-serif"],
      },
      colors: {
        surface: {
          bg: "var(--bg-primary)",
          panel: "var(--bg-secondary)",
          panelSoft: "var(--bg-tertiary)",
        },
        text: {
          primary: "var(--text-primary)",
          secondary: "var(--text-secondary)",
          muted: "var(--text-muted)",
        },
        accent: "var(--accent)",
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(34, 211, 238, 0.28), 0 20px 45px rgba(2, 132, 199, 0.22)",
        glass: "0 16px 45px rgba(2, 6, 23, 0.28)",
      },
    },
  },
  plugins: [],
};
