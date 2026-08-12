import { Layers, MoonStar, Sun, Waves, Wind } from "lucide-react";
import { clsx } from "clsx";

const LAYERS = [
  { value: "all", label: "All Layers" },
  { value: "dz", label: "Dead Zones" },
  { value: "gg", label: "Ghost Gear" },
];

function SegmentedButton({ active, children, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={clsx(
        "rounded-lg px-3 py-2 text-xs font-semibold transition-all duration-300",
        active
          ? "bg-cyan-500/20 text-cyan-200 ring-1 ring-cyan-300/60 shadow-glow"
          : "text-text-secondary hover:bg-white/10 hover:text-text-primary",
      )}
    >
      {children}
    </button>
  );
}

export function TopBar({
  theme,
  layer,
  windEnabled,
  currentsEnabled,
  onChangeLayer,
  onToggleWind,
  onToggleCurrents,
  onToggleTheme,
}) {
  return (
    <header className="glass-panel sticky top-0 z-20 flex min-h-16 w-full flex-wrap items-center justify-between gap-3 px-4 py-3 md:px-6">
      <div className="min-w-[220px]">
        <h1 className="font-display text-2xl font-bold tracking-[0.24em] text-text-primary md:text-3xl">
          ANOXIA
        </h1>
        <p className="mt-0.5 text-xs font-medium text-text-secondary md:text-sm">
          Bay of Bengal ocean intelligence with dead-zone and ghost-gear overlays
        </p>
      </div>

      <div className="flex flex-1 flex-wrap items-center justify-end gap-2">
        <div className="glass-soft flex items-center gap-1 rounded-xl p-1">
          <Layers size={14} className="ml-1 text-text-muted" />
          {LAYERS.map((item) => (
            <SegmentedButton
              key={item.value}
              active={layer === item.value}
              onClick={() => onChangeLayer(item.value)}
            >
              {item.label}
            </SegmentedButton>
          ))}
        </div>

        <button
          type="button"
          onClick={onToggleWind}
          className={clsx(
            "glass-soft inline-flex items-center gap-2 rounded-xl px-3 py-2 text-xs font-semibold transition-all duration-300",
            windEnabled
              ? "bg-sky-500/20 text-sky-100 ring-1 ring-sky-300/70 shadow-glow"
              : "text-text-secondary hover:bg-white/10 hover:text-text-primary",
          )}
        >
          <Wind size={14} />
          Wind Patterns
        </button>

        <button
          type="button"
          onClick={onToggleCurrents}
          className={clsx(
            "glass-soft inline-flex items-center gap-2 rounded-xl px-3 py-2 text-xs font-semibold transition-all duration-300",
            currentsEnabled
              ? "bg-violet-500/20 text-violet-100 ring-1 ring-violet-300/70 shadow-glow"
              : "text-text-secondary hover:bg-white/10 hover:text-text-primary",
          )}
        >
          <Waves size={14} />
          Ocean Currents
        </button>

        <button
          type="button"
          onClick={onToggleTheme}
          className="glass-soft inline-flex items-center gap-2 rounded-xl px-3 py-2 text-xs font-semibold text-text-primary transition-all duration-300 hover:bg-white/15"
          aria-label="Toggle theme"
        >
          {theme === "dark" ? <Sun size={15} /> : <MoonStar size={15} />}
          {theme === "dark" ? "Light" : "Dark"}
        </button>
      </div>
    </header>
  );
}
