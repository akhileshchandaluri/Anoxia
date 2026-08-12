import { AlertTriangle, Anchor, Fish, MapPin } from "lucide-react";
import { clsx } from "clsx";

function barTone(pct) {
  if (pct >= 80) {
    return {
      text: "text-red-300",
      bar: "bg-red-500",
      track: "bg-red-950/25",
    };
  }
  if (pct >= 50) {
    return {
      text: "text-amber-300",
      bar: "bg-amber-500",
      track: "bg-amber-950/25",
    };
  }
  if (pct >= 30) {
    return {
      text: "text-cyan-300",
      bar: "bg-cyan-500",
      track: "bg-cyan-950/25",
    };
  }
  return {
    text: "text-emerald-300",
    bar: "bg-emerald-500",
    track: "bg-emerald-950/25",
  };
}

function priorityTone(priority) {
  switch (priority) {
    case "CRITICAL":
      return "border-red-400/60 bg-red-500/15 text-red-100";
    case "URGENT":
      return "border-orange-400/60 bg-orange-500/15 text-orange-100";
    case "WARNING":
      return "border-yellow-400/60 bg-yellow-500/15 text-yellow-100";
    case "HIGH":
      return "border-violet-400/60 bg-violet-500/15 text-violet-100";
    default:
      return "border-emerald-400/60 bg-emerald-500/15 text-emerald-100";
  }
}

export function SidebarPanel({ data, selectedZone, onChangeZone, precursorState }) {
  const zones = data?.zones || [];
  const paths = data?.drift_paths || [];

  return (
    <aside className="panel flex h-full w-full flex-col gap-3 overflow-hidden rounded-2xl p-3 lg:w-[32%]">
      <section className="panel-soft rounded-xl p-3">
        <label className="mb-2 block text-[11px] font-semibold uppercase tracking-[0.14em] text-text-muted">
          Select Zone To Inspect
        </label>
        <select
          value={selectedZone}
          onChange={(event) => onChangeZone(event.target.value)}
          className="w-full rounded-lg border border-white/15 bg-black/25 px-3 py-2 text-sm font-medium text-text-primary outline-none transition-all duration-300 focus:border-cyan-300/70 focus:ring-2 focus:ring-cyan-400/25"
        >
          {zones.map((zone) => (
            <option key={zone.name} value={zone.name} className="bg-slate-900 text-white">
              {zone.name}
            </option>
          ))}
        </select>
      </section>

      <section className="panel-soft min-h-[260px] flex-1 overflow-auto rounded-xl p-3">
        <div className="mb-3 flex items-center justify-between gap-3">
          <h3 className="text-xs font-bold uppercase tracking-[0.14em] text-text-primary">Precursor Conditions</h3>
          <span className="inline-flex items-center gap-1 text-[11px] text-text-muted">
            <MapPin size={12} />
            {precursorState.locationLabel || "Zone baseline"}
          </span>
        </div>

        {precursorState.loading ? (
          <div className="rounded-lg border border-cyan-300/40 bg-cyan-500/10 px-3 py-2 text-xs text-cyan-100">
            Loading conditions from backend...
          </div>
        ) : null}

        <div className="space-y-3">
          {precursorState.bars.map(([label, value, pct]) => {
            const tone = barTone(Number(pct));
            return (
              <div key={`${label}-${value}`}>
                <div className="mb-1 flex items-center justify-between gap-3">
                  <span className="text-xs font-medium text-text-primary">{label}</span>
                  <span className={clsx("text-xs font-semibold", tone.text)}>{value}</span>
                </div>
                <div className={clsx("h-1.5 overflow-hidden rounded-full", tone.track)}>
                  <div
                    className={clsx("h-full rounded-full transition-all duration-500", tone.bar)}
                    style={{ width: `${Math.max(0, Math.min(100, Number(pct)))}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>

        <div className="mt-4 border-t border-white/10 pt-3">
          <h4 className="mb-2 inline-flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-text-muted">
            <AlertTriangle size={12} />
            Dynamic Interventions
          </h4>

          {precursorState.interventions.length === 0 ? (
            <p className="text-xs text-text-secondary">
              Click a map location to fetch intervention recommendations from backend analysis.
            </p>
          ) : (
            <div className="space-y-2">
              {precursorState.interventions.map((item, index) => (
                <article
                  key={`${item.title}-${index}`}
                  className={clsx(
                    "rounded-lg border p-2.5",
                    priorityTone(item.priority),
                  )}
                >
                  <div className="mb-1 flex items-center justify-between gap-2">
                    <h5 className="text-xs font-semibold">{item.title}</h5>
                    <span className="rounded-full bg-black/25 px-2 py-0.5 text-[10px] font-bold">
                      {item.priority || "ROUTINE"}
                    </span>
                  </div>
                  <p className="mb-1 text-[11px] opacity-90">{item.reason}</p>
                  <p className="text-[11px] opacity-75">Timeline: {item.timeline}</p>
                </article>
              ))}
            </div>
          )}
        </div>
      </section>

      <section className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-1">
        <div className="panel-soft rounded-xl p-3">
          <h3 className="mb-2 inline-flex items-center gap-2 text-xs font-bold uppercase tracking-[0.14em] text-text-primary">
            <Fish size={13} />
            Active Threat Zones
          </h3>
          <div className="space-y-2">
            {zones.map((zone) => (
              <div key={zone.name} className="rounded-lg border border-white/10 bg-black/20 px-2.5 py-2">
                <div className="mb-1 flex items-center justify-between gap-2">
                  <span className="text-xs font-semibold text-text-primary">{zone.name}</span>
                  <span
                    className={clsx(
                      "rounded-full px-2 py-0.5 text-[10px] font-bold",
                      zone.status === "CRITICAL" ? "bg-red-500/25 text-red-100" : "bg-amber-500/25 text-amber-100",
                    )}
                  >
                    {zone.status}
                  </span>
                </div>
                <div className="flex flex-wrap gap-2 text-[11px] text-text-secondary">
                  <span>DO {zone.do} mg/L</span>
                  <span>{zone.days}d window</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="panel-soft rounded-xl p-3">
          <h3 className="mb-2 inline-flex items-center gap-2 text-xs font-bold uppercase tracking-[0.14em] text-text-primary">
            <Anchor size={13} />
            Ghost Gear Paths
          </h3>
          <div className="space-y-1.5">
            {paths.map((path) => (
              <div key={path.id} className="flex items-center justify-between rounded-md border border-white/10 bg-black/20 px-2 py-1.5 text-xs">
                <span className="font-semibold text-text-primary">Path {path.id}</span>
                <span className="text-text-secondary">{path.lats.length} waypoints</span>
              </div>
            ))}
          </div>
        </div>
      </section>
    </aside>
  );
}
