import Plot from "../common/PlotlyChart";

export function FlowPathwaysPanel({ figure, loading, error }) {
  return (
    <section className="panel flex h-[32vh] min-h-[280px] flex-col gap-2 rounded-2xl p-3">
      <header className="panel-soft rounded-xl px-3 py-2">
        <h2 className="font-display text-base font-semibold text-text-primary md:text-lg">
          Runoff to Hypoxia Flow Pathways
        </h2>
        <p className="text-xs font-medium text-text-secondary">
          Nutrient transport from river discharge to hypoxic dead-zone formation
        </p>
      </header>

      <div className="grid min-h-0 flex-1 gap-2 lg:grid-cols-[1fr_auto]">
        <div className="relative min-h-0 overflow-hidden rounded-xl border border-white/15 bg-black/20">
          <Plot
            data={figure.data}
            layout={figure.layout}
            config={figure.config}
            className="h-full w-full"
            useResizeHandler
            style={{ width: "100%", height: "100%" }}
          />

          {loading ? (
            <div className="absolute left-3 top-3 rounded-md border border-cyan-300/50 bg-cyan-500/20 px-3 py-1.5 text-[11px] font-semibold text-cyan-100">
              Loading pathways...
            </div>
          ) : null}

          {error ? (
            <div className="absolute bottom-3 left-3 rounded-md border border-red-400/60 bg-red-500/20 px-3 py-1.5 text-[11px] font-semibold text-red-100">
              {error}
            </div>
          ) : null}
        </div>

        <aside className="panel-soft flex min-w-[220px] flex-row flex-wrap gap-2 rounded-xl p-2 lg:flex-col">
          <LegendTone color="bg-amber-400" text="Weak (0-33%)" />
          <LegendTone color="bg-orange-500" text="Medium (33-66%)" />
          <LegendTone color="bg-red-500" text="Strong (66-100%)" />
          <LegendTone color="bg-emerald-500" text="River source points" />
          <LegendTone color="bg-red-600" text="Dead-zone endpoints" />
        </aside>
      </div>
    </section>
  );
}

function LegendTone({ color, text }) {
  return (
    <div className="inline-flex items-center gap-2 rounded-md border border-white/10 bg-black/20 px-2 py-1.5 text-xs font-medium text-text-primary">
      <span className={`h-3 w-3 rounded-sm ${color}`} />
      {text}
    </div>
  );
}
