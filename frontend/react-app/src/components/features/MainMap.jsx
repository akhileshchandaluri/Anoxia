import { useCallback, useEffect, useRef } from "react";
import Plot from "../common/PlotlyChart";

export function MainMap({ figure, onMapCoordinateClick }) {
  const mapRef = useRef(null);
  const mapClickHandlerRef = useRef(null);

  const detachMapClick = useCallback(() => {
    if (mapRef.current && mapClickHandlerRef.current) {
      mapRef.current.off("click", mapClickHandlerRef.current);
    }
    mapClickHandlerRef.current = null;
    mapRef.current = null;
  }, []);

  const attachMapClick = useCallback(
    (graphDiv) => {
      const map = graphDiv?._fullLayout?.mapbox?._subplot?.map;
      if (!map || !onMapCoordinateClick) {
        return;
      }

      if (mapRef.current === map && mapClickHandlerRef.current) {
        return;
      }

      detachMapClick();

      const handler = (event) => {
        const lat = Number(event?.lngLat?.lat);
        const lon = Number(event?.lngLat?.lng);

        if (Number.isFinite(lat) && Number.isFinite(lon)) {
          onMapCoordinateClick(lat, lon);
        }
      };

      map.on("click", handler);
      mapRef.current = map;
      mapClickHandlerRef.current = handler;
    },
    [detachMapClick, onMapCoordinateClick],
  );

  useEffect(() => () => detachMapClick(), [detachMapClick]);

  return (
    <section className="panel relative min-h-[420px] flex-1 overflow-hidden rounded-2xl lg:min-h-0">
      <Plot
        data={figure.data}
        layout={figure.layout}
        config={figure.config}
        onInitialized={(fig, graphDiv) => attachMapClick(graphDiv)}
        onUpdate={(fig, graphDiv) => attachMapClick(graphDiv)}
        className="h-full w-full"
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
      />

      <div className="pointer-events-none absolute bottom-3 left-3 rounded-md border border-white/20 bg-black/50 px-3 py-2 text-[11px] font-medium text-white backdrop-blur">
        Click anywhere to see backend P(hypoxia) near cursor + live intervention analytics
      </div>
    </section>
  );
}
