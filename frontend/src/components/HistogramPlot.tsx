import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";

export function HistogramPlot(props: { before: number[]; after: number[] }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const x = Array.from({ length: 256 }, (_, i) => i);

  useEffect(() => {
    if (!containerRef.current) return;

    const data = [
      { x, y: props.before, type: "bar" as const, name: "Before", marker: { color: "rgba(59,130,246,0.6)" } },
      { x, y: props.after, type: "bar" as const, name: "After", marker: { color: "rgba(34,197,94,0.6)" } },
    ];

    const layout = {
      width: 520,
      height: 300,
      margin: { l: 35, r: 10, t: 30, b: 35 },
      barmode: "overlay" as const,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      xaxis: { title: "Intensity (0-255)" },
      yaxis: { title: "Count" },
      legend: { orientation: "h" as const },
    };

    Plotly.newPlot(containerRef.current, data as any, layout, { displayModeBar: false });

    return () => {
      if (containerRef.current) {
        Plotly.purge(containerRef.current);
      }
    };
  }, [props.before, props.after]);

  return <div ref={containerRef} />;
}

