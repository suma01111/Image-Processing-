import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";

export function SpectrumPlot(props: { before: number[][]; after?: number[][] }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const data: any[] = [
      {
        z: props.before,
        type: "heatmap",
        colorscale: "Viridis",
        showscale: false,
      },
    ];

    if (props.after) {
      data.push({
        z: props.after,
        type: "heatmap",
        colorscale: "Viridis",
        showscale: false,
        opacity: 0.65,
      });
    }

    const layout = {
      width: 520,
      height: 340,
      margin: { l: 35, r: 10, t: 25, b: 25 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
    };

    Plotly.newPlot(containerRef.current, data, layout, { displayModeBar: false });

    return () => {
      if (containerRef.current) {
        Plotly.purge(containerRef.current);
      }
    };
  }, [props.before, props.after]);

  return <div ref={containerRef} />;
}

