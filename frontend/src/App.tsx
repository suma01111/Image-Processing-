import React, { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { processLab, preprocessImage } from "./services/api";
import type { ImageMode, ProcessLabResponse, PreprocessResponse } from "./services/api";
import { toPngDataUrl } from "./services/image";
import { BeforeAfter } from "./components/BeforeAfter";
import { HistogramPlot } from "./components/HistogramPlot";
import { SpectrumPlot } from "./components/SpectrumPlot";

type LabId = 1 | 2 | 3 | 4 | 5 | 6;

type LabTab = { id: LabId; label: string; subtitle: string; tooltip: string };

const labTabs: LabTab[] = [
  {
    id: 1,
    label: "Calibration & Quantization",
    subtitle: "Sampling / intensity resolution trade-offs",
    tooltip: "Downsample + quantize intensities; compute encoded size vs reconstruction quality.",
  },
  {
    id: 2,
    label: "Geometric Registration",
    subtitle: "Affine warps: scale / rotate / translate / shear",
    tooltip: "Apply a single affine transform matrix via interactive sliders.",
  },
  {
    id: 3,
    label: "Photometric Correction",
    subtitle: "Negative / log / gamma / contrast stretch",
    tooltip: "Transform pixel intensities and use histograms to compare distributions.",
  },
  {
    id: 4,
    label: "Spectral Analysis",
    subtitle: "FFT magnitude + centered low-pass filtering",
    tooltip: "Compute 2D FFT, center spectrum, and suppress high frequencies with a low-pass cutoff.",
  },
  {
    id: 5,
    label: "Image Restoration",
    subtitle: "Noise models + denoising filters; PSNR",
    tooltip: "Add noise, filter it, and track improvement using PSNR.",
  },
  {
    id: 6,
    label: "Colorimetric Mapping",
    subtitle: "Intensity -> color: palettes / continuous / sinusoidal",
    tooltip: "Map grayscale intensity to RGB using multiple professional visualization modes.",
  },
];

function formatMaybeInf(v: any) {
  if (typeof v === "number" && !Number.isFinite(v)) return "∞";
  if (typeof v === "number") return v.toFixed(2);
  return String(v);
}

function isNumericArray(arr: any): arr is number[] {
  return Array.isArray(arr) && arr.length > 0 && arr.every((x) => typeof x === "number");
}

function isNumeric2DArray(arr: any): arr is number[][] {
  return Array.isArray(arr) && arr.length > 0 && arr.every((row) => isNumericArray(row));
}

class ErrorBoundary extends React.Component<{ children: ReactNode; fallback?: ReactNode }, { hasError: boolean; error: Error | null }> {
  constructor(props: { children: ReactNode; fallback?: ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: unknown) {
    console.error("ErrorBoundary caught:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          <p>Component failed to render.</p>
          <p>{String(this.state.error)}</p>
          {this.props.fallback}
        </div>
      );
    }
    return this.props.children;
  }
}

const PRESETS_KEY = "didp_presets_v1";

type Preset = { name: string; paramsByLab: Record<string, any> };

function loadPresets(): Preset[] {
  try {
    const raw = localStorage.getItem(PRESETS_KEY);
    if (!raw) return [];
    return JSON.parse(raw) as Preset[];
  } catch {
    return [];
  }
}

function savePresets(presets: Preset[]) {
  localStorage.setItem(PRESETS_KEY, JSON.stringify(presets));
}

export default function App() {
  const [uploadErr, setUploadErr] = useState<string | null>(null);
  const [preprocess, setPreprocess] = useState<PreprocessResponse | null>(null);
  const [activeLab, setActiveLab] = useState<LabId>(1);
  const [stage, setStage] = useState<"upload" | "modules">("upload");

  const defaultParams = useMemo(() => {
    return {
      1: { spatialResolution: 400, intensityBits: 4 },
      2: {
        scaleX: 1.0,
        scaleY: 1.0,
        rotationDeg: 0,
        translateX: 0,
        translateY: 0,
        shearX: 0,
        shearY: 0,
      },
      3: { negative: false, log: false, gamma: 1.0, contrastStretch: false },
      4: { cutoffRadius: 50.0, spectrumDisplaySize: 256 },
      5: {
        noiseType: "gaussian",
        noiseAmount: 10.0,
        saltPepperAmount: 0.05,
        filterType: "median",
        medianKernel: 3,
        gaussianKernel: 5,
        gaussianSigma: 1.2,
        avgKernel: 3,
        cutoffRadius: 50.0,
      },
      6: {
        colorMode: "intensity_slicing",
        slices: 4,
        sinFrequency: 3.0,
        phaseR: 0,
        phaseG: 2,
        phaseB: 4,
        falseColorGamma: 0.8,
      },
    } as Record<LabId, Record<string, any>>;
  }, []);

  const [paramsByLab, setParamsByLab] = useState<Record<LabId, Record<string, any>>>(defaultParams);
  const activeParams = paramsByLab[activeLab];

  const activeInput = useMemo(() => {
    if (!preprocess) return { imageMode: "grayscale" as ImageMode, imageB64: undefined as string | undefined };
    if (activeLab === 6) return { imageMode: "rgb" as ImageMode, imageB64: preprocess.processed_rgb_b64 };
    return { imageMode: "grayscale" as ImageMode, imageB64: preprocess.processed_gray_b64 };
  }, [preprocess, activeLab]);

  const [labResult, setLabResult] = useState<ProcessLabResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const lastRequestIdRef = useRef(0);

  const [pipelineSteps, setPipelineSteps] = useState<string[]>([]);

  // Presets
  const [presets, setPresets] = useState<Preset[]>([]);
  const [presetName, setPresetName] = useState("");

  useEffect(() => {
    setPresets(loadPresets());
  }, []);

  useEffect(() => {
    if (!preprocess) {
      setPipelineSteps([]);
      return;
    }
    const baseSteps = ["Ingest image", "Normalize (grayscale + centered square crop)"];
    const labStep = `Run ${labTabs.find((t) => t.id === activeLab)?.label ?? `Module ${activeLab}`}`;
    setPipelineSteps([...baseSteps, labStep]);
  }, [preprocess, activeLab]);

  useEffect(() => {
    if (!preprocess) return;
    console.debug("activeLab change", { activeLab, params: paramsByLab[activeLab], inputMode: activeInput.imageMode });
  }, [activeLab, preprocess, paramsByLab, activeInput.imageMode]);

  // Real-time lab processing (debounced).
  useEffect(() => {
    if (!activeInput.imageB64) return;
    const requestId = ++lastRequestIdRef.current;
    setLoading(true);
    setError(null);

    const t = window.setTimeout(async () => {
      try {
        const resp = await processLab(activeLab, activeInput.imageB64!, activeInput.imageMode, activeParams);
        if (requestId !== lastRequestIdRef.current) return; // stale response
        setLabResult(resp);
      } catch (e: any) {
        if (requestId !== lastRequestIdRef.current) return;
        setError(e?.message ?? String(e));
        setLabResult(null);
      } finally {
        if (requestId === lastRequestIdRef.current) setLoading(false);
      }
    }, 180);

    return () => {
      window.clearTimeout(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeLab, activeInput.imageMode, activeInput.imageB64, JSON.stringify(activeParams)]);

  async function onUpload(file: File | null) {
    setUploadErr(null);
    setPreprocess(null);
    setLabResult(null);
    if (!file) return;
    try {
      const resp = await preprocessImage(file);
      setPreprocess(resp);
      setStage("modules");
    } catch (e: any) {
      setUploadErr(e?.message ?? String(e));
    }
  }

  const beforeSrc = useMemo(() => {
    if (!preprocess) return undefined;
    if (activeLab === 6) return toPngDataUrl(preprocess.processed_rgb_b64);
    return toPngDataUrl(preprocess.processed_gray_b64);
  }, [preprocess, activeLab]);

  const afterSrc = useMemo(() => {
    if (!labResult?.image_b64) return undefined;
    return toPngDataUrl(labResult.image_b64);
  }, [labResult]);

  function downloadProcessed() {
    if (!labResult?.image_b64) return;
    const a = document.createElement("a");
    a.href = toPngDataUrl(labResult.image_b64)!;
    a.download = `didp_${activeLab}.png`;
    a.click();
  }

  function setParam(labId: LabId, key: string, value: any) {
    setParamsByLab((prev) => ({ ...prev, [labId]: { ...prev[labId], [key]: value } }));
  }

  function savePreset() {
    const name = presetName.trim();
    if (!name) return;
    const toSave: Preset = { name, paramsByLab: Object.fromEntries(Object.entries(paramsByLab).map(([k, v]) => [k, v])) };
    const next = [...presets.filter((p) => p.name !== name), toSave];
    setPresets(next);
    savePresets(next);
    setPresetName("");
  }

  function applyPreset(p: Preset) {
    const next: any = { ...paramsByLab };
    for (const [k, v] of Object.entries(p.paramsByLab)) {
      const labId = Number(k) as LabId;
      if (labId >= 1 && labId <= 6) next[labId] = v as Record<string, any>;
    }
    setParamsByLab(next);
  }

  const labExplanation = useMemo(() => {
    switch (activeLab) {
      case 1:
        return "Quantization reduces intensity resolution and compresses the image representation. Together with sampling, it trades detail for encoded size.";
      case 2:
        return "Affine registration combines scale, rotation, translation, and shear into one geometric model used for image warping.";
      case 3:
        return "Photometric transforms reshape pixel intensities. Histograms provide immediate feedback about how contrast and brightness change.";
      case 4:
        return "In the frequency domain, low-pass filters preserve smooth components while suppressing edges and noise contributions.";
      case 5:
        return "Restoration attempts to recover the clean structure from corrupted observations. PSNR quantifies the restoration improvement.";
      case 6:
        return "Color mapping converts intensity into RGB using different models (palette, continuous mapping, sinusoidal mapping, composites).";
      default:
        return "";
    }
  }, [activeLab]);

  const activeTab = labTabs.find((t) => t.id === activeLab)!;

  function getLabButtonClass(id: LabId, active: boolean) {
    if (!active) return "border-slate-200 bg-white text-slate-700 hover:bg-slate-50";
    switch (id) {
      case 1:
        return "border-indigo-600 bg-indigo-50 text-indigo-700";
      case 2:
        return "border-cyan-600 bg-cyan-50 text-cyan-700";
      case 3:
        return "border-emerald-600 bg-emerald-50 text-emerald-700";
      case 4:
        return "border-violet-600 bg-violet-50 text-violet-700";
      case 5:
        return "border-rose-600 bg-rose-50 text-rose-700";
      case 6:
        return "border-fuchsia-600 bg-fuchsia-50 text-fuchsia-700";
      default:
        return "border-indigo-600 bg-indigo-50 text-indigo-700";
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-indigo-50/30 to-white p-4">
      <div className="mx-auto max-w-6xl">
        <header className="mb-4 rounded-2xl border border-slate-200 bg-white/70 p-4 shadow-sm">
          <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
            <div>
              <div className="text-2xl font-bold text-slate-900">Interactive Digital Image Processing Console</div>
              <div className="text-sm text-slate-600">
                Gonzalez & Woods-inspired pipeline (modules 01-06) - real-time server processing
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-700">
                {preprocess ? `Loaded: ${preprocess.crop_side}×${preprocess.crop_side} crop` : "Upload an image to begin"}
              </span>
              {labResult && (
                <button
                  className="rounded-xl bg-indigo-600 px-3 py-2 text-sm font-semibold text-white hover:bg-indigo-700 disabled:opacity-50"
                  onClick={downloadProcessed}
                >
                  Download
                </button>
              )}
            </div>
          </div>
        </header>

        <section className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <div className={stage === "modules" ? "hidden" : "md:col-span-3"}>
            <div className="rounded-2xl border border-slate-200 bg-white/70 p-4 shadow-sm">
              <div className="mb-3 flex items-center justify-between">
                <div className="text-sm font-semibold text-slate-800">Image Input</div>
                {preprocess ? (
                  <span className="rounded-full bg-emerald-50 px-2 py-1 text-[11px] font-semibold text-emerald-700">
                    Ready
                  </span>
                ) : (
                  <span className="rounded-full bg-slate-100 px-2 py-1 text-[11px] font-semibold text-slate-700">
                    Waiting
                  </span>
                )}
              </div>
              <input
                type="file"
                accept="image/png,image/jpeg"
                className="w-full cursor-pointer rounded-xl border border-slate-200 bg-white p-2 text-sm shadow-sm"
                onChange={(e) => onUpload(e.target.files?.[0] ?? null)}
              />
              {uploadErr && <div className="mt-3 rounded-lg bg-red-50 p-3 text-sm text-red-700">{uploadErr}</div>}

              <div className="mt-4">
                <div className="text-xs text-slate-600">Pipeline</div>
                <ol className="mt-2 space-y-1 text-sm text-slate-800">
                  {pipelineSteps.map((s, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span
                        className={`mt-0.5 inline-block h-2 w-2 rounded-full ${
                          idx === pipelineSteps.length - 1 ? "bg-indigo-600" : "bg-slate-300"
                        }`}
                      />
                      <span className={idx === pipelineSteps.length - 1 ? "font-semibold text-slate-900" : ""}>{s}</span>
                    </li>
                  ))}
                </ol>
              </div>

              <div className="mt-4 rounded-xl border border-slate-200 bg-white p-3">
                <div className="text-sm font-semibold text-slate-800">Presets</div>
                <div className="mt-2 flex items-center gap-2">
                  <input
                    value={presetName}
                    onChange={(e) => setPresetName(e.target.value)}
                    placeholder="Name preset"
                    className="w-full rounded-lg border border-slate-200 bg-white p-2 text-sm"
                  />
                  <button
                    className="rounded-lg bg-slate-900 px-3 py-2 text-sm font-semibold text-white hover:bg-slate-800"
                    onClick={savePreset}
                    disabled={!presetName.trim()}
                  >
                    Save
                  </button>
                </div>
                <div className="mt-3 flex flex-col gap-2">
                  {presets.length === 0 ? (
                    <div className="text-xs text-slate-500">No presets saved yet.</div>
                  ) : (
                    presets.slice(0, 6).map((p) => (
                      <button
                        key={p.name}
                        className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-left text-sm hover:bg-slate-50"
                        onClick={() => applyPreset(p)}
                        title="Apply saved parameters"
                      >
                        {p.name}
                      </button>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>

          {stage === "modules" && (
            <div className="md:col-span-3">
            <div className="rounded-2xl border border-slate-200 bg-white/70 p-4 shadow-sm">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                <div>
                  <div className="text-sm font-semibold text-slate-800">Modules</div>
                  <div className="mt-1 text-xs font-semibold text-slate-700">{activeTab.label}</div>
                  <div className="mt-1 text-xs text-slate-600">{activeTab.subtitle}</div>
                  <div className="mt-1 text-xs text-slate-600">{labExplanation}</div>
                </div>
                <div className="flex flex-wrap gap-2">
                  {labTabs.map((t) => (
                    <button
                      key={t.id}
                      className={`rounded-full border px-4 py-2 text-xs font-semibold ${getLabButtonClass(t.id, activeLab === t.id)}`}
                      onClick={() => setActiveLab(t.id)}
                      title={`${t.label} — ${t.tooltip}`}
                    >
                      {t.id.toString().padStart(2, "0")} · {t.label}
                    </button>
                  ))}
                </div>
              </div>

              {preprocess && (
                <div className="mb-4 rounded-2xl border border-slate-200 bg-white p-4">
                  <div className="mb-3 flex items-center justify-between gap-3">
                    <div className="text-sm font-semibold text-slate-800">Image Input & Preprocessing</div>
                    <span className="rounded-full bg-indigo-50 px-2 py-1 text-[11px] font-semibold text-indigo-700">
                      Crop {preprocess.crop_side}×{preprocess.crop_side}
                    </span>
                  </div>
                  <BeforeAfter
                    beforeLabel="Original"
                    afterLabel="Normalized: grayscale + centered square crop"
                    beforeSrc={toPngDataUrl(preprocess.original_b64)}
                    afterSrc={toPngDataUrl(preprocess.processed_gray_b64)}
                  />
                </div>
              )}

              <div className="grid grid-cols-1 gap-4 lg:grid-cols-5">
                <div className="lg:col-span-2">
                  <div
                    className={`rounded-2xl border bg-white p-4 ${
                      activeLab === 1
                        ? "border-indigo-200"
                        : activeLab === 2
                          ? "border-cyan-200"
                          : activeLab === 3
                            ? "border-emerald-200"
                            : activeLab === 4
                              ? "border-violet-200"
                              : activeLab === 5
                                ? "border-rose-200"
                                : "border-fuchsia-200"
                    }`}
                  >
                    <div className="mb-3 flex items-center justify-between">
                      <div className="text-sm font-semibold text-slate-800">Controls</div>
                      {loading ? (
                        <span className="rounded-full bg-indigo-50 px-2 py-1 text-[11px] font-semibold text-indigo-700">
                          Live
                        </span>
                      ) : (
                        <span className="rounded-full bg-slate-100 px-2 py-1 text-[11px] font-semibold text-slate-700">
                          Ready
                        </span>
                      )}
                    </div>
                    <div className="mb-4 rounded-xl border border-slate-200 bg-slate-50 p-3">
                      <div className="text-sm font-bold text-slate-900">{activeTab.label}</div>
                      <div className="mt-1 text-xs text-slate-600">{activeTab.tooltip}</div>
                    </div>

                    {activeLab === 1 && (
                      <div className="space-y-4">
                        <div>
                          <div className="flex items-center justify-between text-sm text-slate-700">
                            <span>Spatial Resolution</span>
                            <span className="font-semibold text-slate-900">{paramsByLab[1].spatialResolution}</span>
                          </div>
                          <select
                            className="mt-2 w-full rounded-lg border border-slate-200 bg-white p-2 text-sm"
                            value={paramsByLab[1].spatialResolution}
                            onChange={(e) => setParam(1, "spatialResolution", Number(e.target.value))}
                          >
                            {[100, 200, 400, 800].map((v) => (
                              <option key={v} value={v}>
                                {v}
                              </option>
                            ))}
                          </select>
                        </div>
                        <div>
                          <div className="flex items-center justify-between text-sm text-slate-700">
                            <span>Intensity Levels</span>
                            <span className="font-semibold text-slate-900">{paramsByLab[1].intensityBits} bits</span>
                          </div>
                          <select
                            className="mt-2 w-full rounded-lg border border-slate-200 bg-white p-2 text-sm"
                            value={paramsByLab[1].intensityBits}
                            onChange={(e) => setParam(1, "intensityBits", Number(e.target.value))}
                          >
                            {[1, 2, 4, 8].map((v) => (
                              <option key={v} value={v}>
                                {v} bits
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    )}

                    {activeLab === 2 && (
                      <div className="space-y-4">
                        <Slider label="Scaling X" value={paramsByLab[2].scaleX} min={0.5} max={2.0} step={0.01} onChange={(v) => setParam(2, "scaleX", v)} />
                        <Slider label="Scaling Y" value={paramsByLab[2].scaleY} min={0.5} max={2.0} step={0.01} onChange={(v) => setParam(2, "scaleY", v)} />
                        <Slider label="Rotation (deg)" value={paramsByLab[2].rotationDeg} min={-180} max={180} step={1} onChange={(v) => setParam(2, "rotationDeg", v)} />
                        <Slider label="Translation X" value={paramsByLab[2].translateX} min={-100} max={100} step={1} onChange={(v) => setParam(2, "translateX", v)} />
                        <Slider label="Translation Y" value={paramsByLab[2].translateY} min={-100} max={100} step={1} onChange={(v) => setParam(2, "translateY", v)} />
                        <Slider label="Shear X" value={paramsByLab[2].shearX} min={-1} max={1} step={0.01} onChange={(v) => setParam(2, "shearX", v)} />
                        <Slider label="Shear Y" value={paramsByLab[2].shearY} min={-1} max={1} step={0.01} onChange={(v) => setParam(2, "shearY", v)} />
                      </div>
                    )}

                    {activeLab === 3 && (
                      <div className="space-y-4">
                        <Toggle label="Negative" value={paramsByLab[3].negative} onChange={(v) => setParam(3, "negative", v)} />
                        <Toggle label="Log Transform" value={paramsByLab[3].log} onChange={(v) => setParam(3, "log", v)} />
                        <Slider label="Gamma" value={paramsByLab[3].gamma} min={0.2} max={3.0} step={0.01} onChange={(v) => setParam(3, "gamma", v)} />
                        <Toggle label="Contrast Stretch (Min-Max)" value={paramsByLab[3].contrastStretch} onChange={(v) => setParam(3, "contrastStretch", v)} />
                        <div className="text-xs text-slate-500">
                          Tip: try enabling <b>Negative</b> or <b>Log Transform</b> first.
                        </div>
                      </div>
                    )}

                    {activeLab === 4 && (
                      <div className="space-y-4">
                        <Slider
                          label="Low-pass cutoff D0"
                          value={paramsByLab[4].cutoffRadius}
                          min={5}
                          max={Math.max(10, (preprocess?.crop_side ?? 256) / 2)}
                          step={1}
                          onChange={(v) => setParam(4, "cutoffRadius", v)}
                        />
                      </div>
                    )}

                    {activeLab === 5 && (
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-3">
                          <div>
                            <div className="text-xs font-semibold text-slate-700">Noise Type</div>
                            <select className="mt-1 w-full rounded-lg border border-slate-200 bg-white p-2 text-sm" value={paramsByLab[5].noiseType} onChange={(e) => setParam(5, "noiseType", e.target.value)}>
                              <option value="gaussian">Gaussian</option>
                              <option value="salt_pepper">Salt & Pepper</option>
                              <option value="uniform">Uniform</option>
                            </select>
                          </div>
                          <div>
                            <div className="text-xs font-semibold text-slate-700">Filter</div>
                            <select className="mt-1 w-full rounded-lg border border-slate-200 bg-white p-2 text-sm" value={paramsByLab[5].filterType} onChange={(e) => setParam(5, "filterType", e.target.value)}>
                              <option value="median">Median</option>
                              <option value="gaussian">Gaussian</option>
                              <option value="average">Average</option>
                              <option value="freq_lowpass">Frequency Low-pass</option>
                            </select>
                          </div>
                        </div>

                        {paramsByLab[5].noiseType === "gaussian" && (
                          <Slider label="Gaussian sigma" value={paramsByLab[5].noiseAmount} min={0} max={50} step={0.5} onChange={(v) => setParam(5, "noiseAmount", v)} />
                        )}
                        {paramsByLab[5].noiseType === "uniform" && (
                          <Slider label="Uniform range a" value={paramsByLab[5].noiseAmount} min={0} max={50} step={0.5} onChange={(v) => setParam(5, "noiseAmount", v)} />
                        )}
                        {paramsByLab[5].noiseType === "salt_pepper" && (
                          <Slider label="Salt/Pepper amount" value={paramsByLab[5].saltPepperAmount} min={0} max={0.3} step={0.005} onChange={(v) => setParam(5, "saltPepperAmount", v)} />
                        )}

                        {paramsByLab[5].filterType === "median" && (
                          <Slider label="Median kernel" value={paramsByLab[5].medianKernel} min={1} max={11} step={2} onChange={(v) => setParam(5, "medianKernel", Math.round(v))} />
                        )}
                        {paramsByLab[5].filterType === "gaussian" && (
                          <>
                            <Slider label="Gaussian sigma" value={paramsByLab[5].gaussianSigma} min={0.1} max={5} step={0.05} onChange={(v) => setParam(5, "gaussianSigma", v)} />
                            <Slider label="Gaussian kernel" value={paramsByLab[5].gaussianKernel} min={3} max={15} step={2} onChange={(v) => setParam(5, "gaussianKernel", Math.round(v))} />
                          </>
                        )}
                        {paramsByLab[5].filterType === "average" && (
                          <Slider label="Average kernel" value={paramsByLab[5].avgKernel} min={1} max={11} step={2} onChange={(v) => setParam(5, "avgKernel", Math.round(v))} />
                        )}
                        {paramsByLab[5].filterType === "freq_lowpass" && (
                          <Slider
                            label="Frequency cutoff"
                            value={paramsByLab[5].cutoffRadius}
                            min={5}
                            max={Math.max(10, (preprocess?.crop_side ?? 256) / 2)}
                            step={1}
                            onChange={(v) => setParam(5, "cutoffRadius", v)}
                          />
                        )}
                      </div>
                    )}

                    {activeLab === 6 && (
                      <div className="space-y-4">
                        <div>
                          <div className="text-xs font-semibold text-slate-700">Color Mode</div>
                          <select className="mt-1 w-full rounded-lg border border-slate-200 bg-white p-2 text-sm" value={paramsByLab[6].colorMode} onChange={(e) => setParam(6, "colorMode", e.target.value)}>
                            <option value="intensity_slicing">Intensity Slicing (palette)</option>
                            <option value="continuous_mapping">Continuous Color Mapping</option>
                            <option value="rgb_sinusoidal">RGB Sinusoidal Mapping</option>
                            <option value="false_color">False-color Composite</option>
                          </select>
                        </div>
                        {paramsByLab[6].colorMode === "intensity_slicing" && (
                          <Slider label="Slices" value={paramsByLab[6].slices} min={2} max={10} step={1} onChange={(v) => setParam(6, "slices", Math.round(v))} />
                        )}
                        {paramsByLab[6].colorMode === "rgb_sinusoidal" && (
                          <>
                            <Slider label="Frequency" value={paramsByLab[6].sinFrequency} min={0.5} max={10} step={0.05} onChange={(v) => setParam(6, "sinFrequency", v)} />
                            <Slider label="Phase R" value={paramsByLab[6].phaseR} min={-6.28} max={6.28} step={0.1} onChange={(v) => setParam(6, "phaseR", v)} />
                          </>
                        )}
                        {paramsByLab[6].colorMode === "false_color" && (
                          <Slider label="Gamma" value={paramsByLab[6].falseColorGamma} min={0.2} max={2} step={0.01} onChange={(v) => setParam(6, "falseColorGamma", v)} />
                        )}
                      </div>
                    )}

                    {loading && <div className="mt-3 text-sm text-slate-600">Processing...</div>}
                    {error && <div className="mt-3 rounded-lg bg-red-50 p-3 text-sm text-red-700">{error}</div>}
                  </div>

                  {activeLab === 1 && labResult?.encoded && (
                    <div className="mt-4 rounded-2xl border border-slate-200 bg-white p-4">
                      <div className="text-sm font-semibold text-slate-800">Encoded Size</div>
                      <div className="mt-2 grid grid-cols-2 gap-3 text-sm text-slate-700">
                        <div>
                          <div className="text-xs text-slate-500">Spatial</div>
                          <div className="font-semibold">{labResult.encoded.spatialResolution}x{labResult.encoded.spatialResolution}</div>
                        </div>
                        <div>
                          <div className="text-xs text-slate-500">Intensity</div>
                          <div className="font-semibold">{labResult.encoded.intensityBits} bits</div>
                        </div>
                        <div className="col-span-2">
                          <div className="text-xs text-slate-500">Encoded bits</div>
                          <div className="font-semibold">{labResult.encoded.encodedBits.toLocaleString()} bits</div>
                        </div>
                      </div>
                    </div>
                  )}

                  {activeLab === 5 && labResult?.metrics && (
                    <div className="mt-4 rounded-2xl border border-slate-200 bg-white p-4">
                      <div className="text-sm font-semibold text-slate-800">PSNR (Real-time)</div>
                      <div className="mt-2 grid grid-cols-2 gap-3 text-sm text-slate-700">
                        <div>
                          <div className="text-xs text-slate-500">Noisy</div>
                          <div className="font-semibold">{formatMaybeInf(labResult.metrics.psnrNoisy)}</div>
                        </div>
                        <div>
                          <div className="text-xs text-slate-500">Denoised</div>
                          <div className="font-semibold">{formatMaybeInf(labResult.metrics.psnrDenoised)}</div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                <div className="lg:col-span-3">
                  <BeforeAfter
                    beforeLabel={activeLab === 6 ? "Before (RGB crop)" : "Before (grayscale crop)"}
                    afterLabel={activeLab === 6 ? "After (color processing)" : "After processing"}
                    beforeSrc={beforeSrc}
                    afterSrc={afterSrc}
                  />

                  {activeLab === 3 && (
                    <div className="mt-4 rounded-2xl border border-slate-200 bg-white p-4">
                      <div className="mb-2 text-sm font-semibold text-slate-800">Histogram (Before vs After)</div>
                      {labResult?.histogram?.before && labResult?.histogram?.after && isNumericArray(labResult.histogram.before) && isNumericArray(labResult.histogram.after) ? (
                        <ErrorBoundary>
                          <HistogramPlot before={labResult.histogram.before} after={labResult.histogram.after} />
                        </ErrorBoundary>
                      ) : (
                        <div className="text-sm text-slate-500">Histogram data unavailable yet. Run the module by uploading an image first.</div>
                      )}
                    </div>
                  )}

                  {activeLab === 4 && (
                    <div className="mt-4 rounded-2xl border border-slate-200 bg-white p-4">
                      <div className="mb-2 text-sm font-semibold text-slate-800">Magnitude Spectrum</div>
                      {labResult?.spectrum?.magnitudeBefore && isNumeric2DArray(labResult.spectrum.magnitudeBefore) && (!labResult.spectrum.magnitudeAfter || isNumeric2DArray(labResult.spectrum.magnitudeAfter)) ? (
                        <ErrorBoundary>
                          <SpectrumPlot before={labResult.spectrum.magnitudeBefore} after={labResult.spectrum.magnitudeAfter} />
                        </ErrorBoundary>
                      ) : (
                        <div className="text-sm text-slate-500">Spectrum data unavailable yet. Run the module by uploading an image first.</div>
                      )}
                      <div className="mt-2 text-xs text-slate-500">Low-pass cutoff D0: {labResult?.spectrum?.cutoffRadius ?? "--"}</div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
        </section>

        <footer className="mt-6 text-center text-xs text-slate-500">
          Educational demo: computations are performed server-side (OpenCV + NumPy) and returned as images + metrics.
        </footer>
      </div>
    </div>
  );
}

function Slider(props: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void }) {
  return (
    <div>
      <div className="flex items-center justify-between text-sm text-slate-700">
        <span>{props.label}</span>
        <span className="font-semibold text-slate-900">{props.value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(e) => props.onChange(Number(e.target.value))}
        className="mt-2 w-full"
      />
    </div>
  );
}

function Toggle(props: { label: string; value: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex cursor-pointer items-center justify-between rounded-xl border border-slate-200 bg-white p-3">
      <span className="text-sm font-semibold text-slate-800">{props.label}</span>
      <input type="checkbox" className="h-4 w-4" checked={props.value} onChange={(e) => props.onChange(e.target.checked)} />
    </label>
  );
}
