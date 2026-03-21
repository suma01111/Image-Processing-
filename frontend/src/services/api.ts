export type ImageMode = "grayscale" | "rgb";

export type PreprocessResponse = {
  original_b64: string;
  processed_gray_b64: string;
  processed_rgb_b64: string;
  original_width: number;
  original_height: number;
  crop_side: number;
};

export type ProcessLabResponse = {
  image_b64: string;
  image_mode: ImageMode;
  [key: string]: any;
};

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? "http://localhost:8000";

export async function preprocessImage(file: File): Promise<PreprocessResponse> {
  const form = new FormData();
  form.append("image", file);
  const res = await fetch(`${BACKEND_URL}/api/preprocess`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(`Preprocess failed: ${res.status}`);
  return res.json();
}

export async function processLab(
  labId: number,
  imageB64: string,
  imageMode: ImageMode,
  params: Record<string, any>,
): Promise<ProcessLabResponse> {
  const form = new FormData();
  form.append("image_b64", imageB64);
  form.append("image_mode", imageMode);
  form.append("params_json", JSON.stringify(params ?? {}));

  const res = await fetch(`${BACKEND_URL}/api/labs/${labId}/process`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Lab ${labId} failed: ${res.status} ${text}`);
  }
  return res.json();
}

