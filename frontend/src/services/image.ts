export function toPngDataUrl(base64: string | undefined | null): string | undefined {
  if (!base64) return undefined;
  return `data:image/png;base64,${base64}`;
}

