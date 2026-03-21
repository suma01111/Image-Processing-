export function BeforeAfter(props: {
  beforeLabel: string;
  afterLabel: string;
  beforeSrc?: string;
  afterSrc?: string;
}) {
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
      <div className="rounded-xl border border-slate-200 bg-gradient-to-br from-white via-white to-indigo-50/30 p-3 shadow-sm">
        <div className="mb-2 text-sm font-bold tracking-wide text-slate-800">{props.beforeLabel}</div>
        {props.beforeSrc ? (
          <img
            className="aspect-square w-full rounded-lg object-contain bg-white p-1"
            src={props.beforeSrc}
            alt={props.beforeLabel}
          />
        ) : (
          <div className="aspect-square w-full rounded-lg bg-slate-50" />
        )}
      </div>
      <div className="rounded-xl border border-slate-200 bg-gradient-to-br from-white via-white to-indigo-50/30 p-3 shadow-sm">
        <div className="mb-2 text-sm font-bold tracking-wide text-slate-800">{props.afterLabel}</div>
        {props.afterSrc ? (
          <img
            className="aspect-square w-full rounded-lg object-contain bg-white p-1"
            src={props.afterSrc}
            alt={props.afterLabel}
          />
        ) : (
          <div className="aspect-square w-full rounded-lg bg-slate-50" />
        )}
      </div>
    </div>
  );
}

