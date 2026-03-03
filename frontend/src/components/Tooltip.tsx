interface TooltipProps {
  text: string;
  children: React.ReactNode;
}

export function Tooltip({ text, children }: TooltipProps) {
  return (
    <span className="group relative inline-flex cursor-help">
      {children}
      <span className="invisible group-hover:visible absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 rounded-lg bg-zinc-800 px-3 py-2 text-xs text-zinc-300 shadow-lg z-50 leading-relaxed border border-zinc-700">
        {text}
      </span>
    </span>
  );
}
