// Single source of truth for the reduced-motion check, used to gate the scan sweep, the
// hero animation, the scroll reveals, and the count-up stats.
export function prefersReducedMotion(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}
