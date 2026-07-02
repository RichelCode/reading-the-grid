import { useEffect, useState } from "react";
import { prefersReducedMotion } from "../lib/motion";

// Count-up readout that animates from zero to the target once `active` becomes true (its
// section has scrolled into view). Reduced motion jumps straight to the final value.
type Props = {
  value: number;
  label: string;
  active: boolean;
  decimals?: number;
  suffix?: string;
};

export default function StatCounter({
  value,
  label,
  active,
  decimals = 0,
  suffix = "",
}: Props) {
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    if (!active) return;
    if (prefersReducedMotion()) {
      setDisplay(value);
      return;
    }
    let raf = 0;
    let startTime = 0;
    const duration = 1100;
    const step = (now: number) => {
      if (!startTime) startTime = now;
      const progress = Math.min(1, (now - startTime) / duration);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(value * eased);
      if (progress < 1) raf = requestAnimationFrame(step);
      else setDisplay(value);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [active, value]);

  return (
    <div className="stat">
      <div className="stat-value mono">
        {display.toFixed(decimals)}
        {suffix}
      </div>
      <div className="stat-label">{label}</div>
    </div>
  );
}
