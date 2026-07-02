import type { ReactNode } from "react";
import { useInView } from "../hooks/useInView";

// Fade-and-rise on scroll entry. Under prefers-reduced-motion the CSS collapses this to an
// instant, static appearance (see .reveal in index.css).
type Props = {
  children: ReactNode;
  className?: string;
  delay?: number;
};

export default function Reveal({ children, className = "", delay = 0 }: Props) {
  const [ref, inView] = useInView<HTMLDivElement>();
  return (
    <div
      ref={ref}
      className={`reveal ${inView ? "is-visible" : ""} ${className}`.trim()}
      style={delay ? { transitionDelay: `${delay}ms` } : undefined}
    >
      {children}
    </div>
  );
}
