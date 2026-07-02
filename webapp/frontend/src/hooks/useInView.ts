import { useEffect, useRef, useState, type RefObject } from "react";

// Fires once when the element first scrolls into view, then disconnects. Backs the
// scroll-reveal wrappers and gates the count-up stats so they start only when visible.
// Falls back to visible immediately where IntersectionObserver is unavailable.
export function useInView<T extends Element>(
  rootMargin = "0px 0px -10% 0px",
): [RefObject<T | null>, boolean] {
  const ref = useRef<T>(null);
  const [inView, setInView] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (typeof IntersectionObserver === "undefined") {
      setInView(true);
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          setInView(true);
          observer.disconnect();
        }
      },
      { threshold: 0.15, rootMargin },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [rootMargin]);

  return [ref, inView];
}
