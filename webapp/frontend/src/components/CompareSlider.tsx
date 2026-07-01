import { useCallback, useEffect, useRef, useState } from "react";

// Drag-to-compare wipe. The Grad-CAM overlay sits underneath as the base layer; the
// original image is clipped from the right so that the left `position`% shows the
// original and the remainder reveals the heatmap. A draggable handle sets the split.
// Pointer events cover mouse and touch from one code path; arrow keys move it too.
type Props = {
  originalSrc: string;
  overlaySrc: string;
};

export default function CompareSlider({ originalSrc, overlaySrc }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);
  const [position, setPosition] = useState(50);

  // Reset to center whenever a new pair of images arrives.
  useEffect(() => {
    setPosition(50);
  }, [originalSrc, overlaySrc]);

  const setFromClientX = useCallback((clientX: number) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setPosition(Math.max(0, Math.min(100, pct)));
  }, []);

  const onPointerDown = useCallback(
    (event: React.PointerEvent) => {
      draggingRef.current = true;
      event.currentTarget.setPointerCapture?.(event.pointerId);
      setFromClientX(event.clientX);
    },
    [setFromClientX],
  );

  const onPointerMove = useCallback(
    (event: React.PointerEvent) => {
      if (draggingRef.current) setFromClientX(event.clientX);
    },
    [setFromClientX],
  );

  const endDrag = useCallback(() => {
    draggingRef.current = false;
  }, []);

  const onKeyDown = useCallback((event: React.KeyboardEvent) => {
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      setPosition((p) => Math.max(0, p - 3));
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      setPosition((p) => Math.min(100, p + 3));
    }
  }, []);

  return (
    <div
      ref={containerRef}
      className="compare"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
    >
      {/* Base layer: Grad-CAM overlay, fully visible on the right of the split. */}
      <img src={overlaySrc} alt="Grad-CAM heatmap overlay" draggable={false} />
      {/* Top layer: original cell, clipped to the left `position`%. */}
      <img
        src={originalSrc}
        alt="Uploaded electroluminescence cell"
        className="compare-top"
        draggable={false}
        style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
      />

      <span className="compare-caption left mono">Original</span>
      <span className="compare-caption right mono">Grad-CAM</span>

      <div className="compare-divider" style={{ left: `${position}%` }} />
      <div
        className="compare-handle"
        style={{ left: `${position}%` }}
        role="slider"
        tabIndex={0}
        aria-label="Reveal amount between the original image and the Grad-CAM overlay"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(position)}
        onKeyDown={onKeyDown}
      >
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path
            d="M11 7l-4 5 4 5M13 7l4 5-4 5"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </div>
    </div>
  );
}
