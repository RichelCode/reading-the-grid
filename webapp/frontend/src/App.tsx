import { useCallback, useMemo, useRef, useState } from "react";
import CompareSlider from "./components/CompareSlider";
import ThresholdControl from "./components/ThresholdControl";

// Shape of the /api/predict response. The backend returns the raw faulty-class
// probability (not a hard label) plus both images as base64 PNG data URLs, so the
// decision threshold is applied here on the client without re-running the model.
type Prediction = {
  faulty_probability: number;
  predicted_label: "healthy" | "faulty";
  original_image: string;
  gradcam_overlay: string;
};

type Phase = "idle" | "loading" | "done" | "error";

const ACCEPTED_TYPES = ["image/png", "image/jpeg"];
const LOW_CONFIDENCE = 0.65;

// Bundled sample cells (copied into public/examples). Served by the same origin, so a
// visitor with no image of their own can try the full flow in one tap.
const EXAMPLES = [
  { file: "healthy_1.png", label: "Healthy 1", kind: "healthy" as const },
  { file: "healthy_2.png", label: "Healthy 2", kind: "healthy" as const },
  { file: "healthy_3.png", label: "Healthy 3", kind: "healthy" as const },
  { file: "faulty_1.png", label: "Faulty 1", kind: "faulty" as const },
  { file: "faulty_2.png", label: "Faulty 2", kind: "faulty" as const },
  { file: "faulty_3.png", label: "Faulty 3", kind: "faulty" as const },
];

function prefersReducedMotion(): boolean {
  return (
    typeof window !== "undefined" &&
    window.matchMedia?.("(prefers-reduced-motion: reduce)").matches
  );
}

export default function App() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [error, setError] = useState<string>("");
  const [threshold, setThreshold] = useState<number>(0.5);
  const [fileName, setFileName] = useState<string>("");
  const [dragging, setDragging] = useState<boolean>(false);
  const [scanKey, setScanKey] = useState<number>(0);
  const [showSweep, setShowSweep] = useState<boolean>(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const classify = useCallback(async (file: File) => {
    if (!ACCEPTED_TYPES.includes(file.type)) {
      setPhase("error");
      setError("Unsupported file type. Upload a PNG or JPG image.");
      return;
    }

    setPhase("loading");
    setError("");
    setPrediction(null);
    setFileName(file.name);

    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch("/api/predict", { method: "POST", body });
      if (!res.ok) {
        const detail = await res
          .json()
          .then((d) => d.detail as string)
          .catch(() => "");
        throw new Error(detail || `Request failed (${res.status}).`);
      }
      const data: Prediction = await res.json();
      setPrediction(data);
      setThreshold(0.5);
      setPhase("done");
      // Trigger the single scan-line sweep on the real response arriving.
      if (!prefersReducedMotion()) {
        setScanKey((k) => k + 1);
        setShowSweep(true);
      }
    } catch (err) {
      setPhase("error");
      setError(err instanceof Error ? err.message : "Could not reach the backend.");
    }
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      setDragging(false);
      const file = event.dataTransfer.files?.[0];
      if (file) classify(file);
    },
    [classify],
  );

  const runExample = useCallback(
    async (fileName: string, label: string) => {
      try {
        const res = await fetch(`/examples/${fileName}`);
        const blob = await res.blob();
        classify(new File([blob], `${label}.png`, { type: blob.type || "image/png" }));
      } catch {
        setPhase("error");
        setError("Could not load the example image.");
      }
    },
    [classify],
  );

  // Label and confidence recomputed from the raw probability whenever the slider moves.
  const decision = useMemo(() => {
    if (!prediction) return null;
    const p = prediction.faulty_probability;
    const isFaulty = p >= threshold;
    return {
      isFaulty,
      kind: isFaulty ? "faulty" : "healthy",
      label: isFaulty ? "Faulty" : "Healthy",
      // Confidence in the reported label, not always the faulty probability.
      confidence: isFaulty ? p : 1 - p,
    };
  }, [prediction, threshold]);

  return (
    <div className="page">
      <header className="app-header">
        <p className="eyebrow">Electroluminescence inspection</p>
        <h1 className="app-title">Reading the Grid</h1>
        <p className="app-subtitle">
          Detect faults in solar cells from electroluminescence images, with a Grad-CAM
          view of the regions that drove each prediction.
        </p>
      </header>

      <div className="stack">
        {/* --- Input: upload + example gallery ------------------------------- */}
        <section>
          <label
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            className={`dropzone${dragging ? " is-dragging" : ""}`}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/png,image/jpeg"
              className="sr-only"
              style={{ display: "none" }}
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) classify(file);
                e.target.value = "";
              }}
            />
            <svg className="dropzone-icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path
                d="M12 16V4m0 0L8 8m4-4l4 4M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            <span className="dropzone-primary">Drop an EL cell image, or click to browse</span>
            <span className="dropzone-hint mono">PNG or JPG</span>
            {fileName && <span className="dropzone-file">Loaded: {fileName}</span>}
          </label>

          <div className="gallery">
            <span className="section-label">Or try an example</span>
            <div className="gallery-row">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex.file}
                  type="button"
                  className="thumb"
                  onClick={() => runExample(ex.file, ex.label)}
                  title={`Run ${ex.label}`}
                >
                  <img src={`/examples/${ex.file}`} alt={ex.label} />
                  <span className={`thumb-tag ${ex.kind}`}>{ex.kind}</span>
                </button>
              ))}
            </div>
          </div>
        </section>

        {phase === "loading" && (
          <div className="status-line">
            <span className="spinner" />
            <span className="mono">Running inference and Grad-CAM…</span>
          </div>
        )}

        {phase === "error" && <div className="error-box">{error}</div>}

        {/* --- Result: verdict, signature compare, threshold ----------------- */}
        {phase === "done" && prediction && decision && (
          <>
            <div className={`verdict ${decision.kind}`}>
              <div className="verdict-main">
                <span className="verdict-dot" />
                <span className="verdict-label">{decision.label}</span>
                <span className="verdict-confidence">
                  {(decision.confidence * 100).toFixed(1)}% confidence
                </span>
                {decision.confidence < LOW_CONFIDENCE && (
                  <span className="verdict-note">
                    Borderline — low confidence. Treat this call as uncertain.
                  </span>
                )}
              </div>
              <div className="verdict-readout">
                <div>faulty probability</div>
                <div className="value">
                  {(prediction.faulty_probability * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            <section className="card">
              <span className="section-label">Signature — drag to compare</span>
              <div className="signature" style={{ marginTop: 16 }}>
                <CompareSlider
                  originalSrc={prediction.original_image}
                  overlaySrc={prediction.gradcam_overlay}
                />
                {showSweep && (
                  <div
                    key={scanKey}
                    className="scan-sweep"
                    onAnimationEnd={() => setShowSweep(false)}
                  />
                )}
              </div>
            </section>

            <ThresholdControl value={threshold} onChange={setThreshold} />
          </>
        )}

        {/* --- Honest limitations -------------------------------------------- */}
        <details className="limitations">
          <summary>
            How this works &amp; limitations
            <svg className="chevron" viewBox="0 0 24 24" width="16" height="16" fill="none" aria-hidden="true">
              <path d="M6 9l6 6 6-6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </summary>
          <div className="limitations-body">
            <p>
              The classifier is a <strong>transfer-learned ResNet18</strong>, fine-tuned on
              the ELPV electroluminescence dataset to separate healthy from faulty cells.
            </p>
            <ul>
              <li>
                The <strong>Grad-CAM heatmap is normalized per image</strong>, so it shows
                relative attention, not fault severity — and it never goes fully quiet, even
                on a healthy cell. Warmer regions are where the model looked most, not a
                measure of how bad a fault is.
              </li>
              <li>
                <strong>Faulty localization is partial.</strong> The highlighted region
                overlaps real defects only some of the time; read it as a hint, not a mask.
              </li>
              <li>
                <strong>Dead cells tend to draw edge activation</strong> rather than a
                centered blob, which can look like the model is looking "around" the fault.
              </li>
              <li>
                This is an <strong>inspection-assist aid, not a calibrated localizer.</strong>{" "}
                Use it to prioritize a human review, not to certify a cell.
              </li>
            </ul>
            <div className="attribution">
              Data: the ELPV dataset (Buerhop-Lutz et al.; Deitsch et al.), used here for
              non-commercial research purposes.
            </div>
          </div>
        </details>
      </div>
    </div>
  );
}
