import { useCallback, useMemo, useRef, useState } from "react";

// Shape of the /api/predict response. The backend returns the raw faulty-class
// probability (not a hard label) plus both images as base64 PNG data URLs, so the
// decision threshold can be applied here on the client without re-running the model.
type Prediction = {
  faulty_probability: number;
  predicted_label: "healthy" | "faulty";
  original_image: string;
  gradcam_overlay: string;
};

type Phase = "idle" | "loading" | "done" | "error";

const ACCEPTED_TYPES = ["image/png", "image/jpeg"];

export default function App() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [error, setError] = useState<string>("");
  const [threshold, setThreshold] = useState<number>(0.5);
  const [fileName, setFileName] = useState<string>("");
  const [dragging, setDragging] = useState<boolean>(false);
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

  // Label and confidence recomputed from the raw probability whenever the slider moves.
  const decision = useMemo(() => {
    if (!prediction) return null;
    const p = prediction.faulty_probability;
    const isFaulty = p >= threshold;
    return {
      isFaulty,
      label: isFaulty ? "Faulty" : "Healthy",
      // Confidence in the reported label, not always the faulty probability.
      confidence: isFaulty ? p : 1 - p,
    };
  }, [prediction, threshold]);

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <div className="mx-auto max-w-4xl px-6 py-10">
        <header>
          <h1 className="text-2xl font-semibold">Reading the Grid</h1>
          <p className="mt-1 text-sm text-gray-500">
            Solar-cell fault detection from electroluminescence images, with a Grad-CAM
            heatmap of the regions driving the prediction.
          </p>
        </header>

        <label
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={`mt-8 flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-10 text-center transition-colors ${
            dragging
              ? "border-blue-400 bg-blue-50"
              : "border-gray-300 bg-white hover:border-gray-400"
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/png,image/jpeg"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) classify(file);
              e.target.value = "";
            }}
          />
          <p className="text-sm font-medium text-gray-700">
            Drop an EL cell image here, or click to browse
          </p>
          <p className="mt-1 text-xs text-gray-400">PNG or JPG</p>
          {fileName && (
            <p className="mt-3 text-xs text-gray-500">Selected: {fileName}</p>
          )}
        </label>

        {phase === "loading" && (
          <div className="mt-6 flex items-center gap-3 text-sm text-gray-600">
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-gray-300 border-t-gray-600" />
            Running inference and Grad-CAM…
          </div>
        )}

        {phase === "error" && (
          <div className="mt-6 rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            {error}
          </div>
        )}

        {phase === "done" && prediction && decision && (
          <div className="mt-8">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-center gap-3">
                <span
                  className={`inline-flex h-3 w-3 rounded-full ${
                    decision.isFaulty ? "bg-red-500" : "bg-green-500"
                  }`}
                />
                <span className="text-lg font-semibold">{decision.label}</span>
                <span className="text-sm text-gray-500">
                  {(decision.confidence * 100).toFixed(1)}% confidence
                </span>
              </div>
              <div className="text-sm text-gray-500">
                Faulty probability:{" "}
                <span className="font-mono text-gray-800">
                  {(prediction.faulty_probability * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            <div className="mt-5 rounded-md border border-gray-200 bg-white p-4">
              <div className="flex items-center justify-between text-sm text-gray-600">
                <label htmlFor="threshold">Decision threshold</label>
                <span className="font-mono text-gray-800">
                  {threshold.toFixed(2)}
                </span>
              </div>
              <input
                id="threshold"
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                className="mt-2 w-full accent-blue-600"
              />
              <p className="mt-1 text-xs text-gray-400">
                A cell is labeled faulty when its faulty probability is at or above the
                threshold. Lower the threshold to catch more faults at the cost of more
                false positives.
              </p>
            </div>

            <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2">
              <figure className="overflow-hidden rounded-md border border-gray-200 bg-white">
                <img
                  src={prediction.original_image}
                  alt="Uploaded electroluminescence cell"
                  className="w-full"
                />
                <figcaption className="border-t border-gray-100 px-3 py-2 text-xs text-gray-500">
                  Original
                </figcaption>
              </figure>
              <figure className="overflow-hidden rounded-md border border-gray-200 bg-white">
                <img
                  src={prediction.gradcam_overlay}
                  alt="Grad-CAM heatmap overlay"
                  className="w-full"
                />
                <figcaption className="border-t border-gray-100 px-3 py-2 text-xs text-gray-500">
                  Grad-CAM overlay — warmer regions drove the prediction
                </figcaption>
              </figure>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
