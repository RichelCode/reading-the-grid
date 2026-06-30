import { useEffect, useState } from "react";

type Status = "checking" | "online" | "offline";

// Scaffold page: reports backend reachability via /api/health and shows the placeholder
// message from /api/hello. Styling is intentionally minimal; the real design comes later.
export default function App() {
  const [status, setStatus] = useState<Status>("checking");
  const [message, setMessage] = useState<string>("");

  useEffect(() => {
    fetch("/api/health")
      .then((res) => (res.ok ? res.json() : Promise.reject()))
      .then((data) => setStatus(data.status === "ok" ? "online" : "offline"))
      .catch(() => setStatus("offline"));

    fetch("/api/hello")
      .then((res) => (res.ok ? res.json() : Promise.reject()))
      .then((data) => setMessage(data.message))
      .catch(() => setMessage("Could not reach the backend."));
  }, []);

  const statusColor =
    status === "online"
      ? "bg-green-500"
      : status === "offline"
        ? "bg-red-500"
        : "bg-gray-400";

  const statusLabel =
    status === "online"
      ? "Backend online"
      : status === "offline"
        ? "Backend offline"
        : "Checking backend…";

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 text-gray-900">
      <div className="w-full max-w-md rounded-lg border border-gray-200 bg-white p-8 shadow-sm">
        <h1 className="text-2xl font-semibold">Reading the Grid</h1>
        <p className="mt-1 text-sm text-gray-500">Web app scaffold</p>

        <div className="mt-6 flex items-center gap-2">
          <span className={`inline-block h-2.5 w-2.5 rounded-full ${statusColor}`} />
          <span className="text-sm text-gray-700">{statusLabel}</span>
        </div>

        <div className="mt-4 rounded-md bg-gray-100 p-4 font-mono text-sm">
          {message || "…"}
        </div>
      </div>
    </div>
  );
}
