// Decision-threshold control. Purely client-side: the backend returns the faulty
// probability once, and this slider re-derives the label and confidence upstream with no
// re-fetch. Styled as an instrument readout with the value in mono.
type Props = {
  value: number;
  onChange: (value: number) => void;
};

export default function ThresholdControl({ value, onChange }: Props) {
  return (
    <div className="card">
      <div className="control-head">
        <span className="section-label">Decision threshold</span>
        <span className="control-value">{value.toFixed(2)}</span>
      </div>

      <input
        className="range"
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        aria-label="Decision threshold"
      />
      <div className="range-scale mono">
        <span>0.00 — healthy</span>
        <span>faulty — 1.00</span>
      </div>

      <p className="helper-text">
        A cell is called faulty when its faulty probability is at or above this threshold.
        Lower it to catch more faults, at the cost of more false positives; raise it to flag
        only high-confidence faults.
      </p>
    </div>
  );
}
