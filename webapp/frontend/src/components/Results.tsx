import { useInView } from "../hooks/useInView";
import StatCounter from "./StatCounter";

// Real held-out test metrics. Counts up when the block enters view.
const STATS = [
  { value: 84.8, decimals: 1, suffix: "%", label: "Accuracy" },
  { value: 85.4, decimals: 1, suffix: "%", label: "Faulty recall" },
  { value: 71.4, decimals: 1, suffix: "%", label: "Faulty precision" },
  { value: 0.778, decimals: 3, suffix: "", label: "F1 score" },
];

export default function Results() {
  const [ref, inView] = useInView<HTMLDivElement>();

  return (
    <div ref={ref}>
      <div className="stats-grid">
        {STATS.map((s) => (
          <StatCounter
            key={s.label}
            value={s.value}
            decimals={s.decimals}
            suffix={s.suffix}
            label={s.label}
            active={inView}
          />
        ))}
      </div>

      <div className="confusion">
        <span className="section-label">Confusion matrix, held-out test set</span>
        <div className="cm-grid">
          <div className="cm-corner mono">n = 394</div>
          <div className="cm-colhead mono">pred healthy</div>
          <div className="cm-colhead mono">pred faulty</div>

          <div className="cm-rowhead mono">true healthy</div>
          <div className="cm-cell hit-healthy mono">229</div>
          <div className="cm-cell mono">42</div>

          <div className="cm-rowhead mono">true faulty</div>
          <div className="cm-cell mono">18</div>
          <div className="cm-cell hit-faulty mono">105</div>
        </div>
      </div>

      <p className="results-note">
        Recall on the faulty class was prioritized deliberately: a missed fault that fails in
        the field costs far more than a false alarm a human can wave off in review. That
        choice is why recall (85.4%) runs ahead of precision (71.4%). The model is tuned to
        catch faults, then let a person confirm them.
      </p>
    </div>
  );
}
