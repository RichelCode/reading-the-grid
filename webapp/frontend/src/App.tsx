import type { ReactNode } from "react";
import Hero from "./components/Hero";
import Demo from "./components/Demo";
import Reveal from "./components/Reveal";
import Results from "./components/Results";

const GITHUB_URL = "https://github.com/RichelCode/reading-the-grid";
const HUGGINGFACE_URL = "https://huggingface.co/spaces/RichelCode/reading-the-grid";

type Step = { title: string; body: ReactNode };

const STEPS: Step[] = [
  {
    title: "Transfer learning",
    body: (
      <>
        The model starts as a ResNet18 pretrained on over a million everyday photographs. It
        already knows how to see edges, textures, and shapes, so it never has to learn
        vision from scratch on a small solar dataset.
      </>
    ),
  },
  {
    title: "Fine-tuning",
    body: (
      <>
        We unfreeze the last block and specialize it to electroluminescence images at a low
        learning rate. This adaptation lifted fault recall from{" "}
        <span className="mono accent">0.72</span> to{" "}
        <span className="mono accent">0.85</span> without destroying the pretrained
        features.
      </>
    ),
  },
  {
    title: "Grad-CAM",
    body: (
      <>
        To keep the prediction inspectable, Grad-CAM highlights the regions the model
        weighted most. It shows where the model looked, which is attention rather than
        fault severity, so read a hot region as a cue to check, not a measure of damage.
      </>
    ),
  },
];

export default function App() {
  return (
    <>
      <Hero />

      <main>
        {/* --- The problem ------------------------------------------------- */}
        <section id="problem" className="section">
          <div className="container">
            <Reveal>
              <div className="media-split">
                <div className="media-text">
                  <div className="section-head">
                    <span className="eyebrow">The problem</span>
                    <h2 className="section-title">Faults that hide from the eye</h2>
                  </div>
                  <p className="section-lead">
                    A single solar farm can hold millions of cells. Micro-cracks, broken
                    interconnects, and dead regions often leave no visible mark on the
                    surface. But under electroluminescence imaging, where a current makes
                    working silicon glow, those faults show up as dark areas.
                  </p>
                  <p className="section-lead">
                    Catching them early, before a weak cell drags down a whole string or
                    fails out in the field, is the difference between a quick swap and lost
                    generation. The hard part is that there are far too many cells to
                    inspect by hand.
                  </p>
                </div>
                <figure className="media-figure">
                  <img
                    src="/images/panel-array.jpg"
                    width={1500}
                    height={1015}
                    alt="A large rooftop solar installation stretching to the horizon at sunset"
                    loading="lazy"
                    decoding="async"
                  />
                  <figcaption className="photo-credit">Nuno Marques / Unsplash</figcaption>
                </figure>
              </div>
            </Reveal>
          </div>
        </section>

        {/* --- Try it: the live demo --------------------------------------- */}
        <section id="inspect" className="section">
          <div className="container">
            <Reveal>
              <div className="section-head">
                <span className="eyebrow">Live model</span>
                <h2 className="section-title">Inspect a cell</h2>
                <p className="section-lead">
                  Upload an EL cell image or pick an example. The model returns a prediction
                  and a Grad-CAM overlay you can wipe against the original.
                </p>
              </div>
            </Reveal>

            <Reveal>
              <div className="el-contrast">
                <figure className="contrast-card">
                  <img
                    src="/images/panel-closeup.jpg"
                    width={900}
                    height={1350}
                    alt="A close-up daylight photograph of solar-cell modules"
                    loading="lazy"
                    decoding="async"
                  />
                  <figcaption>
                    <span className="tag">Daylight photo</span>
                    The panel, as a camera sees it
                  </figcaption>
                </figure>
                <figure className="contrast-card">
                  <img
                    src="/examples/faulty_1.png"
                    width={300}
                    height={300}
                    alt="An electroluminescence scan of a single solar cell with dark fault regions"
                    loading="lazy"
                    decoding="async"
                  />
                  <figcaption>
                    <span className="tag el">EL scan</span>
                    What the model reads
                  </figcaption>
                </figure>
              </div>
              <p className="el-note">
                Electroluminescence imaging runs a current through the cell and captures the
                infrared glow, so breaks that are invisible in daylight appear as dark lines.
                The model reads these EL scans, not ordinary photos of a panel. The demo
                below takes EL cell images only.
              </p>
            </Reveal>

            <Demo />
          </div>
        </section>

        {/* --- How it works ------------------------------------------------ */}
        <section id="how" className="section">
          <div className="container">
            <Reveal>
              <div className="section-head">
                <span className="eyebrow">How it works</span>
                <h2 className="section-title">Three steps, in plain terms</h2>
              </div>
            </Reveal>
            <ol className="steps">
              {STEPS.map((step, i) => (
                <Reveal key={step.title} delay={i * 80}>
                  <li className="step">
                    <span className="step-num mono">{i + 1}</span>
                    <div>
                      <h3 className="step-title">{step.title}</h3>
                      <p className="step-body">{step.body}</p>
                    </div>
                  </li>
                </Reveal>
              ))}
            </ol>
          </div>
        </section>

        {/* --- Results ----------------------------------------------------- */}
        <section id="results" className="section">
          <div className="container">
            <Reveal>
              <div className="section-head">
                <span className="eyebrow">Results</span>
                <h2 className="section-title">Measured on held-out cells</h2>
                <p className="section-lead">
                  Metrics on a stratified test split the model never trained on.
                </p>
              </div>
            </Reveal>
            <Results />
          </div>
        </section>

        {/* --- Footer ------------------------------------------------------ */}
        <footer className="site-footer">
          <div className="container">
            <div className="footer-links">
              <a href={GITHUB_URL} target="_blank" rel="noreferrer">
                GitHub repository
              </a>
              <a href={HUGGINGFACE_URL} target="_blank" rel="noreferrer">
                Model on Hugging Face
              </a>
            </div>
            <div className="footer-meta">
              <p>
                Data: the ELPV dataset (Buerhop-Lutz et al.; Deitsch et al.), used for
                non-commercial research purposes.
              </p>
              <p>Built by Richel Attafuah. Transfer-learned ResNet18 with Grad-CAM.</p>
              <p>Photos: Chelsea, Nuno Marques, and Sören H. on Unsplash.</p>
            </div>
          </div>
        </footer>
      </main>
    </>
  );
}
