// Bright "daylight lab" hero. Blue headline, one-line thesis, a gold primary CTA that
// smooth-scrolls to the demo, and a real photograph of a solar array alongside the
// headline, grounding the page in the physical subject rather than a synthetic motif.
export default function Hero() {
  return (
    <section className="hero">
      <div className="hero-inner">
        <div className="hero-content">
          <p className="eyebrow">Electroluminescence fault inspection</p>
          <h1 className="hero-title">Reading the Grid</h1>
          <p className="hero-thesis">
            Solar cells hide their faults. This model finds them, and shows its work.
          </p>
          <div className="hero-actions">
            <a href="#inspect" className="btn-primary">
              Inspect a cell
              <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path
                  d="M12 5v14m0 0l6-6m-6 6l-6-6"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </a>
            <a href="#results" className="link-secondary">
              See the results
            </a>
          </div>
        </div>

        <figure className="hero-photo">
          <img
            src="/images/hero-farm.jpg"
            width={1500}
            height={1125}
            alt="A ground-mounted solar array under a clear blue sky"
            loading="eager"
            decoding="async"
          />
          <figcaption className="photo-credit">Chelsea / Unsplash</figcaption>
        </figure>
      </div>
    </section>
  );
}
