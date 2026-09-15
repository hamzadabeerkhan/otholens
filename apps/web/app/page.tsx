"use client";

import { ChangeEvent, FormEvent, useEffect, useState } from "react";

type Result = {
  study_id: string;
  research_only: true;
  input: { sha256: string; format: string; original_width: number; original_height: number };
  pipeline_version: string;
  prediction: { model_version: string; selected_kl_grade: number; probabilities: Record<string, number>; uncertainty: number };
  warnings: { code: string; message: string }[];
};

const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);
  useEffect(() => {
    fetch(`${apiBase}/health`)
      .then((response) => setApiOnline(response.ok))
      .catch(() => setApiOnline(false));
  }, []);

  function chooseFile(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0] ?? null;
    if (preview) URL.revokeObjectURL(preview);
    setFile(selected); setPreview(selected ? URL.createObjectURL(selected) : null); setResult(null); setError(null);
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!file) return;
    setLoading(true); setResult(null); setError(null);
    const body = new FormData(); body.append("file", file);
    try {
      const response = await fetch(`${apiBase}/v1/studies/analyse`, { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? "Analysis failed.");
      setResult(payload);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Analysis failed.");
    } finally { setLoading(false); }
  }

  return (
    <main>
      <header>
        <div><span className="eyebrow">ORTHOLENS</span><h1>Knee radiograph research workspace</h1></div>
        <span className={`status ${apiOnline ? "online" : ""}`}>{apiOnline === null ? "Checking API" : apiOnline ? "API online" : "API offline"}</span>
      </header>
      <section className="warning"><strong>Research prototype</strong><span>Not for clinical diagnosis, triage, treatment planning, or patient use.</span></section>
      <div className="grid">
        <section className="panel">
          <h2>Input study</h2><p className="muted">Use only synthetic or legally cleared, de-identified PNG or JPEG images.</p>
          <form onSubmit={submit}>
            <label className="upload">
              {preview ? <img src={preview} alt="Selected radiograph preview" /> : <span>Select a knee radiograph</span>}
              <input type="file" accept="image/png,image/jpeg" onChange={chooseFile} />
            </label>
            <div className="file-row"><span>{file?.name ?? "No file selected"}</span><button disabled={!file || loading}>{loading ? "Processing…" : "Run placeholder analysis"}</button></div>
          </form>
        </section>
        <section className="panel results">
          <h2>Research result</h2>
          {!result && !error && <p className="empty">A structured result will appear here after validation.</p>}
          {error && <p className="error">{error}</p>}
          {result && <>
            <div className="grade"><span>Placeholder KL grade</span><strong>{result.prediction.selected_kl_grade}</strong><small>Uncertainty {(result.prediction.uncertainty * 100).toFixed(0)}%</small></div>
            <div className="bars">{Object.entries(result.prediction.probabilities).map(([grade, probability]) => <div className="bar" key={grade}><span>KL {grade}</span><div><i style={{ width: `${probability * 100}%` }} /></div><b>{(probability * 100).toFixed(0)}%</b></div>)}</div>
            <dl><div><dt>Study</dt><dd>{result.study_id}</dd></div><div><dt>Model</dt><dd>{result.prediction.model_version}</dd></div><div><dt>Pipeline</dt><dd>{result.pipeline_version}</dd></div></dl>
            {result.warnings.map(item => <p className="notice" key={item.code}>{item.message}</p>)}
          </>}
        </section>
      </div>
    </main>
  );
}
