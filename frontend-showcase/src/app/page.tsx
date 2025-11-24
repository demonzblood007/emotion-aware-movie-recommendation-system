import { RecommendationShowcase } from "@/components/showcase/recommender-showcase";
import { StatusIndicator } from "@/components/status-indicator";

export default function Home() {
  return (
    <main className="mx-auto flex min-h-screen w-full max-w-6xl flex-col gap-10 px-4 py-12 sm:px-8 lg:py-16">
      <header className="rounded-3xl border border-border bg-card/80 p-8 shadow-lg backdrop-blur">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
          <div className="space-y-4">
            <h1 className="text-4xl font-semibold leading-tight text-foreground sm:text-5xl">
              Emotion Aware Movie Recommender System
            </h1>
            <p className="max-w-2xl text-base text-muted-foreground">
              Discover personalized movie recommendations based on your current mood. 
              Our system analyzes your emotions, maps them to genres, and ranks movies 
              tailored to your preferences.
            </p>
            <div className="flex flex-wrap items-center gap-4 text-sm text-muted-foreground">
              <span className="font-medium">Developed by:</span>
              <span className="rounded-full bg-primary/10 px-3 py-1">
                Bhavik Mehta
              </span>
              <span className="rounded-full bg-primary/10 px-3 py-1">
                Bhavya Ameta
              </span>
              <span className="rounded-full bg-primary/10 px-3 py-1">
                Anushka Singh
              </span>
            </div>
          </div>
          <StatusIndicator />
        </div>
      </header>

      <RecommendationShowcase />

      <section className="grid gap-6 rounded-3xl border border-border bg-card/90 p-8 shadow-lg backdrop-blur lg:grid-cols-2">
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold text-foreground">How the pipeline flows</h2>
          <ol className="space-y-3 text-sm text-muted-foreground">
            <li>
              <span className="font-semibold text-foreground">1.</span> Mood text
              is embedded via the local BERT emotion head in the FastAPI service.
            </li>
            <li>
              <span className="font-semibold text-foreground">2.</span> The
              detected emotions are mapped to normalized genre buckets.
            </li>
            <li>
              <span className="font-semibold text-foreground">3.</span> Genres +
              user context feed the NCF ranker to score the full MovieLens slate.
            </li>
            <li>
              <span className="font-semibold text-foreground">4.</span> JSON
              response mirrors the CLI output for deterministic diffs.
            </li>
          </ol>
        </div>
        <div className="space-y-4">
          <h3 className="text-xl font-semibold text-foreground">Sample response</h3>
          <div className="rounded-2xl bg-muted p-4 text-xs text-muted-foreground shadow-inner">
            <pre>
              {`POST ${process.env.NEXT_PUBLIC_RECOMMENDER_API_URL ?? "https://api.example.com"}/recommendations
{
  "mood_text": "Need a cozy comfort watch after a long week",
  "strategy": "match",
  "top_n": 10
}`}
            </pre>
            <p className="mt-3 text-[11px] text-muted-foreground/70">
              Use the console or Raw JSON tabs above to inspect real responses.
            </p>
          </div>
        </div>
      </section>
    </main>
  );
}
