"use client";

import { useEffect, useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { toast } from "sonner";
import { requestRecommendations, requestRecommendationsGranular } from "@/lib/api";
import {
  EmotionScore,
  RecommendationPayload,
  RecommendationResponse,
  Strategy,
} from "@/lib/types";
import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ThemeToggle } from "@/components/theme-toggle";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  Radar,
} from "recharts";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Activity,
  BadgeCheck,
  CircleUserRound,
  Sparkles,
  Terminal,
  LayoutGrid,
  List,
  Filter,
  ArrowUpDown,
  Code,
} from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

const DEFAULT_TOP_N = 10;

const strategies: {
  value: Strategy;
  label: string;
  helper: string;
  example: string;
}[] = [
  {
    value: "match",
    label: "Match",
    helper: "Lean into the detected mood",
    example: "Feeling sad? Here are more dramas to lean into it",
  },
  {
    value: "neutral",
    label: "Neutral",
    helper: "Blend mood cues with baseline taste",
    example: "A balanced mix that respects your mood",
  },
  {
    value: "shift",
    label: "Shift",
    helper: "Counterbalance with uplifting genres",
    example: "Feeling down? Let's lift you up with comedies",
  },
];

type FormState = {
  moodText: string;
  userId: string;
  strategy: Strategy;
};

const initialForm: FormState = {
  moodText: "",
  userId: "",
  strategy: "match",
};

const formatPercent = (value?: number) =>
  typeof value === "number" ? `${(value * 100).toFixed(0)}%` : "—";

export const RecommendationShowcase = () => {
  const [form, setForm] = useState<FormState>(initialForm);
  const [lastResponse, setLastResponse] =
    useState<RecommendationResponse | null>(null);
  const [currentStep, setCurrentStep] = useState<string>("");
  const [showDeveloperMode, setShowDeveloperMode] = useState<boolean>(false);
  const [viewMode, setViewMode] = useState<"list" | "grid">("list");
  const [sortBy, setSortBy] = useState<"rating" | "imdb" | "relevance">("relevance");
  const [filterGenre, setFilterGenre] = useState<string>("all");

  const mutation = useMutation({
    mutationKey: ["recommendations"],
    mutationFn: async (payload: RecommendationPayload) => {
      // Use granular endpoints for better visibility and agentic support
      return requestRecommendationsGranular(payload, (step) => {
        setCurrentStep(step);
      });
    },
    onSuccess: (data) => {
      setLastResponse(data);
      setCurrentStep("");
      toast.success("Recommendations ready");
    },
    onError: (error: Error) => {
      setCurrentStep("");
      toast.error("Unable to fetch recommendations", {
        description: error.message,
      });
    },
  });

  const isLoading = mutation.isPending;

  const topEmotion = useMemo(() => {
    if (!lastResponse?.emotions?.length) return null;
    return [...lastResponse.emotions].sort((a, b) => b.score - a.score)[0];
  }, [lastResponse]);

  // Get unique genres for filtering
  const availableGenres = useMemo(() => {
    if (!lastResponse?.results) return [];
    const genres = new Set<string>();
    lastResponse.results.forEach((movie) => {
      movie.genres.forEach((genre) => genres.add(genre));
    });
    return Array.from(genres).sort();
  }, [lastResponse]);

  // Filtered and sorted recommendations
  const filteredAndSortedResults = useMemo(() => {
    if (!lastResponse?.results) return [];
    
    let results = [...lastResponse.results];
    
    // Filter by genre
    if (filterGenre !== "all") {
      results = results.filter((movie) =>
        movie.genres.includes(filterGenre)
      );
    }
    
    // Sort
    if (sortBy === "rating") {
      results.sort((a, b) => b.predicted_rating - a.predicted_rating);
    } else if (sortBy === "imdb") {
      results.sort((a, b) => {
        const aRating = a.imdb_rating || 0;
        const bRating = b.imdb_rating || 0;
        return bRating - aRating;
      });
    } else {
      // relevance (original order)
      results = results; // Already sorted by rank
    }
    
    return results;
  }, [lastResponse, filterGenre, sortBy]);


  const consoleLines = useMemo(() => {
    if (isLoading && currentStep) {
      return [
        `[status] ${currentStep}`,
        "[hint] using granular endpoints for step-by-step processing",
      ];
    }
    if (!lastResponse) {
      return [
        "[boot] awaiting first inference...",
        "[hint] describe your mood and choose a strategy",
      ];
    }
    const emotionLine = `[emotion] ${lastResponse.emotions
      .map((e) => `${e.emotion}:${(e.score * 100).toFixed(1)}%`)
      .join(" | ")}`;
    const genreLine = `[genres] ${lastResponse.genres
      .map((g) => `${g.genre}:${(g.score * 100).toFixed(0)}%`)
      .join(" → ")}`;
    const userLine = `[user] ${lastResponse.existing_user ? "MovieLens #" : "Cold start idx "}${
      lastResponse.existing_user ? lastResponse.user_id : lastResponse.user_idx
    }`;
    const resultLine = `[ranker] surfaced ${lastResponse.results.length} titles via NCF`;
    return [userLine, emotionLine, genreLine, resultLine];
  }, [lastResponse, isLoading, currentStep]);

  const handleSubmit = () => {
    if (!form.moodText.trim()) {
      toast.info("Describe your mood to get started");
      return;
    }

    const payload: RecommendationPayload = {
      mood_text: form.moodText.trim(),
      strategy: form.strategy,
      top_n: DEFAULT_TOP_N,
      user_id: form.userId ? Number(form.userId) : undefined,
    };
    mutation.mutate(payload);
  };

  const resetForm = () => {
    setForm(initialForm);
    setLastResponse(null);
    mutation.reset();
  };

  return (
    <section className="grid gap-8 lg:grid-cols-[minmax(0,360px)_1fr]">
      <Card className="border border-primary/10 bg-gradient-to-b from-card via-card to-primary/5 dark:from-card dark:via-card/95 dark:to-primary/10 shadow-lg shadow-primary/10 dark:shadow-primary/20">
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Sparkles className="h-5 w-5 text-primary" />
              Mood Input
            </CardTitle>
            <p className="text-sm text-muted-foreground mt-1">
              Mirrors the CLI prompt: describe the current mood, optionally pin a
              MovieLens user, and choose how the recommender should react.
            </p>
          </div>
          <ThemeToggle />
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-2">
            <Label htmlFor="moodText">Mood description</Label>
            <Textarea
              id="moodText"
              placeholder="I just wrapped a long week at work and want something uplifting but not too goofy..."
              value={form.moodText}
              onChange={(event) =>
                setForm((prev) => ({ ...prev, moodText: event.target.value }))
              }
              className="min-h-[130px] resize-none"
            />
          </div>

          <div className="space-y-3">
            <Label>Strategy</Label>
            <div className="flex flex-wrap gap-2">
              {strategies.map((option) => (
                <Tooltip key={option.value}>
                  <TooltipTrigger asChild>
                    <button
                      type="button"
                      onClick={() =>
                        setForm((prev) => ({ ...prev, strategy: option.value }))
                      }
                      className={cn(
                        "flex flex-1 min-w-[95px] flex-col rounded-xl border px-4 py-3 text-left transition hover:border-primary/50",
                        form.strategy === option.value
                          ? "border-primary bg-primary/5"
                          : "border-border bg-background",
                      )}
                    >
                      <span className="text-sm font-semibold uppercase text-muted-foreground">
                        {option.label}
                      </span>
                      <span className="text-sm text-foreground">{option.helper}</span>
                    </button>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs">
                    <p className="text-xs font-medium">{option.example}</p>
                  </TooltipContent>
                </Tooltip>
              ))}
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Label htmlFor="userId">User ID (optional)</Label>
              <Tooltip>
                <TooltipTrigger asChild>
                  <CircleUserRound className="h-4 w-4 text-muted-foreground" />
                </TooltipTrigger>
                <TooltipContent className="max-w-xs text-xs">
                  Provide an existing MovieLens user (1-943). If omitted, the
                  backend falls back to the nearest-neighbor cold-start index.
                </TooltipContent>
              </Tooltip>
            </div>
            <Input
              id="userId"
              type="number"
              min={1}
              value={form.userId}
              onChange={(event) =>
                setForm((prev) => ({ ...prev, userId: event.target.value }))
              }
              placeholder="e.g. 42"
            />
          </div>

          <div className="flex gap-2">
            <Button
              className="flex-1"
              size="lg"
              onClick={handleSubmit}
              disabled={isLoading}
            >
              {isLoading
              ? currentStep || "Processing..."
              : "Get Recommendations"}
            </Button>
            <Button
              type="button"
              variant="outline"
              size="lg"
              onClick={resetForm}
              disabled={isLoading}
            >
              Reset
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            POST /recommendations · payload mirrors the CLI: {"{ mood_text, user_id?, strategy, top_n }"}
          </p>
        </CardContent>
      </Card>

      <div className="space-y-6">
        <Card className="border border-primary/10 bg-card/80 dark:bg-card/90 shadow-md backdrop-blur dark:backdrop-blur-md">
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2 text-lg">
                <BadgeCheck className="h-5 w-5 text-primary" />
                Detection Summary
              </CardTitle>
              <p className="text-sm text-muted-foreground">
                Emotion and user context from your mood
              </p>
            </div>
            {lastResponse && (
              <Badge variant="secondary" className="text-xs">
                {lastResponse.existing_user
                  ? "Existing user"
                  : "New user recommendations"}
              </Badge>
            )}
          </CardHeader>
          <CardContent className="grid gap-4 md:grid-cols-2">
            <div className="space-y-3">
              <div className="rounded-2xl border border-dashed p-4">
                <p className="text-xs uppercase tracking-wide text-muted-foreground">
                  Lead emotion
                </p>
                {isLoading ? (
                  <Skeleton className="mt-3 h-6 w-32" />
                ) : topEmotion ? (
                  <div>
                    <p className="text-2xl font-semibold">{topEmotion.emotion}</p>
                    <p className="text-sm text-muted-foreground">
                      confidence {formatPercent(topEmotion.score)}
                    </p>
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    Submit a mood to see detected emotions.
                  </p>
                )}
              </div>

              <div className="flex flex-wrap gap-2">
                {(lastResponse?.emotions ?? []).map((emotion) => (
                  <Badge
                    key={emotion.emotion}
                    variant="outline"
                    className="text-sm"
                  >
                    {emotion.emotion} · {formatPercent(emotion.score)}
                  </Badge>
                ))}
                {isLoading && (
                  <>
                    <Skeleton className="h-6 w-16 rounded-full" />
                    <Skeleton className="h-6 w-20 rounded-full" />
                  </>
                )}
              </div>

              {showDeveloperMode && lastResponse && (
                <div className="text-xs text-muted-foreground font-mono bg-muted/50 p-2 rounded">
                  <p>User idx: {lastResponse.user_idx}</p>
                  {lastResponse.user_id && <p>User id: {lastResponse.user_id}</p>}
                </div>
              )}
            </div>

            <div className="grid gap-3">
              <Card className="bg-primary/5">
                <CardContent className="p-4">
                  <p className="text-xs uppercase tracking-wide text-muted-foreground">
                    Genre balance
                  </p>
                  <div className="mt-3 h-48">
                    {lastResponse ? (
                      <GenreBarChart data={lastResponse.genres} />
                    ) : (
                      <Skeleton className="h-full w-full" />
                    )}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-4">
                  <p className="text-xs uppercase tracking-wide text-muted-foreground">
                    Emotion blend
                  </p>
                  <div className="mt-3 h-48">
                    {lastResponse ? (
                      <EmotionRadarChart data={lastResponse.emotions} />
                    ) : (
                      <Skeleton className="h-full w-full" />
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
          </CardContent>
        </Card>

        <Card className="border border-border/60 bg-card/90 dark:bg-card/95 shadow dark:shadow-xl">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Activity className="h-5 w-5 text-primary" />
                  Recommendations
                </CardTitle>
                <p className="text-sm text-muted-foreground">
                  {filteredAndSortedResults.length} movie
                  {filteredAndSortedResults.length !== 1 ? "s" : ""} found
                </p>
              </div>
              <div className="flex items-center gap-2">
                {/* View Mode Toggle */}
                <div className="flex items-center gap-1 border rounded-lg p-1">
                  <button
                    type="button"
                    onClick={() => setViewMode("list")}
                    className={cn(
                      "p-1.5 rounded transition",
                      viewMode === "list"
                        ? "bg-primary text-primary-foreground"
                        : "hover:bg-muted"
                    )}
                    title="List view"
                  >
                    <List className="h-4 w-4" />
                  </button>
                  <button
                    type="button"
                    onClick={() => setViewMode("grid")}
                    className={cn(
                      "p-1.5 rounded transition",
                      viewMode === "grid"
                        ? "bg-primary text-primary-foreground"
                        : "hover:bg-muted"
                    )}
                    title="Grid view"
                  >
                    <LayoutGrid className="h-4 w-4" />
                  </button>
                </div>
              </div>
            </div>
            
            {/* Filters and Sort */}
            {lastResponse && lastResponse.results.length > 0 && (
              <div className="flex flex-wrap items-center gap-3 mt-4 pt-4 border-t">
                <div className="flex items-center gap-2">
                  <Filter className="h-4 w-4 text-muted-foreground" />
                  <Label className="text-xs">Genre:</Label>
                  <select
                    value={filterGenre}
                    onChange={(e) => setFilterGenre(e.target.value)}
                    className="text-xs border rounded px-2 py-1 bg-background"
                  >
                    <option value="all">All genres</option>
                    {availableGenres.map((genre) => (
                      <option key={genre} value={genre}>
                        {genre}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="flex items-center gap-2">
                  <ArrowUpDown className="h-4 w-4 text-muted-foreground" />
                  <Label className="text-xs">Sort:</Label>
                  <select
                    value={sortBy}
                    onChange={(e) =>
                      setSortBy(e.target.value as "rating" | "imdb" | "relevance")
                    }
                    className="text-xs border rounded px-2 py-1 bg-background"
                  >
                    <option value="relevance">Relevance</option>
                    <option value="rating">Predicted Rating</option>
                    <option value="imdb">IMDB Rating</option>
                  </select>
                </div>
              </div>
            )}
          </CardHeader>
          <CardContent className="space-y-4">
            {isLoading && (
              <div className="space-y-3">
                {Array.from({ length: 3 }).map((_, index) => (
                  <Skeleton key={index} className="h-20 w-full rounded-xl" />
                ))}
              </div>
            )}
            {!isLoading && lastResponse && lastResponse.results.length === 0 && (
              <p className="text-sm text-muted-foreground">
                No titles returned. Try a different strategy or ensure the user id
                exists in MovieLens.
              </p>
            )}
            {lastResponse && filteredAndSortedResults.length > 0 && (
              <div
                className={
                  viewMode === "grid"
                    ? "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
                    : "grid gap-4"
                }
              >
                {filteredAndSortedResults.map((movie) => {
                  return (
                  <Card
                    key={movie.movie_id}
                    className={cn(
                      "border border-border/70 bg-gradient-to-r from-card via-card to-primary/5 dark:from-card dark:via-card/98 dark:to-primary/10 overflow-hidden",
                      "shadow-sm dark:shadow-md hover:shadow-md dark:hover:shadow-lg transition-shadow duration-300",
                      viewMode === "grid" && "flex flex-col"
                    )}
                  >
                    <CardContent className={cn("p-0", viewMode === "grid" ? "flex flex-col h-full" : "")}>
                      <div className={cn(
                        viewMode === "grid" ? "flex flex-col" : "flex gap-4"
                      )}>
                        {/* Poster */}
                        {movie.poster_url ? (
                          <div className={cn(
                            viewMode === "grid" 
                              ? "w-full aspect-[2/3] overflow-hidden" 
                              : "flex-shrink-0"
                          )}>
                            <img
                              src={movie.poster_url}
                              alt={`${movie.title} poster`}
                              className={cn(
                                viewMode === "grid"
                                  ? "w-full h-full object-cover"
                                  : "h-32 w-24 object-cover rounded-l-lg"
                              )}
                              onError={(e) => {
                                (e.target as HTMLImageElement).style.display = "none";
                              }}
                            />
                          </div>
                        ) : (
                          <div className={cn(
                            viewMode === "grid"
                              ? "w-full aspect-[2/3] bg-muted flex items-center justify-center"
                              : "flex-shrink-0 h-32 w-24 bg-muted rounded-l-lg flex items-center justify-center"
                          )}>
                            <span className="text-xs text-muted-foreground">No poster</span>
                          </div>
                        )}

                        {/* Content */}
                        <div className={cn(
                          "p-4 space-y-2",
                          viewMode === "grid" ? "flex-1 flex flex-col" : "flex-1"
                        )}>
                          <div className={cn(
                            viewMode === "grid" 
                              ? "space-y-2" 
                              : "flex items-start justify-between gap-4"
                          )}>
                            <div className="flex-1">
                              <div className={cn(
                                "mb-1",
                                viewMode === "grid" 
                                  ? "flex flex-col gap-1" 
                                  : "flex items-center gap-2"
                              )}>
                                <div className="flex items-center gap-2">
                                  <span className={cn(
                                    "font-bold text-primary",
                                    viewMode === "grid" ? "text-lg" : "text-2xl"
                                  )}>
                                    #{movie.rank}
                                  </span>
                                  <h3 className={cn(
                                    "font-semibold",
                                    viewMode === "grid" ? "text-base" : "text-lg"
                                  )}>
                                    {movie.title}
                                  </h3>
                                </div>
                              </div>
                              <div className="flex flex-wrap gap-1.5 mb-2">
                                {movie.genres.map((genre) => (
                                  <Badge key={genre} variant="secondary" className="text-xs">
                                    {genre}
                                  </Badge>
                                ))}
                              </div>
                              {/* Ratings in grid view */}
                              {viewMode === "grid" && (
                                <div className="flex items-center gap-3 mb-2">
                                  <div>
                                    <p className="text-xs text-muted-foreground">Predicted</p>
                                    <p className="text-lg font-semibold">
                                      {movie.predicted_rating.toFixed(1)}
                                    </p>
                                  </div>
                                  {movie.imdb_rating && (
                                    <div>
                                      <p className="text-xs text-muted-foreground">IMDB</p>
                                      <p className="text-base font-semibold text-yellow-600">
                                        ⭐ {movie.imdb_rating.toFixed(1)}
                                      </p>
                                    </div>
                                  )}
                                </div>
                              )}
                            </div>

                            {/* Ratings - List view only */}
                            {viewMode === "list" && (
                              <div className="flex flex-col items-end gap-1 text-right">
                                <div>
                                  <p className="text-xs text-muted-foreground">Predicted</p>
                                  <p className="text-xl font-semibold">
                                    {movie.predicted_rating.toFixed(1)}
                                  </p>
                                </div>
                                {movie.imdb_rating && (
                                  <div>
                                    <p className="text-xs text-muted-foreground">IMDB</p>
                                    <p className="text-lg font-semibold text-yellow-600">
                                      ⭐ {movie.imdb_rating.toFixed(1)}
                                    </p>
                                  </div>
                                )}
                              </div>
                            )}
                          </div>

                          {/* Watch Providers & Links */}
                          <div className="flex flex-wrap items-center gap-3 pt-2 border-t">
                            {movie.watch_providers && movie.watch_providers.length > 0 ? (
                              <div className="flex-1 space-y-2">
                                <span className="text-xs font-medium text-muted-foreground">Where to watch:</span>
                                <div className="flex flex-wrap gap-2">
                                  {movie.watch_providers.map((provider, idx) => {
                                    const providerTypeLabel = 
                                      provider.type === "flatrate" ? "Stream" :
                                      provider.type === "rent" ? "Rent" :
                                      provider.type === "buy" ? "Buy" : provider.type;
                                    
                                    // Use provider URL, or fallback to TMDB watch page
                                    const watchUrl = provider.url || (movie.tmdb_url ? `${movie.tmdb_url}/watch` : "#");
                                    
                                    return (
                                      <a
                                        key={`${provider.name}-${idx}`}
                                        href={watchUrl}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="inline-flex items-center gap-1.5 rounded-md border border-border bg-background px-2.5 py-1.5 text-xs font-medium transition-all hover:bg-primary/10 hover:border-primary/50 hover:shadow-sm cursor-pointer active:scale-95 no-underline"
                                        title={`Click to watch on ${provider.name} (${providerTypeLabel})`}
                                      >
                                        {provider.logo ? (
                                          <img
                                            src={provider.logo}
                                            alt={provider.name}
                                            className="h-4 w-4 object-contain flex-shrink-0"
                                            onError={(e) => {
                                              (e.target as HTMLImageElement).style.display = "none";
                                            }}
                                          />
                                        ) : (
                                          <span className="h-4 w-4 flex items-center justify-center text-[10px] flex-shrink-0">📺</span>
                                        )}
                                        <span className="whitespace-nowrap">{provider.name}</span>
                                        <span className="text-[10px] text-muted-foreground whitespace-nowrap">({providerTypeLabel})</span>
                                      </a>
                                    );
                                  })}
                                </div>
                              </div>
                            ) : (
                              <p className="text-xs text-muted-foreground italic">
                                Watch provider info not available
                              </p>
                            )}
                            
                            {/* Ratings Display */}
                            {(movie.imdb_rating != null || movie.tmdb_rating != null) && (
                              <div className="flex flex-wrap items-center gap-4 pt-2 border-t">
                                <span className="text-xs font-medium text-muted-foreground">Ratings:</span>
                                <div className="flex flex-wrap gap-4 items-center">
                                  {movie.imdb_rating != null && typeof movie.imdb_rating === 'number' && (
                                    <div className="flex items-center gap-1.5">
                                      <span className="text-xs font-semibold text-yellow-600 dark:text-yellow-500">
                                        IMDb
                                      </span>
                                      <span className="text-sm font-bold">
                                        {movie.imdb_rating.toFixed(1)}
                                      </span>
                                      <span className="text-xs text-muted-foreground">/10</span>
                                    </div>
                                  )}
                                  {movie.tmdb_rating != null && typeof movie.tmdb_rating === 'number' && (
                                    <div className="flex items-center gap-1.5">
                                      <span className="text-xs font-semibold text-blue-600 dark:text-blue-500">
                                        TMDB
                                      </span>
                                      <span className="text-sm font-bold">
                                        {movie.tmdb_rating.toFixed(1)}
                                      </span>
                                      <span className="text-xs text-muted-foreground">/10</span>
                                    </div>
                                  )}
                                </div>
                              </div>
                            )}
                            
                            <div className="flex gap-3 ml-auto items-center">
                              {movie.imdb_url && (
                                <a
                                  href={movie.imdb_url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-xs text-primary hover:underline font-medium"
                                >
                                  IMDB →
                                </a>
                              )}
                              {movie.tmdb_url && (
                                <a
                                  href={movie.tmdb_url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-xs text-primary hover:underline font-medium"
                                >
                                  TMDB →
                                </a>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Developer Mode Toggle */}
        <div className="flex items-center justify-between">
          <button
            type="button"
            onClick={() => setShowDeveloperMode(!showDeveloperMode)}
            className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition"
          >
            <Code className="h-4 w-4" />
            {showDeveloperMode ? "Hide" : "Show"} Developer Mode
          </button>
        </div>

        {/* Developer Console - Collapsible */}
        {showDeveloperMode && (
          <Tabs defaultValue="console" className="w-full">
            <TabsList className="w-full justify-start bg-muted/40">
              <TabsTrigger value="console" className="flex items-center gap-2">
                <Terminal className="h-4 w-4" />
                Console
              </TabsTrigger>
              <TabsTrigger value="json">Raw JSON</TabsTrigger>
            </TabsList>
            <TabsContent value="console">
              <Card className="bg-black dark:bg-black/90 text-green-400 dark:text-green-300 border border-green-500/20">
                <CardContent className="space-y-1 overflow-x-auto p-4 font-mono text-sm">
                  {consoleLines.map((line, idx) => (
                    <p key={`${line}-${idx}`} className="whitespace-pre-wrap">
                      {line}
                    </p>
                  ))}
                </CardContent>
              </Card>
            </TabsContent>
            <TabsContent value="json">
              <Card className="bg-slate-950 dark:bg-slate-950/95 text-slate-100 dark:text-slate-200 border border-slate-700/50">
                <CardContent className="overflow-x-auto p-4">
                  <pre className="text-xs">
                    {lastResponse
                      ? JSON.stringify(lastResponse, null, 2)
                      : "// Submit a request to inspect the payload"}
                  </pre>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        )}
      </div>
    </section>
  );
};

const GenreBarChart = ({ data }: { data: RecommendationResponse["genres"] }) => {
  if (!data?.length) return <Skeleton className="h-full w-full" />;
  const prepared = data.map((item) => ({
    genre: item.genre,
    score: Number(item.score.toFixed(3)),
  }));
  
  // Get theme-aware colors - use a hook to detect theme
  const [isDark, setIsDark] = useState(false);
  
  useEffect(() => {
    const checkTheme = () => {
      const isDarkMode = document.documentElement.classList.contains('dark');
      setIsDark(isDarkMode);
    };
    checkTheme();
    const observer = new MutationObserver(checkTheme);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    return () => observer.disconnect();
  }, []);
  
  // Use theme-aware colors with fallbacks
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
  const textColor = isDark ? '#f8f9fa' : '#1a1a1a';
  const tooltipBg = isDark ? '#1e1e1e' : '#ffffff';
  const tooltipBorder = isDark ? 'rgba(255, 255, 255, 0.2)' : 'rgba(0, 0, 0, 0.1)';
  const tooltipText = isDark ? '#f8f9fa' : '#1a1a1a';
  // Use chart-3 color - blue/purple that works in both themes
  const barColor = isDark ? '#8b5cf6' : '#6366f1'; // Purple/indigo that's visible in both
  
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={prepared} layout="vertical" margin={{ left: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
        <XAxis
          type="number"
          tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
          domain={[0, 1]}
          tick={{ fill: textColor, fontSize: 12 }}
          axisLine={{ stroke: gridColor }}
          label={{ value: 'Score', position: 'insideBottom', fill: textColor, style: { textAnchor: 'middle' } }}
        />
        <YAxis 
          type="category" 
          dataKey="genre"
          tick={{ fill: textColor, fontSize: 12 }}
          axisLine={{ stroke: gridColor }}
          width={100}
        />
        <RechartsTooltip
          formatter={(value: number) => formatPercent(value)}
          contentStyle={{
            background: tooltipBg,
            color: tooltipText,
            borderRadius: "0.75rem",
            borderColor: tooltipBorder,
            borderWidth: "1px",
          }}
          labelStyle={{ color: tooltipText, fontWeight: 'bold' }}
        />
        <Bar 
          dataKey="score" 
          fill={barColor}
          radius={6}
        />
      </BarChart>
    </ResponsiveContainer>
  );
};

const EmotionRadarChart = ({ data }: { data: EmotionScore[] }) => {
  if (!data?.length) return <Skeleton className="h-full w-full" />;
  const prepared = data.map((item) => ({
    emotion: item.emotion,
    score: Number(item.score.toFixed(3)),
  }));
  
  // Get theme-aware colors using CSS variables
  const gridColor = "hsl(var(--border))";
  const textColor = "hsl(var(--foreground))";
  const tooltipBg = "hsl(var(--popover))";
  const tooltipBorder = "hsl(var(--border))";
  const tooltipText = "hsl(var(--popover-foreground))";
  
  return (
    <ResponsiveContainer width="100%" height="100%">
      <RadarChart data={prepared}>
        <PolarGrid stroke={gridColor} />
        <PolarAngleAxis 
          dataKey="emotion"
          tick={{ fill: textColor }}
        />
        <RechartsTooltip
          formatter={(value: number) => formatPercent(value)}
          contentStyle={{
            background: tooltipBg,
            color: tooltipText,
            borderRadius: "0.75rem",
            borderColor: tooltipBorder,
            borderWidth: "1px",
          }}
        />
        <Radar
          name="confidence"
          dataKey="score"
          stroke="hsl(var(--chart-1))"
          fill="hsl(var(--chart-1))"
          fillOpacity={0.4}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
};



