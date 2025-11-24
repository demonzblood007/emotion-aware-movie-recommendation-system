"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchHealth } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Loader2, ShieldCheck, ShieldAlert } from "lucide-react";

export const StatusIndicator = () => {
  const { data, isLoading, isError, refetch, isFetching } = useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    refetchInterval: 60_000,
    retry: 1,
    staleTime: 30_000, // Cache for 30 seconds
  });

  const label = isError
    ? "API unreachable"
    : data?.status === "ok"
      ? "API healthy"
      : "Unknown";

  const Icon = isError ? ShieldAlert : ShieldCheck;

  return (
    <button
      onClick={() => refetch()}
      className="inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-primary/5 to-primary/10 px-4 py-2 text-sm font-medium text-primary shadow-sm ring-1 ring-primary/10 transition hover:opacity-80"
    >
      <Badge
        variant={isError ? "destructive" : "outline"}
        className="flex items-center gap-1 border-transparent bg-white/80 text-xs font-semibold uppercase tracking-wide text-primary"
      >
        {isLoading || isFetching ? (
          <Loader2 className="h-3 w-3 animate-spin" />
        ) : (
          <Icon className="h-3 w-3" />
        )}
        status
      </Badge>
      <span className="text-sm">{label}</span>
      {data?.env && (
        <span className="rounded-full bg-white/70 px-2 py-0.5 text-xs text-muted-foreground">
          {data.env}
        </span>
      )}
    </button>
  );
};



