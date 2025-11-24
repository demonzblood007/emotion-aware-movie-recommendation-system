"use client";

import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  // Prevent hydration mismatch
  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <Button
        variant="outline"
        size="icon"
        className="h-10 w-10 rounded-full"
        disabled
      >
        <Sun className="h-5 w-5" />
      </Button>
    );
  }

  return (
    <Button
      variant="outline"
      size="icon"
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
      className={cn(
        "h-10 w-10 rounded-full relative overflow-hidden",
        "transition-all duration-300 ease-in-out",
        "hover:scale-110 hover:shadow-lg",
        "border-2",
        theme === "dark"
          ? "border-primary/20 bg-primary/5 hover:bg-primary/10"
          : "border-primary/30 bg-background hover:bg-accent"
      )}
      aria-label="Toggle theme"
    >
      <div className="relative w-full h-full flex items-center justify-center">
        <Sun
          className={cn(
            "h-5 w-5 absolute transition-all duration-500 ease-in-out",
            "transform-gpu",
            theme === "dark"
              ? "rotate-180 scale-0 opacity-0"
              : "rotate-0 scale-100 opacity-100"
          )}
        />
        <Moon
          className={cn(
            "h-5 w-5 absolute transition-all duration-500 ease-in-out",
            "transform-gpu",
            theme === "dark"
              ? "rotate-0 scale-100 opacity-100"
              : "-rotate-180 scale-0 opacity-0"
          )}
        />
      </div>
    </Button>
  );
}

