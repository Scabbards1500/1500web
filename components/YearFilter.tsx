"use client";

import { useState, useMemo, useEffect } from "react";
import type { Post } from "@/lib/posts";

type YearFilterProps = {
  posts: Post[];
  onFilterChange: (filteredPosts: Post[]) => void;
};

export default function YearFilter({ posts, onFilterChange }: YearFilterProps) {
  // 可选的年份列表（从新到旧）
  const availableYears = ["2025", "2024", "2023", "2022"];

  const [selectedYears, setSelectedYears] = useState<Set<string>>(new Set());

  // 根据选中的年份筛选文章（并集逻辑：文章属于任意一个选中的年份就显示）
  const filteredPosts = useMemo(() => {
    if (selectedYears.size === 0) {
      return posts;
    }
    return posts.filter((post) => {
      if (!post.date) return false;
      const postYear = new Date(post.date).getFullYear().toString();
      // 并集（OR）：只要文章属于任意一个选中的年份就显示
      return selectedYears.has(postYear);
    });
  }, [posts, selectedYears]);

  // 当筛选结果变化时通知父组件
  useEffect(() => {
    onFilterChange(filteredPosts);
  }, [filteredPosts, onFilterChange]);

  const toggleYear = (year: string) => {
    setSelectedYears((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(year)) {
        newSet.delete(year);
      } else {
        newSet.add(year);
      }
      return newSet;
    });
  };

  const clearFilters = () => {
    setSelectedYears(new Set());
  };

  return (
    <section className="mx-auto max-w-5xl px-6 pt-6 md:px-12">
      <div className="flex flex-wrap items-center gap-3">
        {availableYears.map((year) => (
          <button
            key={year}
            onClick={() => toggleYear(year)}
            className={`rounded-full px-3 py-1 text-sm font-medium transition ${
              selectedYears.has(year)
                ? "bg-slate-900 text-white"
                : "bg-slate-100 text-slate-600 hover:bg-slate-200"
            }`}
          >
            {year}
          </button>
        ))}
        {selectedYears.size > 0 && (
          <button
            onClick={clearFilters}
            className="rounded-full bg-slate-200 px-3 py-1 text-sm font-medium text-slate-600 transition hover:bg-slate-300"
          >
            Clean
          </button>
        )}
      </div>
      {selectedYears.size > 0 && (
        <p className="mt-3 text-sm text-slate-500">
          {filteredPosts.length} results
        </p>
      )}
    </section>
  );
}

