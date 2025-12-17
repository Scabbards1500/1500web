"use client";

import { useState, useMemo, useEffect } from "react";
import type { Post } from "@/lib/posts";

type KeywordFilterProps = {
  posts: Post[];
  onFilterChange: (filteredPosts: Post[]) => void;
};

export default function KeywordFilter({ posts, onFilterChange }: KeywordFilterProps) {
  // 提取所有唯一的 keywords，并按出现频率排序
  const allKeywords = useMemo(() => {
    const keywordCount = new Map<string, number>();
    posts.forEach((post) => {
      if (post.keywords) {
        post.keywords.forEach((keyword) => {
          keywordCount.set(keyword, (keywordCount.get(keyword) || 0) + 1);
        });
      }
    });
    // 按频率从高到低排序，频率相同时按字母顺序排序
    return Array.from(keywordCount.entries())
      .sort((a, b) => {
        if (b[1] !== a[1]) {
          return b[1] - a[1]; // 频率高的在前
        }
        return a[0].localeCompare(b[0]); // 频率相同时按字母顺序
      })
      .map(([keyword]) => keyword);
  }, [posts]);

  const [selectedKeywords, setSelectedKeywords] = useState<Set<string>>(new Set());

  // 根据选中的 keywords 筛选文章（并集逻辑：文章包含任意一个选中的 keyword 就显示）
  const filteredPosts = useMemo(() => {
    if (selectedKeywords.size === 0) {
      return posts;
    }
    return posts.filter((post) => {
      if (!post.keywords || post.keywords.length === 0) {
        return false;
      }
      // 并集（OR）：只要文章包含任意一个选中的 keyword 就显示
      // 例如：选中 "A" 和 "B"，会显示包含 A 或 B 的文章
      return post.keywords.some((keyword) => selectedKeywords.has(keyword));
    });
  }, [posts, selectedKeywords]);

  // 当筛选结果变化时通知父组件
  useEffect(() => {
    onFilterChange(filteredPosts);
  }, [filteredPosts, onFilterChange]);

  const toggleKeyword = (keyword: string) => {
    setSelectedKeywords((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(keyword)) {
        newSet.delete(keyword);
      } else {
        newSet.add(keyword);
      }
      return newSet;
    });
  };

  const clearFilters = () => {
    setSelectedKeywords(new Set());
  };

  if (allKeywords.length === 0) {
    return null;
  }

  return (
    <section className="mx-auto max-w-5xl px-6 pt-6 md:px-12">
      <div className="flex flex-wrap items-center gap-3">
        {allKeywords.map((keyword) => (
          <button
            key={keyword}
            onClick={() => toggleKeyword(keyword)}
            className={`rounded-full px-3 py-1 text-sm font-medium transition ${
              selectedKeywords.has(keyword)
                ? "bg-slate-900 text-white"
                : "bg-slate-100 text-slate-600 hover:bg-slate-200"
            }`}
          >
            {keyword}
          </button>
        ))}
        {selectedKeywords.size > 0 && (
          <button
            onClick={clearFilters}
            className="rounded-full bg-slate-200 px-3 py-1 text-sm font-medium text-slate-600 transition hover:bg-slate-300"
          >
            Clean
          </button>
        )}
      </div>
      {selectedKeywords.size > 0 && (
        <p className="mt-3 text-sm text-slate-500">
          {filteredPosts.length} results
        </p>
      )}
    </section>
  );
}

