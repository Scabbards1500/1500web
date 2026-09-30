"use client";

import { createContext, Fragment, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { search, textFragmentHref, type Range, type SearchDoc } from "@/lib/search";

type SearchState = {
  docs: SearchDoc[] | null;
  loadError: boolean;
  /** 按回车提交的查询；null 表示没有在搜索，页面显示原内容 */
  submitted: string | null;
  loadIndex: () => void;
  submit: (query: string) => void;
  clear: () => void;
};

const SearchContext = createContext<SearchState | null>(null);

function useSearch() {
  const ctx = useContext(SearchContext);
  if (!ctx) throw new Error("useSearch must be used inside <SearchProvider>");
  return ctx;
}

export function SearchProvider({ children }: { children: ReactNode }) {
  const [docs, setDocs] = useState<SearchDoc[] | null>(null);
  const [loadError, setLoadError] = useState(false);
  const [submitted, setSubmitted] = useState<string | null>(null);
  const loadingRef = useRef(false);

  // 展开搜索框时就开始加载索引，按回车时通常已经就绪
  const loadIndex = useCallback(() => {
    if (loadingRef.current) return;
    loadingRef.current = true;
    fetch("/search-index.json")
      .then((res) => (res.ok ? res.json() : Promise.reject(res.status)))
      .then((data: SearchDoc[]) => setDocs(data))
      .catch(() => {
        loadingRef.current = false;
        setLoadError(true);
      });
  }, []);

  const submit = useCallback(
    (query: string) => {
      const trimmed = query.trim();
      if (!trimmed) return;
      loadIndex();
      setLoadError(false);
      setSubmitted(trimmed);
      window.scrollTo({ top: 0 });
    },
    [loadIndex],
  );

  const clear = useCallback(() => setSubmitted(null), []);

  const value = useMemo(
    () => ({ docs, loadError, submitted, loadIndex, submit, clear }),
    [docs, loadError, submitted, loadIndex, submit, clear],
  );
  return <SearchContext.Provider value={value}>{children}</SearchContext.Provider>;
}

function SearchIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24" aria-hidden>
      <circle cx="11" cy="11" r="7" />
      <path strokeLinecap="round" d="m20 20-3.5-3.5" />
    </svg>
  );
}

/** 导航栏里的放大镜：点击后在原位展开输入框，其他导航项随宽度动画向左让开 */
export function SearchBox() {
  const { submitted, loadIndex, submit, clear } = useSearch();
  const [open, setOpen] = useState(false);
  const [value, setValue] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const expand = useCallback(() => {
    setOpen(true);
    loadIndex();
    inputRef.current?.focus();
  }, [loadIndex]);

  const collapse = () => {
    setOpen(false);
    setValue("");
    clear();
    inputRef.current?.blur();
  };

  // Ctrl/Cmd + K 展开并聚焦
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        expand();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expand]);

  const onIconClick = () => {
    if (!open) expand();
    else if (value.trim()) submit(value);
    else collapse();
  };

  return (
    <div
      className={`flex items-center rounded-md border transition-colors duration-300 ${
        open ? "border-slate-300 bg-white shadow-sm" : "border-transparent"
      }`}
    >
      <button
        type="button"
        onClick={onIconClick}
        aria-label={open ? "搜索" : "展开搜索"}
        className="rounded-md px-3 py-2 text-slate-500 transition-colors hover:text-slate-900"
      >
        <SearchIcon className="h-4 w-4" />
      </button>
      <div
        className={`overflow-hidden transition-[width,opacity] duration-300 ease-out ${
          open ? "w-44 opacity-100 xl:w-60" : "w-0 opacity-0"
        }`}
      >
        <input
          ref={inputRef}
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              submit(value);
            } else if (event.key === "Escape") {
              collapse();
            }
          }}
          onBlur={() => {
            if (!value.trim() && submitted === null) setOpen(false);
          }}
          tabIndex={open ? 0 : -1}
          placeholder="关键词或正则，回车搜索"
          spellCheck={false}
          className="w-full bg-transparent py-1.5 pr-3 text-sm font-normal normal-case tracking-normal text-slate-900 outline-none placeholder:text-slate-400"
        />
      </div>
    </div>
  );
}

function Highlight({ text, ranges }: { text: string; ranges: Range[] }) {
  if (ranges.length === 0) return <>{text}</>;
  const parts: ReactNode[] = [];
  let cursor = 0;
  ranges.forEach(([start, end], index) => {
    if (start < cursor) return;
    if (start > cursor) parts.push(<Fragment key={`t${index}`}>{text.slice(cursor, start)}</Fragment>);
    parts.push(
      <mark key={`m${index}`} className="rounded bg-sky-100 px-0.5 font-medium text-sky-700">
        {text.slice(start, end)}
      </mark>,
    );
    cursor = end;
  });
  if (cursor < text.length) parts.push(<Fragment key="rest">{text.slice(cursor)}</Fragment>);
  return <>{parts}</>;
}

function SearchResults({ query }: { query: string }) {
  const { docs, loadError, clear } = useSearch();
  const outcome = useMemo(() => (docs ? search(docs, query) : null), [docs, query]);

  return (
    <section className="mx-auto max-w-5xl px-6 pt-12 md:px-12">
      <div className="flex flex-wrap items-end justify-between gap-4 border-b border-slate-200 pb-4">
        <div>
          <span className="text-xs font-semibold uppercase tracking-[0.35em] text-slate-400">Search</span>
          <h2 className="mt-2 text-2xl font-semibold text-slate-900">
            “<span className="text-sky-700">{query}</span>”
          </h2>
          {outcome && (
            <p className="mt-1 text-sm text-slate-500">
              {outcome.results.length} 篇相关文章
              {outcome.usedRegex && " · 已同时按正则匹配"}
            </p>
          )}
        </div>
        <button
          type="button"
          onClick={clear}
          className="rounded-full border border-slate-300 px-3 py-1 text-sm text-slate-600 transition hover:border-slate-900 hover:text-slate-900"
        >
          × 清除搜索
        </button>
      </div>

      {loadError ? (
        <p className="py-12 text-center text-sm text-red-600">搜索索引加载失败，请刷新重试</p>
      ) : !outcome ? (
        <p className="py-12 text-center text-sm text-slate-400">正在加载索引…</p>
      ) : outcome.results.length === 0 ? (
        <p className="py-12 text-center text-sm text-slate-400">没有找到相关内容</p>
      ) : (
        <ul className="mt-6 flex flex-col gap-4">
          {outcome.results.map((result) => (
            <li key={result.doc.href} className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <a href={result.doc.href} target="_blank" rel="noopener noreferrer" className="group block">
                <div className="flex items-baseline justify-between gap-3">
                  <h3 className="text-lg font-semibold text-slate-900 group-hover:underline">
                    <Highlight text={result.doc.title} ranges={result.titleRanges} />
                  </h3>
                  {result.bodyCount > 0 && <span className="shrink-0 text-xs text-slate-400">正文 {result.bodyCount} 处</span>}
                </div>
                <p className="mt-0.5 text-xs text-slate-400">
                  {result.doc.category}
                  {result.doc.series && ` / ${result.doc.series}`}
                  {result.doc.date && ` · ${result.doc.date.replace(/-/g, ".")}`}
                </p>
                {result.abstractRanges.length > 0 && (
                  <p className="mt-2 text-sm text-slate-600">
                    <Highlight text={result.doc.abstract} ranges={result.abstractRanges} />
                  </p>
                )}
              </a>
              {result.snippets.length > 0 && (
                <div className="mt-3 space-y-1 border-t border-slate-100 pt-3">
                  {result.snippets.map((snippet, index) => (
                    <a
                      key={index}
                      // 点击片段直接定位到正文中的对应位置
                      href={textFragmentHref(result.doc.href, snippet.match)}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block rounded-md px-2 py-1 text-sm leading-6 text-slate-600 transition hover:bg-slate-50"
                    >
                      {snippet.leadingEllipsis && "…"}
                      <Highlight text={snippet.text} ranges={snippet.ranges} />
                      {snippet.trailingEllipsis && "…"}
                    </a>
                  ))}
                </div>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

/** 有搜索时在当前页面主体显示结果；原内容只是隐藏，清除搜索后原样恢复（包括滚动状态外的所有状态） */
export function SearchMain({ children }: { children: ReactNode }) {
  const { submitted } = useSearch();
  return (
    <>
      {submitted !== null && <SearchResults query={submitted} />}
      <div className={submitted !== null ? "hidden" : undefined}>{children}</div>
    </>
  );
}
