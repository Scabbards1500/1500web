"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import type { TocItem } from "@/lib/markdown";

type TableOfContentsProps = {
  items: TocItem[];
  title: string;
  backHref: string;
  backLabel: string;
};

// 顶部导航栏高度 + 一点余量：标题越过这条线就算“正在阅读”
const ACTIVE_OFFSET = 120;

// 按相对层级（0 = 文章里最高一级标题）区分字号、字重和颜色
const LEVEL_STYLES = [
  "py-1.5 text-[15px] font-semibold text-slate-900",
  "py-1 text-sm font-medium text-slate-600",
  "py-1 text-[13px] text-slate-500",
  "py-0.5 text-xs text-slate-500",
  "py-0.5 text-xs text-slate-500",
];

export default function TableOfContents({ items, title, backHref, backLabel }: TableOfContentsProps) {
  const [activeId, setActiveId] = useState<string | null>(items[0]?.id ?? null);
  const [mobileOpen, setMobileOpen] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);
  // 点击跳转的平滑滚动期间不更新高亮，避免高亮在中间的标题间闪烁
  const lockUntilRef = useRef(0);

  const updateActive = useCallback(() => {
    if (Date.now() < lockUntilRef.current) return;
    const headings = items
      .map((item) => document.getElementById(item.id))
      .filter((el): el is HTMLElement => el !== null);
    if (headings.length === 0) return;

    const atBottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4;
    let current = headings[0].id;
    if (atBottom) {
      current = headings[headings.length - 1].id;
    } else {
      for (const heading of headings) {
        if (heading.getBoundingClientRect().top <= ACTIVE_OFFSET) current = heading.id;
        else break;
      }
    }
    setActiveId(current);
  }, [items]);

  useEffect(() => {
    let frame = 0;
    const onScroll = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(updateActive);
    };
    updateActive();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, [updateActive]);

  // 目录很长时，让高亮项保持在侧栏可视范围内（只滚动侧栏，不影响正文）
  useEffect(() => {
    const container = listRef.current;
    const active = container?.querySelector<HTMLElement>(`[data-toc-id="${CSS.escape(activeId ?? "")}"]`);
    if (!container || !active) return;
    const top = active.offsetTop - container.offsetTop;
    if (top < container.scrollTop + 40 || top > container.scrollTop + container.clientHeight - 60) {
      container.scrollTo({ top: top - container.clientHeight / 3, behavior: "smooth" });
    }
  }, [activeId]);

  const handleClick = (event: React.MouseEvent<HTMLAnchorElement>, id: string) => {
    const target = document.getElementById(id);
    if (!target) return;
    event.preventDefault();
    lockUntilRef.current = Date.now() + 1000;
    setActiveId(id);
    setMobileOpen(false);
    target.scrollIntoView({ behavior: "smooth", block: "start" });
    history.replaceState(null, "", `#${encodeURIComponent(id)}`);
  };

  const baseLevel = Math.min(...items.map((item) => item.level));

  return (
    <div className="flex flex-col px-6 py-6 md:h-full md:py-10">
      <Link href={backHref} className="text-xs font-semibold uppercase tracking-[0.25em] text-slate-400 transition hover:text-slate-900">
        ← {backLabel}
      </Link>
      <p className="mt-4 text-base font-semibold leading-snug text-slate-900">{title}</p>

      {items.length > 0 && (
        <button
          type="button"
          onClick={() => setMobileOpen((open) => !open)}
          className="mt-4 self-start text-xs font-medium text-slate-500 hover:text-slate-900 md:hidden"
        >
          {mobileOpen ? "收起章节 ▴" : "展开章节 ▾"}
        </button>
      )}

      <div
        ref={listRef}
        className={`${mobileOpen ? "block" : "hidden"} toc-scroll mt-4 max-h-[60vh] overflow-y-auto border-t border-slate-200 pt-4 md:mt-6 md:block md:max-h-none md:min-h-0 md:flex-1`}
      >
        {items.length === 0 ? (
          <p className="text-sm text-slate-400">本文没有章节标题</p>
        ) : (
          <ul className="border-l border-slate-200">
            {items.map((item) => {
              const active = item.id === activeId;
              const depth = Math.min(item.level - baseLevel, LEVEL_STYLES.length - 1);
              return (
                <li key={item.id} className={depth === 0 ? "mt-3 first:mt-0" : ""}>
                  <a
                    href={`#${item.id}`}
                    data-toc-id={item.id}
                    onClick={(event) => handleClick(event, item.id)}
                    style={{ paddingLeft: `${depth * 18 + 14}px` }}
                    className={`-ml-px block border-l-2 pr-2 leading-snug transition-colors ${LEVEL_STYLES[depth]} ${
                      active
                        ? "border-sky-600 text-sky-700!"
                        : "border-transparent hover:border-slate-300 hover:text-slate-900!"
                    }`}
                  >
                    {item.text}
                  </a>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      <button
        type="button"
        onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
        className="mt-4 hidden self-start text-xs font-medium text-slate-400 transition hover:text-slate-900 md:block"
      >
        ↑ 回到顶部
      </button>
    </div>
  );
}
