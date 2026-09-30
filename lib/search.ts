export type SearchDoc = {
  href: string;
  title: string;
  abstract: string;
  category: string;
  series: string;
  date?: string;
  text: string;
};

/** [start, end) */
export type Range = [number, number];

export type Snippet = {
  text: string;
  ranges: Range[];
  /** 命中的原文，用于生成跳转到正文位置的 text fragment */
  match: string;
  leadingEllipsis: boolean;
  trailingEllipsis: boolean;
};

export type SearchResult = {
  doc: SearchDoc;
  titleRanges: Range[];
  abstractRanges: Range[];
  snippets: Snippet[];
  bodyCount: number;
};

export type SearchOutcome = {
  results: SearchResult[];
  /** 输入里带正则符号且是合法正则时为 true */
  usedRegex: boolean;
};

type Matcher = {
  find: (text: string) => Range[];
  accepts: (haystack: string) => boolean;
};

const MAX_MATCHES_PER_FIELD = 300;
const MAX_SNIPPETS = 3;
const SNIPPET_BEFORE = 30;
const SNIPPET_AFTER = 70;
const MAX_RESULTS = 50;
const REGEX_SYNTAX = /[.*+?^${}()|[\]\\]/;

function escapeRegExp(text: string) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function findAll(regex: RegExp, text: string): Range[] {
  const ranges: Range[] = [];
  regex.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = regex.exec(text)) && ranges.length < MAX_MATCHES_PER_FIELD) {
    if (m[0].length === 0) {
      // 空匹配（如 a*）不高亮，但要推进位置避免死循环
      regex.lastIndex += 1;
      continue;
    }
    ranges.push([m.index, m.index + m[0].length]);
  }
  return ranges;
}

/** 排序、合并重叠区间；只隔着空白的相邻命中也合并，如 “KV cache” 整体高亮 */
function mergeRanges(text: string, ranges: Range[]): Range[] {
  const sorted = [...ranges].sort((a, b) => a[0] - b[0] || b[1] - a[1]);
  const merged: Range[] = [];
  for (const [start, end] of sorted) {
    const last = merged[merged.length - 1];
    if (last && (start <= last[1] || /^\s*$/.test(text.slice(last[1], start)))) {
      last[1] = Math.max(last[1], end);
    } else {
      merged.push([start, end]);
    }
  }
  return merged;
}

/** 普通匹配：忽略大小写，空格分隔的每个关键词都要出现 */
function literalMatcher(query: string): Matcher {
  const terms = [...new Set(query.toLowerCase().split(/\s+/).filter(Boolean))];
  // 长词优先，避免短词抢先匹配导致长词高亮不完整
  const combined = new RegExp([...terms].sort((a, b) => b.length - a.length).map(escapeRegExp).join("|"), "gi");
  return {
    find: (text) => findAll(combined, text),
    accepts: (haystack) => {
      const lower = haystack.toLowerCase();
      return terms.every((term) => lower.includes(term));
    },
  };
}

/** 输入里有正则符号且能编译时，额外按正则匹配；不合法就当普通文本 */
function regexMatcher(query: string): Matcher | null {
  if (!REGEX_SYNTAX.test(query)) return null;
  let regex: RegExp;
  try {
    // 不加 u：u 模式会拒绝 \- 这类常见写法，中文匹配也不需要它
    regex = new RegExp(query, "gi");
  } catch {
    return null;
  }
  return {
    find: (text) => findAll(regex, text),
    accepts: (haystack) => findAll(regex, haystack).length > 0,
  };
}

/** 标题的“按顺序出现”模糊匹配，如 lcheap → Leetcode - Heap */
function subsequenceRanges(title: string, query: string): Range[] {
  const needle = query.toLowerCase().replace(/\s+/g, "");
  if (needle.length < 2) return [];
  const lower = title.toLowerCase();
  const ranges: Range[] = [];
  let from = 0;
  for (const char of needle) {
    const index = lower.indexOf(char, from);
    if (index === -1) return [];
    const last = ranges[ranges.length - 1];
    if (last && last[1] === index) last[1] = index + 1;
    else ranges.push([index, index + 1]);
    from = index + 1;
  }
  return ranges;
}

function buildSnippets(text: string, ranges: Range[]): Snippet[] {
  const snippets: Snippet[] = [];
  let coveredUntil = -1;
  for (const [start, end] of ranges) {
    if (start < coveredUntil) continue;
    const from = Math.max(0, start - SNIPPET_BEFORE);
    const to = Math.min(text.length, end + SNIPPET_AFTER);
    snippets.push({
      // 换行替换成空格，长度不变，所以位置不用重算
      text: text.slice(from, to).replace(/\n/g, " "),
      ranges: ranges.filter(([s, e]) => s >= from && e <= to).map(([s, e]): Range => [s - from, e - from]),
      match: text.slice(start, end),
      leadingEllipsis: from > 0,
      trailingEllipsis: to < text.length,
    });
    coveredUntil = to;
    if (snippets.length >= MAX_SNIPPETS) break;
  }
  return snippets;
}

/**
 * 一次搜索同时做三种匹配，结果取并集：
 * 1. 普通关键词（空格分隔，全部出现）
 * 2. 正则（仅当输入含正则符号且合法）
 * 3. 标题按顺序模糊匹配
 */
export function search(docs: SearchDoc[], rawQuery: string): SearchOutcome {
  const query = rawQuery.trim();
  if (!query) return { results: [], usedRegex: false };

  const regex = regexMatcher(query);
  const matchers = [literalMatcher(query), ...(regex ? [regex] : [])];
  const findAllMatchers = (text: string) => mergeRanges(text, matchers.flatMap((m) => m.find(text)));

  const scored: Array<SearchResult & { score: number }> = [];
  for (const doc of docs) {
    const haystack = `${doc.title}\n${doc.abstract}\n${doc.text}`;
    const accepted = matchers.some((m) => m.accepts(haystack));
    let titleRanges = accepted ? findAllMatchers(doc.title) : [];
    let fuzzyTitle = false;

    if (titleRanges.length === 0) {
      const subsequence = subsequenceRanges(doc.title, query);
      if (subsequence.length > 0) {
        titleRanges = subsequence;
        fuzzyTitle = true;
      } else if (!accepted) {
        continue;
      }
    }

    const abstractRanges = accepted ? findAllMatchers(doc.abstract) : [];
    const bodyRanges = accepted ? findAllMatchers(doc.text) : [];
    const score =
      (titleRanges.length > 0 ? (fuzzyTitle ? 8 : 20) : 0) +
      Math.min(abstractRanges.length, 5) * 3 +
      Math.min(bodyRanges.length, 50);

    scored.push({
      doc,
      titleRanges,
      abstractRanges,
      snippets: buildSnippets(doc.text, bodyRanges),
      bodyCount: bodyRanges.length,
      score,
    });
  }

  const results = scored
    .sort((a, b) => b.score - a.score || (b.doc.date ?? "").localeCompare(a.doc.date ?? ""))
    .slice(0, MAX_RESULTS);
  return { results, usedRegex: regex !== null };
}

/** 生成浏览器 text fragment（#:~:text=），打开文章时自动滚动并高亮到命中位置 */
export function textFragmentHref(href: string, match: string): string {
  const text = match.trim();
  if (text.length < 2 || text.includes("\n")) return href;
  return `${href}#:~:text=${encodeURIComponent(text).replace(/-/g, "%2D")}`;
}
