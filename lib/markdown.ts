import GithubSlugger from "github-slugger";
import { toString } from "mdast-util-to-string";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import remarkCjkFriendly from "remark-cjk-friendly";
import { unified } from "unified";
import type { Heading, Root, RootContent } from "mdast";

export type TocItem = {
  id: string;
  text: string;
  level: number;
};

// Obsidian 的 ![[image.png]] 嵌入语法转成标准 Markdown
export function normalizeObsidian(content: string): string {
  return content.replace(/!\[\[([^\]|]+)(?:\|[^\]]*)?\]\]/g, (_, file: string) => `![](${encodeURI(file.trim())})`);
}

function parseMarkdown(content: string): Root {
  return unified().use(remarkParse).use(remarkGfm).use(remarkCjkFriendly).use(remarkMath).parse(normalizeObsidian(content));
}

// 这些节点结束时换行，保证不同段落/单元格的文字不会粘在一起
const BLOCK_TYPES = new Set(["paragraph", "heading", "code", "math", "blockquote", "listItem", "tableRow", "thematicBreak"]);

function collectText(node: Root | RootContent, out: string[]) {
  if (node.type === "image" || node.type === "html") return;
  if ("value" in node && typeof node.value === "string") out.push(node.value);
  if (node.type === "break") out.push("\n");
  if ("children" in node) {
    for (const child of node.children) {
      collectText(child, out);
      if (child.type === "tableCell") out.push(" ");
    }
  }
  if (BLOCK_TYPES.has(node.type)) out.push("\n");
}

/** 把 Markdown 转成用于搜索的纯文本（保留代码块、表格内容，去掉语法符号和图片） */
export function toPlainText(content: string): string {
  const out: string[] = [];
  collectText(parseMarkdown(content), out);
  return out
    .join("")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{2,}/g, "\n")
    .trim();
}

function collectHeadings(node: Root | RootContent, out: Heading[]) {
  if (node.type === "heading") {
    out.push(node);
    return;
  }
  if ("children" in node) {
    for (const child of node.children) collectHeadings(child, out);
  }
}

/**
 * 提取文章标题生成目录。id 的生成方式与 rehype-slug 一致（github-slugger，按出现顺序去重），
 * 所以必须遍历所有级别的标题，即使目录里只展示其中一部分。
 */
export function extractToc(content: string, maxLevel = 5): TocItem[] {
  const tree = parseMarkdown(content);
  const headings: Heading[] = [];
  collectHeadings(tree, headings);

  const slugger = new GithubSlugger();
  return headings
    .map((heading) => {
      const text = toString(heading).trim();
      return { id: slugger.slug(text), text, level: heading.depth };
    })
    .filter((item) => item.text && item.level <= maxLevel);
}
