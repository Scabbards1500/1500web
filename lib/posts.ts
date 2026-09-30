import fs from "fs";
import path from "path";

export type Category = {
  slug: string;
  label: string;
  dir: string;
  description: string;
};

export const CATEGORIES: Category[] = [
  { slug: "ai-engineering", label: "AI Engineering", dir: "AI-Enginnering", description: "AI Infra、环境配置与工程实践" },
  { slug: "paper-notes", label: "Paper Notes", dir: "AI-PaperNotes", description: "论文阅读笔记" },
  { slug: "course", label: "Course", dir: "Course", description: "课程笔记" },
  { slug: "interview", label: "Interview", dir: "Interview", description: "Leetcode 与系统设计面试准备" },
];

export type PostMeta = {
  title: string;
  abstract?: string;
  date?: string;
  tags?: string[];
  category: string;
  /** 子文件夹路径，如 "Leetcode"、"AI Infra"；根目录下为空串 */
  series: string;
  /** 相对分类目录的路径段（不含 .md），用于 URL */
  slug: string[];
};

export type Post = PostMeta & {
  content: string;
  /** 相对 content/ 的目录，用于解析文章里的相对图片路径 */
  assetDir: string;
};

const CONTENT_DIR = path.join(process.cwd(), "content");

// Obsidian 新建仓库时自带的欢迎页
const IGNORED_FILES = new Set(["欢迎.md"]);
const HEADER_KEYS = new Set(["abstract", "time", "date", "tags"]);

type Header = Record<string, string>;

/**
 * 支持两种头部写法：
 * 1. 标准 front matter（--- 包裹）
 * 2. 文件开头连续的 `key: value` 行（abstract / time / tags），遇到第一行非头部内容即停止
 */
function parseHeader(markdown: string): { header: Header; content: string } {
  const text = markdown.replace(/^﻿/, "").replace(/\r\n/g, "\n");
  const header: Header = {};

  const frontMatter = text.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/);
  if (frontMatter) {
    for (const line of frontMatter[1].split("\n")) {
      const m = line.match(/^(\w+):\s*(.*)$/);
      if (m) header[m[1].toLowerCase()] = m[2].trim().replace(/^['"]|['"]$/g, "");
    }
    return { header, content: frontMatter[2].trim() };
  }

  const lines = text.split("\n");
  let index = 0;
  while (index < lines.length) {
    const m = lines[index].match(/^(\w+):\s*(.*)$/);
    if (!m || !HEADER_KEYS.has(m[1].toLowerCase())) break;
    header[m[1].toLowerCase()] = m[2].trim();
    index += 1;
  }
  return { header, content: lines.slice(index).join("\n").trim() };
}

function normalizeDate(value?: string): string | undefined {
  if (!value) return undefined;
  const m = value.match(/(\d{4})[/.-](\d{1,2})[/.-](\d{1,2})/);
  if (!m) return undefined;
  return `${m[1]}-${m[2].padStart(2, "0")}-${m[3].padStart(2, "0")}`;
}

function walkMarkdown(dir: string): string[] {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    if (entry.name.startsWith(".")) return [];
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) return walkMarkdown(full);
    if (entry.name.endsWith(".md") && !IGNORED_FILES.has(entry.name)) return [full];
    return [];
  });
}

function sortByDate<T extends { date?: string; title: string }>(posts: T[]): T[] {
  return posts.sort((a, b) => {
    if (a.date && b.date) return b.date.localeCompare(a.date);
    if (a.date) return -1;
    if (b.date) return 1;
    return a.title.localeCompare(b.title);
  });
}

function readCategory(category: Category): Post[] {
  const baseDir = path.join(CONTENT_DIR, category.dir);

  const posts = walkMarkdown(baseDir).flatMap((file) => {
    const { header, content } = parseHeader(fs.readFileSync(file, "utf-8"));
    if (!content) return [];

    const relative = path.relative(baseDir, file).split(path.sep);
    const fileName = relative[relative.length - 1].replace(/\.md$/, "");
    const post: Post = {
      title: fileName,
      category: category.slug,
      series: relative.slice(0, -1).join(" / "),
      slug: [...relative.slice(0, -1), fileName],
      content,
      assetDir: path.relative(CONTENT_DIR, path.dirname(file)).split(path.sep).join("/"),
    };
    if (header.abstract) post.abstract = header.abstract;
    const date = normalizeDate(header.time ?? header.date);
    if (date) post.date = date;
    if (header.tags) post.tags = header.tags.split(/[;,，]/).map((t) => t.trim()).filter(Boolean);
    return [post];
  });

  return sortByDate(posts);
}

export function getCategory(slug: string): Category | undefined {
  return CATEGORIES.find((c) => c.slug === slug);
}

export function getPostsByCategory(slug: string): Post[] {
  const category = getCategory(slug);
  return category ? readCategory(category) : [];
}

export function getAllPosts(): Post[] {
  return sortByDate(CATEGORIES.flatMap(readCategory));
}

export function getPost(categorySlug: string, slug: string[]): Post | undefined {
  const key = slug.join("/");
  return getPostsByCategory(categorySlug).find((post) => post.slug.join("/") === key);
}

export function postHref(post: Pick<PostMeta, "category" | "slug">): string {
  return `/blog/${post.category}/${post.slug.map(encodeURIComponent).join("/")}`;
}

/** 列表页只需要元信息，不把正文传给客户端 */
export function toMeta({ content: _content, assetDir: _assetDir, ...meta }: Post): PostMeta {
  return meta;
}
