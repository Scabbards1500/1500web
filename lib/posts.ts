import fs from "fs";
import path from "path";

export type PostMeta = {
  title: string;
  subtitle?: string;
  date?: string;
  image?: string;
  summary?: string;
  links?: Record<string, string>;
};

export type Post = PostMeta & {
  slug: string;
  content: string;
};

const CONTENT_DIR = path.join(process.cwd(), "content");
const POSTS_DIR = path.join(CONTENT_DIR, "posts");

type FrontMatterResult = {
  data: Record<string, string | Record<string, string>>;
  content: string;
};

function parseFrontMatter(markdown: string): FrontMatterResult {
  const frontMatterMatch = markdown.match(/^---\s*([\s\S]*?)---\s*([\s\S]*)$/);

  if (!frontMatterMatch) {
    return { data: {}, content: markdown.trim() };
  }

  const [, frontMatterBlock, body] = frontMatterMatch;
  const data: Record<string, string | Record<string, string>> = {};
  const lines = frontMatterBlock.split("\n");

  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    const trimmed = line.trim();

    if (!trimmed) {
      index += 1;
      continue;
    }

    const [rawKey, ...valueParts] = trimmed.split(":");
    if (!rawKey) {
      index += 1;
      continue;
    }

    const key = rawKey.trim();
    const value = valueParts.join(":").trim();

    if (!value) {
      // 处理缩进嵌套对象
      index += 1;
      const nested: Record<string, string> = {};

      while (index < lines.length) {
        const nestedLine = lines[index];
        if (!nestedLine.startsWith("  ")) {
          break;
        }

        const nestedTrimmed = nestedLine.trim();
        if (!nestedTrimmed) {
          index += 1;
          continue;
        }

        const [nestedKey, ...nestedValueParts] = nestedTrimmed.split(":");
        if (nestedKey) {
          nested[nestedKey.trim()] = nestedValueParts.join(":").trim();
        }

        index += 1;
      }

      data[key] = nested;
      continue;
    }

    if (value.startsWith("{") || value.startsWith("[")) {
      try {
        data[key] = JSON.parse(value);
      } catch {
        data[key] = value;
      }
      index += 1;
      continue;
    }

    const sanitizedValue = value.replace(/^['"]|['"]$/g, "");
    data[key] = sanitizedValue;
    index += 1;
  }

  return { data, content: body.trim() };
}

function readMarkdownCollection(baseDir: string): Post[] {
  if (!fs.existsSync(baseDir)) {
    return [];
  }

  const directories = fs
    .readdirSync(baseDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory());

  const posts = directories
    .map((dir) => {
      const folderPath = path.join(baseDir, dir.name);
      const markdownFile = fs
        .readdirSync(folderPath)
        .find((file) => file.endsWith(".md") || file.endsWith(".mdx"));

      if (!markdownFile) {
        return null;
      }

      const fileContent = fs.readFileSync(path.join(folderPath, markdownFile), "utf-8");
      const { data, content } = parseFrontMatter(fileContent);

      const post: Post = {
        slug: dir.name,
        title: typeof data.title === "string" ? data.title : dir.name,
        content,
      };

      if (typeof data.subtitle === "string") {
        post.subtitle = data.subtitle;
      }
      if (typeof data.date === "string") {
        post.date = data.date;
      }
      if (typeof data.image === "string") {
        post.image = data.image;
      }
      if (typeof data.summary === "string") {
        post.summary = data.summary;
      }
      if (typeof data.links === "object" && data.links !== null) {
        post.links = data.links as Record<string, string>;
      }

      return post;
    })
    .filter((post) => post !== null)
    .sort((a, b) => {
      if (!a.date || !b.date) return 0;
      return new Date(b.date).getTime() - new Date(a.date).getTime();
    });

  return posts as Post[];
}

export function getPosts(): Post[] {
  return readMarkdownCollection(POSTS_DIR);
}

export function getNotes(): Post[] {
  const notesDir = path.join(CONTENT_DIR, "notes");
  return readMarkdownCollection(notesDir);
}


