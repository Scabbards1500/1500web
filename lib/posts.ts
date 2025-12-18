import fs from "fs";
import path from "path";

export type PostMeta = {
  title: string;
  subtitle?: string;
  date?: string;
  period?: string;
  image?: string;
  summary?: string;
  author?: string;
  keywords?: string[];
  links?: Record<string, string>;
  highlights?: string[];
  status?: string;
};

export type Post = PostMeta & {
  slug: string;
  content: string;
  contentType?: string;
};

const CONTENT_DIR = path.join(process.cwd(), "content");


type FrontMatterResult = {
  data: Record<string, string | Record<string, string> | string[]>;
  content: string;
};

function parseFrontMatter(markdown: string): FrontMatterResult {
  const frontMatterMatch = markdown.match(/^---\s*([\s\S]*?)---\s*([\s\S]*)$/);

  if (!frontMatterMatch) {
    return { data: {}, content: markdown.trim() };
  }

  const [, frontMatterBlock, body] = frontMatterMatch;
  const data: Record<string, string | Record<string, string> | string[]> = {};
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
      // 检查是否是数组（以 - 开头）
      index += 1;
      if (index < lines.length) {
        const firstNestedLine = lines[index];
        const firstTrimmed = firstNestedLine.trim();
        if (firstTrimmed.startsWith("-")) {
          // 处理数组
          const array: string[] = [];
          while (index < lines.length) {
            const arrayLine = lines[index];
            const arrayTrimmed = arrayLine.trim();
            
            // 如果下一行不是缩进的数组项，则停止
            if (!arrayLine.startsWith("  ") && !arrayTrimmed.startsWith("-")) {
              break;
            }

            if (arrayTrimmed.startsWith("-")) {
              const arrayValue = arrayTrimmed.substring(1).trim();
              if (arrayValue) {
                array.push(arrayValue.replace(/^['"]|['"]$/g, ""));
              }
            }

            index += 1;
          }
          data[key] = array;
          continue;
        }
      }

      // 处理缩进嵌套对象
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
      if (typeof data.author === "string") {
        post.author = data.author;
      }
      if (typeof data.period === "string") {
        post.period = data.period;
      }
      if (typeof data.keywords === "string") {
        // 支持分号分隔的字符串格式
        post.keywords = data.keywords.split(";").map((k) => k.trim()).filter(Boolean);
      } else if (Array.isArray(data.keywords)) {
        post.keywords = data.keywords as string[];
      }
      if (Array.isArray(data.highlights)) {
        post.highlights = data.highlights as string[];
      }
      if (typeof data.links === "object" && data.links !== null) {
        post.links = data.links as Record<string, string>;
      }
      if (typeof data.status === "string") {
        post.status = data.status;
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
  const POSTS_DIR = path.join(CONTENT_DIR, "research");
  const posts = readMarkdownCollection(POSTS_DIR);
  return posts.map(post => ({ ...post, contentType: "research" }));
}

export function getNotes(): Post[] {
  const notesDir = path.join(CONTENT_DIR, "notes");
  const posts = readMarkdownCollection(notesDir);
  return posts.map(post => ({ ...post, contentType: "notes" }));
}

export function getInternships(): Post[] {
  const internshipsDir = path.join(CONTENT_DIR, "internship");
  const posts = readMarkdownCollection(internshipsDir);
  return posts.map(post => ({ ...post, contentType: "internship" }));
}

export function getAwards(): Post[] {
  const awardsDir = path.join(CONTENT_DIR, "awards");
  const posts = readMarkdownCollection(awardsDir);
  return posts.map(post => ({ ...post, contentType: "awards" }));
}

export function getAllPosts(): Post[] {
  const allPosts = [
    ...getPosts(),
    ...getInternships(),
    ...getAwards(),
    ...getNotes(),
  ];
  
  // 按日期排序（最新的在前）
  return allPosts.sort((a, b) => {
    if (!a.date || !b.date) return 0;
    return new Date(b.date).getTime() - new Date(a.date).getTime();
  });
}


